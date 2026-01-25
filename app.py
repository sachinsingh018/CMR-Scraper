import os
import re
import json
import requests
import pdfplumber
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal, Dict, Any

# -----------------------------
# Config
# -----------------------------
GEMINI_API_KEY = "AIzaSyB4iljwvtSPEF300Okg0cpRwOxj_f9LBZ8"
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env var")

# Pick a Gemini model you have access to
# Options often include: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, gemini-2.5-flash
MODEL = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

# -----------------------------
# Output schema (strict-ish)
# -----------------------------
class Facility(BaseModel):
    bank: Optional[str] = None
    lender: Optional[str] = None  # sometimes lender != bank
    facility_type: Optional[str] = None  # e.g., Demand loan, Bank guarantee, Cash credit
    account_number: Optional[str] = None  # the "A/C: ...." identifier
    status_open_closed: Optional[Literal["OPEN", "CLOSED"]] = None
    asset_classification: Optional[str] = None  # STANDARD / DOUBTFUL / SUB-STANDARD / NPA etc.
    sanctioned_amount_inr: Optional[float] = None
    drawing_power_inr: Optional[float] = None
    outstanding_balance_inr: Optional[float] = None
    last_reported_date: Optional[str] = None  # keep as string; parse later if needed
    sanctioned_date: Optional[str] = None

    # Useful flags
    is_delinquent: Optional[bool] = None
    adverse_information: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class ExtractionResult(BaseModel):
    company_name: Optional[str] = None
    report_order_number: Optional[str] = None
    report_order_date: Optional[str] = None
    facilities: List[Facility] = Field(default_factory=list)

# -----------------------------
# Helpers
# -----------------------------
RUPEE = "₹"

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # normalize weird hyphens, whitespace
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def parse_inr_amount(raw: str) -> Optional[float]:
    """
    Convert strings like:
      '₹3.1 Cr', '₹ 40.3 Lac', '₹2,24,49,387', '40,28,020', '9490.0'
    into numeric INR.
    """
    if not raw:
        return None
    t = raw.replace(RUPEE, "").replace(",", "").strip()
    # handle "Cr"/"Lac"
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(Cr|Lac|Lakhs|Lakh)?$", t, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "cr":
        return val * 10_000_000
    if unit in ("lac", "lakh", "lakhs"):
        return val * 100_000
    return val

def split_account_blocks(full_text: str) -> List[str]:
    """
    CIBIL facility sections reliably contain "A/C:" anchors.
    We'll split by occurrences of 'A/C:' and keep header context.
    """
    text = normalize_text(full_text)

    # Keep some header context before the first A/C by injecting a marker
    parts = re.split(r"(?=A/C:\s*)", text, flags=re.IGNORECASE)
    # Filter: keep only chunks that actually look like a facility
    blocks = []
    for p in parts:
        if re.search(r"\bA/C:\s*", p, re.IGNORECASE) and (len(p) > 150):
            blocks.append(p.strip())
    return blocks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def gemini_extract(block: str, global_header: Dict[str, str]) -> ExtractionResult:
    """
    Send a single block to Gemini and ask for STRICT JSON matching ExtractionResult.
    We pass global header so the model can copy company/order fields consistently.
    """

    system_instructions = f"""
You are extracting structured data from an Indian Company Credit Report (CIBIL).
Return ONLY valid JSON. No markdown. No commentary.

Rules:
- Output must match this schema:
{{
  "company_name": string|null,
  "report_order_number": string|null,
  "report_order_date": string|null,
  "facilities": [
    {{
      "bank": string|null,
      "lender": string|null,
      "facility_type": string|null,
      "account_number": string|null,
      "status_open_closed": "OPEN"|"CLOSED"|null,
      "asset_classification": string|null,
      "sanctioned_amount_inr": number|null,
      "drawing_power_inr": number|null,
      "outstanding_balance_inr": number|null,
      "last_reported_date": string|null,
      "sanctioned_date": string|null,
      "is_delinquent": boolean|null,
      "adverse_information": [{{"amount": any, "reported_on": any, "type": any, "status": any}}]
    }}
  ]
}}

- If you see currency like '₹3.1 Cr' or '₹40.3 Lac' convert to INR number.
- If missing, use null (not empty string).
- Pull only facts present in the provided text.
"""

    header_json = json.dumps(global_header, ensure_ascii=False)
    user_prompt = f"""
GLOBAL HEADER (use if present):
{header_json}

FACILITY TEXT BLOCK:
{block}
"""

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system_instructions + "\n\n" + user_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.1,
            "maxOutputTokens": 2048
        }
    }

    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Gemini response text
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

    # Hard cleanup: sometimes models wrap with ```json
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Parse JSON -> validate
    obj = json.loads(text)

    # Convert any string amounts Gemini didn't convert (belt-and-suspenders)
    for f in obj.get("facilities", []):
        for k in ("sanctioned_amount_inr", "drawing_power_inr", "outstanding_balance_inr"):
            if isinstance(f.get(k), str):
                f[k] = parse_inr_amount(f[k])

    try:
        return ExtractionResult.model_validate(obj)
    except ValidationError as ve:
        raise RuntimeError(f"Schema validation failed: {ve}\nRaw:\n{text}")

def extract_global_header(text: str) -> Dict[str, str]:
    """
    Capture company/order fields once from the whole document text.
    """
    t = normalize_text(text)
    # This report has: "Dhruv Containers Private Limited | Report Order Number : W-... | Report Order Date : 19/01/2026"
    company = None
    order_no = None
    order_date = None

    m = re.search(r"^(.*?)\s*\|\s*Report Order Number\s*:\s*([A-Z0-9-]+)\s*\|\s*Report Order Date\s*:\s*([0-9/.-]+)",
                  t, re.IGNORECASE | re.MULTILINE)
    if m:
        company = m.group(1).strip()
        order_no = m.group(2).strip()
        order_date = m.group(3).strip()

    return {
        "company_name": company or None,
        "report_order_number": order_no or None,
        "report_order_date": order_date or None
    }

def extract_text_from_pdf(pdf_path: str) -> str:
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            all_text.append(txt)
    return "\n\n".join(all_text)

def run_llm_extraction(pdf_path: str) -> ExtractionResult:
    raw_text = extract_text_from_pdf(pdf_path)
    header = extract_global_header(raw_text)

    blocks = split_account_blocks(raw_text)

    # If you also want to include the "LIST OF CREDIT FACILITIES" summary page (page 4),
    # you can create extra blocks based on that header — but A/C blocks cover details well.
    merged = ExtractionResult(
        company_name=header.get("company_name"),
        report_order_number=header.get("report_order_number"),
        report_order_date=header.get("report_order_date"),
        facilities=[]
    )

    for b in blocks:
        partial = gemini_extract(b, header)
        merged.facilities.extend(partial.facilities)

    # Deduplicate by account_number (sometimes repeated across pages)
    uniq = {}
    for f in merged.facilities:
        key = (f.account_number or "").strip().upper()
        if not key:
            # fall back on (bank + facility_type + last_reported_date)
            key = f"{f.bank}|{f.facility_type}|{f.last_reported_date}"
        # keep the richer record (more non-nulls)
        score = sum(v is not None and v != [] for v in f.model_dump().values())
        if key not in uniq or score > uniq[key][0]:
            uniq[key] = (score, f)
    merged.facilities = [v[1] for v in uniq.values()]

    return merged

def to_dataframe(result: ExtractionResult) -> pd.DataFrame:
    rows = []
    for f in result.facilities:
        d = f.model_dump()
        d["company_name"] = result.company_name
        d["report_order_number"] = result.report_order_number
        d["report_order_date"] = result.report_order_date
        rows.append(d)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    pdf_path = "/mnt/data/Report_W-565043764 19.01.2026.pdf"
    res = run_llm_extraction(pdf_path)
    df = to_dataframe(res)

    # Save to Excel
    out = "cibil_facilities_extracted.xlsx"
    df.to_excel(out, index=False)
    print(f"Extracted facilities: {len(df)}")
    print(f"Saved: {out}")
