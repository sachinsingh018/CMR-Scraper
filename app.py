import re
import json
import os
import pdfplumber
import pandas as pd
import streamlit as st
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

# NEW: Gemini official Python SDK
from google import genai


# =============================
# CONFIG
# =============================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets")
    st.stop()

# SDK expects env var by default
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Choose model (you can swap to gemini-2.5-flash if you want)
MODEL = "gemini-2.0-flash"

# Create client once
client = genai.Client()


# =============================
# DATA MODELS
# =============================
class Facility(BaseModel):
    bank: Optional[str] = None
    lender: Optional[str] = None
    facility_type: Optional[str] = None
    account_number: Optional[str] = None
    status_open_closed: Optional[Literal["OPEN", "CLOSED"]] = None
    asset_classification: Optional[str] = None
    sanctioned_amount_inr: Optional[float] = None
    drawing_power_inr: Optional[float] = None
    outstanding_balance_inr: Optional[float] = None
    last_reported_date: Optional[str] = None
    sanctioned_date: Optional[str] = None
    is_delinquent: Optional[bool] = None
    adverse_information: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    company_name: Optional[str] = None
    report_order_number: Optional[str] = None
    report_order_date: Optional[str] = None
    facilities: List[Facility] = Field(default_factory=list)


# =============================
# HELPERS
# =============================
RUPEE = "₹"


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00ad", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_inr_amount(raw: str) -> Optional[float]:
    if not raw:
        return None
    t = raw.replace(RUPEE, "").replace(",", "").strip()
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


def split_account_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    parts = re.split(r"(?=A/C:\s*)", text, flags=re.IGNORECASE)
    return [
        p.strip()
        for p in parts
        if re.search(r"\bA/C:\s*", p, re.IGNORECASE) and len(p) > 150
    ]


def _build_gemini_prompt(block: str, header: Dict[str, str]) -> str:
    # Keep your schema + approach exactly; only change transport layer to SDK.
    system_prompt = """
Return ONLY valid JSON matching this schema. No markdown. No text.

{
  "company_name": string|null,
  "report_order_number": string|null,
  "report_order_date": string|null,
  "facilities": [
    {
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
      "adverse_information": []
    }
  ]
}
""".strip()

    # IMPORTANT: cap the block size to reduce truncation / malformed JSON
    block_capped = block[:12000]

    return (
        system_prompt
        + "\n\nHEADER:\n"
        + json.dumps(header, ensure_ascii=False)
        + "\n\nBLOCK:\n"
        + block_capped
    )



def gemini_extract(block, header):
    prompt = _build_gemini_prompt(block, header)
    st.caption(f"🧠 Prompt size: {len(prompt)} chars")

    # SDK call (matches docs)
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    raw = (resp.text or "").strip()

    st.caption(f"📨 Gemini response length: {len(raw)}")

    if not raw:
        st.error("❌ Gemini returned EMPTY response")


    text = (resp.text or "").strip()
    if not text:
        raise ValueError("Empty response from Gemini")

    # Cleanup if model wraps output
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        st.error("❌ JSON parsing failed")
        st.code(str(e))
        with st.expander("🧠 Invalid JSON from Gemini"):
            st.text(text[:2000])
        raise


    # Preserve your post-processing exactly
    for f in obj.get("facilities", []):
        for k in ("sanctioned_amount_inr", "drawing_power_inr", "outstanding_balance_inr"):
            if isinstance(f.get(k), str):
                f[k] = parse_inr_amount(f[k])

    return ExtractionResult.model_validate(obj)


def extract_global_header(text: str) -> Dict[str, str]:
    m = re.search(
        r"^(.*?)\s*\|\s*Report Order Number\s*:\s*([A-Z0-9-]+)\s*\|\s*Report Order Date\s*:\s*([0-9/.-]+)",
        text,
        re.MULTILINE,
    )
    return {
        "company_name": m.group(1).strip() if m else None,
        "report_order_number": m.group(2).strip() if m else None,
        "report_order_date": m.group(3).strip() if m else None,
    }


def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)

    st.info(f"📄 Extracted {len(text)} characters from PDF")
    if len(text) < 500:
        st.warning("⚠️ Very little text extracted — PDF may be scanned")

    return text


def run_llm_extraction(file) -> ExtractionResult:
    text = extract_text_from_pdf(file)
    header = extract_global_header(text)
    blocks = split_account_blocks(text)

    st.info(f"🧩 Found {len(blocks)} account blocks")

# Optional: show preview of first block
    with st.expander("🔍 Preview first account block"):
        st.text(blocks[0][:1000] if blocks else "No blocks found")


    # merged = ExtractionResult(**header, facilities=[])
    # for i, b in enumerate(blocks):
    #     st.caption(f"🚀 Sending block {i+1}/{len(blocks)} to Gemini ({len(b)} chars)")
    #     partial = gemini_extract(b, header)

    # for b in blocks:
    #     partial = gemini_extract(b, header)
    #     merged.facilities.extend(partial.facilities)
    merged = ExtractionResult(**header, facilities=[])

    TEST_LIMIT = 1
    test_blocks = blocks[:TEST_LIMIT]

    st.warning(f"🧪 TEST MODE: processing only first {len(test_blocks)} blocks")

    for i, b in enumerate(test_blocks):
        st.caption(
            f"🚀 Sending block {i+1}/{len(test_blocks)} to Gemini ({len(b)} chars)"
        )
        partial = gemini_extract(b, header)
        merged.facilities.extend(partial.facilities)


    uniq = {}
    for f in merged.facilities:
        key = (f.account_number or f"{f.bank}|{f.facility_type}|{f.last_reported_date}").upper()
        score = sum(v is not None and v != [] for v in f.model_dump().values())
        if key not in uniq or score > uniq[key][0]:
            uniq[key] = (score, f)

    merged.facilities = [v[1] for v in uniq.values()]
    return merged


def to_dataframe(result: ExtractionResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **f.model_dump(),
                "company_name": result.company_name,
                "report_order_number": result.report_order_number,
                "report_order_date": result.report_order_date,
            }
            for f in result.facilities
        ]
    )


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Facilities")
    buf.seek(0)
    return buf.getvalue()


# =============================
# STREAMLIT UI
# =============================
st.title("📄 CIBIL / CMR LLM Extractor")

uploaded_file = st.file_uploader("Upload CIBIL / CMR PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting using Gemini…"):
        result = run_llm_extraction(uploaded_file)

    df = to_dataframe(result)
    st.success(f"Extracted {len(df)} facilities")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download Excel",
        df_to_excel_bytes(df),
        "cibil_facilities_extracted.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
