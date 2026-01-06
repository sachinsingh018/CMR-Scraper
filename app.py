import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="CIBIL Company Credit Extractor", layout="wide")
st.title("🏢 CIBIL Company Credit Facilities Extractor")

# ---------------- Helpers ----------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    # Fix odd hyphen/soft-break characters sometimes found in CIBIL extraction
    text = text.replace("SUB￾STANDARD", "SUB-STANDARD")
    text = text.replace("NON \nPERFORMING \nASSETS", "NON PERFORMING ASSETS")
    return text.strip()

def clean_money(val: str) -> str:
    if not val:
        return ""
    # Keep Lac/Cr units as-is if they appear
    if "Lac" in val or "Cr" in val:
        return val.replace("₹", "").strip()
    return re.sub(r"[₹, ]", "", val).strip()

def extract(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def pick_first(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""

def parse_date_any(val: str) -> str:
    """Keep CIBIL's date format as extracted."""
    return (val or "").strip()

# ---------------- Section extraction ----------------
def get_list_section(full_text: str) -> str:
    # Page 5 structure in this report: LIST OF CREDIT FACILITIES ... then next section is "LIST OF CREDIT FACILITIES AS GUARANTOR"
    return extract(r"LIST OF CREDIT FACILITIES(.*?)LIST OF CREDIT FACILITIES AS GUARANTOR", full_text)

def get_details_section(full_text: str) -> str:
    # Details section starts with CREDIT FACILITY DETAILS and continues till end (ratings etc. after)
    # We'll just take from CREDIT FACILITY DETAILS onward to simplify
    return extract(r"CREDIT FACILITY DETAILS(.*)", full_text)

# ---------------- Parse LIST OF CREDIT FACILITIES (table) ----------------
def parse_facilities_from_list_section(list_section: str):
    """
    Tailored for this PDF:
    - Open facilities row includes "A/C ... | Opened: <date> ₹<amt> <asset> <last reported>"
    - Closed facilities row includes "A/C ... | Opened: <date> ₹<amt> <closed date> <asset> [<last reported>]"
    """
    rows = []
    if not list_section:
        return rows

    # Normalize a bit more inside the section
    s = normalize(list_section)

    # We'll parse within each subsection to correctly label Open/Closed + Regular/Delinquent
    # These headings are present on page 5 of this report. :contentReference[oaicite:2]{index=2}
    subsections = [
        ("OPEN", "REGULAR", extract(r"OPEN CREDIT FACILITIES(.*?)DELINQUENT CREDIT FACILITIES", s)),
        ("OPEN", "DELINQUENT", extract(r"DELINQUENT CREDIT FACILITIES(.*?)CLOSED CREDIT FACILITIES", s)),
        ("CLOSED", "REGULAR", extract(r"CLOSED CREDIT FACILITIES(.*?)DELINQUENT CREDIT FACILITIES Total Outstanding:.*?across 4 Credit Facilities", s)),
        ("CLOSED", "DELINQUENT", extract(r"DELINQUENT CREDIT FACILITIES Total Outstanding:.*?across 4 Credit Facilities(.*)", s)),
    ]

    # Row pattern helpers:
    # Bank name appears as lines before loan type, but in extracted text it can be broken:
    # e.g. "Bank Of \nIndia" or "Areion \nFinserve \nPrivate \nLimited" :contentReference[oaicite:3]{index=3}
    bank_block = r"(?P<bank>(?:[A-Za-z][A-Za-z &]+(?:\n[A-Za-z][A-Za-z &]+)*)?)"

    # Loan type in this report
    loan_type = r"(?P<loan>(Demand loan|Bank guarantee|Cash credit|Short term loan \(less than 1 year\)))"

    # Account + opened
    ac_opened = r"A/C\s+(?P<ac>[A-Z0-9/.\-]+)\s*\|\s*Opened:\s*(?P<opened>\d{1,2}\s+[A-Za-z]{3},\s+\d{4})"

    # Sanctioned amount: sometimes "₹2,124" sometimes "₹ 1,03,00,000" on next line
    sanc = r"₹\s*(?P<sanc>[0-9,]+)"

    # Asset classifications appearing in this report (table)
    asset = r"(?P<asset>STANDARD|DOUBTFUL|SUB-STANDARD|NON PERFORMING ASSETS|0 DAY PAST DUE)"

    # Dates in table: "31 Jul, 2021" style
    # We'll accept comma optional
    dt = r"(?P<dt>\d{1,2}\s+[A-Za-z]{3},\s+\d{4})"

    # OPEN row pattern includes last reported date after asset
    open_row_re = re.compile(
        rf"{bank_block}\n{loan_type}\n{ac_opened}\s+{sanc}\s+{asset}\s+{dt}",
        re.IGNORECASE
    )

    # CLOSED row pattern has sanctioned amount then closed date then asset, and sometimes last reported repeats
    closed_row_re = re.compile(
        rf"{bank_block}\n{loan_type}\n{ac_opened}\s+{sanc}\s+(?P<closed>{dt})\s+{asset}(?:\s+(?P<last>{dt}))?",
        re.IGNORECASE
    )

    def norm_bank(b: str) -> str:
        b = (b or "").strip()
        b = re.sub(r"\s*\n\s*", " ", b).strip()
        b = re.sub(r"\s{2,}", " ", b)
        return b

    for status, bucket, chunk in subsections:
        if not chunk:
            continue
        chunk = normalize(chunk)

        if status == "OPEN":
            for m in open_row_re.finditer(chunk):
                bank = norm_bank(m.group("bank"))
                rows.append({
                    "Member Name": bank,
                    "Account Type": m.group("loan").strip(),
                    "Account Number": m.group("ac").strip(),
                    "Ownership": "Company",
                    "Sanctioned Amount (₹)": clean_money(m.group("sanc")),
                    "Current Balance (₹)": "",
                    "Amount Overdue (₹)": "",
                    "Date Opened": parse_date_any(m.group("opened")),
                    "Date Closed": "",
                    "Date Reported": parse_date_any(m.group("dt")),
                    "Status of Loan": m.group("asset").strip(),
                    "Facility Status": status,
                    "Bucket": bucket
                })
        else:
            for m in closed_row_re.finditer(chunk):
                bank = norm_bank(m.group("bank"))
                last = pick_first(m.groupdict().get("last"), m.group("closed"))
                rows.append({
                    "Member Name": bank,
                    "Account Type": m.group("loan").strip(),
                    "Account Number": m.group("ac").strip(),
                    "Ownership": "Company",
                    "Sanctioned Amount (₹)": clean_money(m.group("sanc")),
                    "Current Balance (₹)": "",
                    "Amount Overdue (₹)": "",
                    "Date Opened": parse_date_any(m.group("opened")),
                    "Date Closed": parse_date_any(m.group("closed")),
                    "Date Reported": parse_date_any(last),
                    "Status of Loan": "Closed",
                    "Facility Status": status,
                    "Bucket": bucket
                })

    # De-dup by account no (table parsing can overlap if chunk boundaries shift)
    seen = set()
    deduped = []
    for r in rows:
        key = r.get("Account Number", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped

# ---------------- Parse CREDIT FACILITY DETAILS (pages 7-12) ----------------
def build_details_index(details_text: str):
    """
    Build dict by account number from "CREDIT FACILITY DETAILS" pages.
    This PDF uses:
      BANK ... <loan type>
      A/C: <acc> OPEN/CLOSED
      <asset> Last Reported
      <date>
      SANCTIONED AMOUNT ₹...
      OUTSTANDING BALANCE ₹...
    :contentReference[oaicite:4]{index=4}
    """
    idx = {}
    if not details_text:
        return idx

    t = normalize(details_text)

    # Split into blocks starting at "A/C:"
    blocks = re.split(r"\n(?=[A-Z ]+\s+[A-Za-z].*?\nA/C:\s*)", t)
    # Fallback: if split fails, do a direct finditer scan
    if len(blocks) <= 1:
        blocks = re.split(r"\n(?=A/C:\s*)", t)

    for b in blocks:
        if "A/C:" not in b:
            continue

        acc = extract(r"A/C:\s*([A-Z0-9/.\-]+)", b)
        if not acc:
            continue

        open_closed = extract(rf"A/C:\s*{re.escape(acc)}\s*(OPEN|CLOSED)", b)
        asset = extract(r"\n(STANDARD|DOUBTFUL|SUB-STANDARD|NON PERFORMING ASSETS|0 DAY PAST DUE)\s+Last Reported", b)
        last_reported = extract(r"Last Reported\s*\n\s*([0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})", b)

        sanctioned = clean_money(extract(r"SANCTIONED AMOUNT\s*₹\s*([0-9,]+)", b))
        outstanding = clean_money(extract(r"OUTSTANDING BALANCE\s*₹\s*([0-9,]+)", b))

        # Overdue amount exists in Adverse Info table lines: "<amt> <date> OVERDUE NA"
        overdue = clean_money(extract(r"([0-9,]+)\s+[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4}\s+OVERDUE", b))

        idx[acc] = {
            "detail_status": open_closed,
            "detail_asset": asset,
            "detail_last_reported": last_reported,
            "detail_sanctioned": sanctioned,
            "detail_outstanding": outstanding,
            "detail_overdue": overdue,
            "raw": b
        }

    return idx

def enrich_from_details(rows, details_index):
    for r in rows:
        acc = r.get("Account Number")
        if not acc:
            continue

        d = details_index.get(acc)
        if not d:
            continue

        # Prefer details for sanctioned/outstanding/last reported (more reliable than table)
        r["Sanctioned Amount (₹)"] = pick_first(r.get("Sanctioned Amount (₹)"), d.get("detail_sanctioned"))
        r["Current Balance (₹)"] = pick_first(r.get("Current Balance (₹)"), d.get("detail_outstanding"))
        r["Amount Overdue (₹)"] = pick_first(r.get("Amount Overdue (₹)"), d.get("detail_overdue"), "0")
        r["Date Reported"] = pick_first(d.get("detail_last_reported"), r.get("Date Reported"))

        # If details says CLOSED, treat as closed even if list parser missed closed date
        if (d.get("detail_status") or "").upper() == "CLOSED":
            r["Status of Loan"] = "Closed"
            # If we don't have closed date from table, fall back to last reported
            if not r.get("Date Closed"):
                r["Date Closed"] = d.get("detail_last_reported") or ""

        # Asset class: if closed we keep Closed, else details asset is authoritative
        if r.get("Status of Loan") != "Closed":
            r["Status of Loan"] = pick_first(d.get("detail_asset"), r.get("Status of Loan"))

    return rows

# ---------------- Streamlit UI ----------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)

    text = normalize(raw)

    list_section = get_list_section(text)
    details_section = get_details_section(text)

    rows = parse_facilities_from_list_section(list_section)
    details_index = build_details_index(details_section)
    rows = enrich_from_details(rows, details_index)

    # Output format (keep your original columns first, add status metadata at end)
    df = pd.DataFrame(rows)

    # Reorder columns to match your “flat accounts” style
    desired = [
        "Member Name", "Account Type", "Account Number", "Ownership",
        "Sanctioned Amount (₹)", "Current Balance (₹)", "Amount Overdue (₹)",
        "Date Opened", "Date Closed", "Date Reported", "Status of Loan",
        "Facility Status", "Bucket"
    ]
    df = df[[c for c in desired if c in df.columns]]

    st.success(f"Extracted {len(df)} credit facilities")
    st.dataframe(df, use_container_width=True)

    out = BytesIO()
    df.to_excel(out, index=False)

    st.download_button(
        "⬇️ Download Excel",
        out.getvalue(),
        "company_cibil_flat_accounts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
