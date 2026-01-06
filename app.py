import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="CIBIL Company Credit Extractor", layout="wide")
st.title("🏢 CIBIL Company Credit Facilities Extractor")

# --------------------------------------------------
# Normalization
# --------------------------------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    fixes = {
        "NON \nPERFORMING \nASSETS": "NON PERFORMING ASSETS",
        "SUBSTANDARD": "SUB-STANDARD",
        "SUBSTANDARD": "SUB-STANDARD",
        "₹ ": "₹",
    }
    for k, v in fixes.items():
        text = text.replace(k, v)

    return text.strip()

# --------------------------------------------------
# Money normalization
# --------------------------------------------------
def money_to_number(val: str):
    if not val:
        return ""
    v = val.replace("₹", "").replace(",", "").strip()

    if "Cr" in v:
        return int(float(v.replace("Cr", "").strip()) * 1_00_00_000)
    if "Lac" in v:
        return int(float(v.replace("Lac", "").strip()) * 1_00_000)

    try:
        return int(v)
    except:
        return ""

# --------------------------------------------------
# Page-aware PDF extraction
# --------------------------------------------------
def extract_relevant_text(pdf):
    pages_text = []
    for i, page in enumerate(pdf.pages):
        if i >= 4:  # STRICT RULE: start only after page 4
            pages_text.append(page.extract_text() or "")
    return normalize("\n".join(pages_text))

# --------------------------------------------------
# Section extractors
# --------------------------------------------------
def extract_section(text, start, end=None):
    if end:
        m = re.search(rf"{start}(.*?){end}", text, re.DOTALL | re.IGNORECASE)
    else:
        m = re.search(rf"{start}(.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def get_list_section(text):
    return extract_section(
        text,
        "LIST OF CREDIT FACILITIES",
        "LIST OF CREDIT FACILITIES AS GUARANTOR"
    )

def get_details_section(text):
    return extract_section(text, "CREDIT FACILITY DETAILS")

# --------------------------------------------------
# Helpers for classification
# --------------------------------------------------
def detect_facility_status(block_text: str):
    return "CLOSED" if re.search(r"\bCLOSED\b", block_text, re.IGNORECASE) else "OPEN"

def detect_facility_bucket(block_text: str):
    if re.search(r"NON PERFORMING ASSETS|DOUBTFUL|SUB-STANDARD", block_text):
        return "DELINQUENT"
    return "REGULAR"

def detect_adverse_flags(text: str):
    return {
        "Suit Filed": bool(re.search(r"SUIT FILED", text)),
        "Written Off": bool(re.search(r"WRITTEN OFF", text)),
        "Invoked": bool(re.search(r"INVOKED|DEVOLVED", text)),
    }

# --------------------------------------------------
# Facility block parsing (LIST OF CREDIT FACILITIES)
# --------------------------------------------------
def parse_facility_blocks(list_text):
    rows = []
    if not list_text:
        return rows

    list_text = normalize(list_text)
    blocks = re.split(r"\n(?=A/C\s)", list_text)

    for b in blocks:
        if "A/C" not in b:
            continue

        acc = re.search(r"A/C\s+([A-Z0-9/.\-]+)", b)
        if not acc:
            continue
        acc_no = acc.group(1)

        bank = re.search(r"^([A-Za-z &]+(?:\n[A-Za-z &]+)*)", b)
        loan = re.search(r"(Demand loan|Bank guarantee|Cash credit|Short term loan)", b, re.I)
        opened = re.search(r"Opened:\s*([0-9]{1,2}\s+[A-Za-z]{3},?\s+[0-9]{4})", b)
        amount = re.search(r"₹([0-9,]+)", b)
        asset = re.search(
            r"(STANDARD|SUB-STANDARD|DOUBTFUL|NON PERFORMING ASSETS|0 DAY PAST DUE)",
            b
        )
        reported = re.search(r"([0-9]{1,2}\s+[A-Za-z]{3},?\s+[0-9]{4})", b)

        rows.append({
            "Member Name": re.sub(r"\s+", " ", bank.group(1)).strip() if bank else "",
            "Account Type": loan.group(1) if loan else "",
            "Account Number": acc_no,
            "Ownership": "Company",
            "Facility Status": detect_facility_status(b),
            "Facility Bucket": detect_facility_bucket(b),
            "Is Delinquent": detect_facility_bucket(b) == "DELINQUENT",
            "Asset Classification": asset.group(1) if asset else "",
            "Sanctioned Amount (₹)": money_to_number(amount.group(1)) if amount else "",
            "Outstanding Balance (₹)": "",
            "Amount Overdue (₹)": "",
            "Suit Filed": False,
            "Written Off": False,
            "Invoked": False,
            "Date Opened": opened.group(1) if opened else "",
            "Date Closed": "",
            "Date Reported": reported.group(1) if reported else ""
        })

    # Deduplicate by Account Number
    seen = set()
    final = []
    for r in rows:
        if r["Account Number"] not in seen:
            seen.add(r["Account Number"])
            final.append(r)

    return final

# --------------------------------------------------
# Details enrichment (CREDIT FACILITY DETAILS)
# --------------------------------------------------
def enrich_from_details(rows, details_text):
    if not details_text:
        return rows

    details_text = normalize(details_text)
    blocks = re.split(r"\n(?=A/C:\s)", details_text)

    index = {}
    for b in blocks:
        acc = re.search(r"A/C:\s*([A-Z0-9/.\-]+)", b)
        if not acc:
            continue

        index[acc.group(1)] = {
            "outstanding": money_to_number(
                re.search(r"OUTSTANDING BALANCE\s*₹([0-9,]+)", b).group(1)
            ) if re.search(r"OUTSTANDING BALANCE", b) else "",
            "overdue": money_to_number(
                re.search(r"([0-9,]+)\s+[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4}\s+OVERDUE", b).group(1)
            ) if re.search(r"OVERDUE", b) else "",
            **detect_adverse_flags(b)
        }

    for r in rows:
        d = index.get(r["Account Number"])
        if d:
            r["Outstanding Balance (₹)"] = d.get("outstanding", "")
            r["Amount Overdue (₹)"] = d.get("overdue", "")
            r["Suit Filed"] = d.get("Suit Filed", False)
            r["Written Off"] = d.get("Written Off", False)
            r["Invoked"] = d.get("Invoked", False)

    return rows

# --------------------------------------------------
# Risk derivation
# --------------------------------------------------
def derive_risk(row):
    if row["Written Off"]:
        return "VERY HIGH"
    if row["Suit Filed"] or row["Asset Classification"] == "NON PERFORMING ASSETS":
        return "HIGH"
    if row["Facility Bucket"] == "DELINQUENT":
        return "MEDIUM"
    return "LOW"

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        text = extract_relevant_text(pdf)

    rows = parse_facility_blocks(get_list_section(text))
    rows = enrich_from_details(rows, get_details_section(text))

    df = pd.DataFrame(rows)

    df["Risk Level"] = df.apply(derive_risk, axis=1)

    COLUMNS = [
        "Member Name", "Account Type", "Account Number", "Ownership",
        "Facility Status", "Facility Bucket", "Is Delinquent",
        "Asset Classification", "Risk Level",
        "Sanctioned Amount (₹)", "Outstanding Balance (₹)", "Amount Overdue (₹)",
        "Suit Filed", "Written Off", "Invoked",
        "Date Opened", "Date Closed", "Date Reported"
    ]

    df = df.reindex(columns=COLUMNS)

    st.success(f"Extracted {len(df)} credit facilities")
    st.dataframe(df, use_container_width=True)

    out = BytesIO()
    df.to_excel(out, index=False)

    st.download_button(
        "⬇️ Download Excel",
        out.getvalue(),
        "cibil_company_credit_facilities.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
