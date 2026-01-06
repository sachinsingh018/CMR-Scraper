import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="CIBIL Company Credit Extractor", layout="wide")
st.title("🏢 CIBIL Company Credit Facilities Extractor")

# ---------------- Normalization ----------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    replacements = {
        "NON \nPERFORMING \nASSETS": "NON PERFORMING ASSETS",
        "SUBSTANDARD": "SUB-STANDARD",
        "SUBSTANDARD": "SUB-STANDARD",
        "₹ ": "₹",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()

# ---------------- Money normalization ----------------
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
        return v

# ---------------- Section extractors ----------------
def section(text, start, end=None):
    if end:
        return re.search(rf"{start}(.*?){end}", text, re.DOTALL | re.IGNORECASE)
    return re.search(rf"{start}(.*)", text, re.DOTALL | re.IGNORECASE)

def get_list_section(text):
    m = section(text, "LIST OF CREDIT FACILITIES", "LIST OF CREDIT FACILITIES AS GUARANTOR")
    return m.group(1) if m else ""

def get_details_section(text):
    m = section(text, "CREDIT FACILITY DETAILS")
    return m.group(1) if m else ""

# ---------------- Facility block parser ----------------
def parse_facility_blocks(list_text):
    rows = []
    if not list_text:
        return rows

    list_text = normalize(list_text)

    # Split by A/C occurrences
    parts = re.split(r"\n(?=A/C\s)", list_text)

    for p in parts:
        if "A/C" not in p:
            continue

        acc = re.search(r"A/C\s+([A-Z0-9/.\-]+)", p)
        if not acc:
            continue
        acc_no = acc.group(1)

        bank = re.search(r"^([A-Za-z &\n]+)", p)
        loan = re.search(r"(Demand loan|Bank guarantee|Cash credit|Short term loan)", p, re.I)
        opened = re.search(r"Opened:\s*([0-9]{1,2}\s+[A-Za-z]{3},?\s+[0-9]{4})", p)
        amount = re.search(r"₹([0-9,]+)", p)
        asset = re.search(r"(STANDARD|SUB-STANDARD|DOUBTFUL|NON PERFORMING ASSETS|0 DAY PAST DUE)", p)
        reported = re.search(r"([0-9]{1,2}\s+[A-Za-z]{3},?\s+[0-9]{4})", p)

        is_closed = "CLOSED CREDIT FACILITIES" in list_text and acc_no in list_text
        is_delinquent = asset and asset.group(1) not in ["STANDARD", "0 DAY PAST DUE"]

        rows.append({
            "Member Name": re.sub(r"\s+", " ", bank.group(1)).strip() if bank else "",
            "Account Type": loan.group(1) if loan else "",
            "Account Number": acc_no,
            "Ownership": "Company",
            "Facility Status": "CLOSED" if is_closed else "OPEN",
            "Is Delinquent": bool(is_delinquent),
            "Asset Classification": asset.group(1) if asset else "",
            "Sanctioned Amount (₹)": money_to_number(amount.group(1)) if amount else "",
            "Date Opened": opened.group(1) if opened else "",
            "Date Closed": "",
            "Date Reported": reported.group(1) if reported else "",
            "Outstanding Balance (₹)": "",
            "Amount Overdue (₹)": ""
        })

    # De-dup
    seen = set()
    final = []
    for r in rows:
        if r["Account Number"] not in seen:
            seen.add(r["Account Number"])
            final.append(r)

    return final

# ---------------- Details enrichment ----------------
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
                extract := re.search(r"OUTSTANDING BALANCE\s*₹([0-9,]+)", b).group(1)
                if re.search(r"OUTSTANDING BALANCE\s*₹([0-9,]+)", b) else ""
            ),
            "overdue": money_to_number(
                extract := re.search(r"([0-9,]+)\s+[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4}\s+OVERDUE", b).group(1)
                if re.search(r"OVERDUE", b) else ""
            )
        }

    for r in rows:
        d = index.get(r["Account Number"])
        if d:
            r["Outstanding Balance (₹)"] = d.get("outstanding", "")
            r["Amount Overdue (₹)"] = d.get("overdue", "")

    return rows

# ---------------- UI ----------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)

    text = normalize(raw)

    rows = parse_facility_blocks(get_list_section(text))
    rows = enrich_from_details(rows, get_details_section(text))

    COLUMNS = [
        "Member Name", "Account Type", "Account Number", "Ownership",
        "Facility Status", "Is Delinquent", "Asset Classification",
        "Sanctioned Amount (₹)", "Outstanding Balance (₹)", "Amount Overdue (₹)",
        "Date Opened", "Date Closed", "Date Reported"
    ]

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)

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
