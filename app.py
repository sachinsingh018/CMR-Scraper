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
    """
    CIBIL company reports:
    - Pages 0–3  : cover, invoice, summary, profile
    - Page >= 4  : credit facilities + details
    """
    pages_text = []
    for i, page in enumerate(pdf.pages):
        if i >= 4:  # 👈 HARD RULE
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
# Facility block parsing (LIST OF CREDIT FACILITIES)
# --------------------------------------------------
def parse_facility_blocks(list_text):
    rows = []
    if not list_text:
        return rows

    list_text = normalize(list_text)

    # Split by A/C blocks
    blocks = re.split(r"\n(?=A/C\s)", list_text)

    for b in blocks:
        if "A/C" not in b:
            continue

        acc = re.search(r"A/C\s+([A-Z0-9/.\-]+)", b)
        if not acc:
            continue
        acc_no = acc.group(1)

        bank = re.search(r"^([A-Za-z &\n]+)", b)
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
            "Facility Status": "CLOSED" if "CLOSED CREDIT FACILITIES" in list_text else "OPEN",
            "Is Delinquent": bool(asset and asset.group(1) not in ["STANDARD", "0 DAY PAST DUE"]),
            "Asset Classification": asset.group(1) if asset else "",
            "Sanctioned Amount (₹)": money_to_number(amount.group(1)) if amount else "",
            "Outstanding Balance (₹)": "",
            "Amount Overdue (₹)": "",
            "Date Opened": opened.group(1) if opened else "",
            "Date Closed": "",
            "Date Reported": reported.group(1) if reported else ""
        })

    # Deduplicate
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
            ) if re.search(r"OVERDUE", b) else ""
        }

    for r in rows:
        d = index.get(r["Account Number"])
        if d:
            r["Outstanding Balance (₹)"] = d["outstanding"]
            r["Amount Overdue (₹)"] = d["overdue"]

    return rows

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        text = extract_relevant_text(pdf)

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
