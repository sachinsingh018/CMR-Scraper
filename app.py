import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="CIBIL Company Credit Extractor", layout="wide")
st.title("🏢 CIBIL Company Credit Facilities Extractor")

# ---------------- Helpers ----------------
def normalize(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def clean_money(val):
    if not val:
        return ""
    if "Lac" in val or "Cr" in val:
        return val.replace("₹", "").strip()
    return re.sub(r"[₹, ]", "", val)

def extract(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

# ---------------- Core parsing ----------------
def parse_list_of_credit_facilities(full_text):
    """
    Parses LIST OF CREDIT FACILITIES section into flat rows
    """
    rows = []

    # Cut only this section
    section = extract(
        r"LIST OF CREDIT FACILITIES(.*?)LIST OF CREDIT FACILITIES AS GUARANTOR",
        full_text
    )

    if not section:
        return rows

    # Each account line always contains: A/C XXXXX | Opened:
    account_blocks = re.split(r"\n(?=[A-Z][A-Z &]+?\n[A-Za-z ]+\nA/C )", section)

    for block in account_blocks:
        if "A/C" not in block:
            continue

        bank = extract(r"^([A-Z][A-Z &]+)", block)
        loan_type = extract(r"\n([A-Za-z ]+)\nA/C", block)
        account_no = extract(r"A/C\s*([A-Z0-9/-]+)", block)

        opened = extract(r"Opened:\s*([0-9]{1,2} [A-Za-z]{3}, [0-9]{4})", block)
        sanctioned = clean_money(extract(r"₹\s*([0-9,]+)", block))
        asset_class = extract(r"\b(STANDARD|SUBSTANDARD|DOUBTFUL|NPA|LOSS)", block)
        last_reported = extract(r"([0-9]{1,2} [A-Za-z]{3}, [0-9]{4})", block)

        rows.append({
            "Member Name": bank,
            "Account Type": loan_type,
            "Account Number": account_no,
            "Ownership": "Company",
            "Sanctioned Amount (₹)": sanctioned,
            "Current Balance (₹)": "",
            "Amount Overdue (₹)": "",
            "Date Opened": opened,
            "Date Closed": "",
            "Date Reported": last_reported,
            "Status of Loan": asset_class
        })

    return rows

def enrich_from_details(rows, full_text):
    """
    Fills Current Balance, Overdue, Closed Date from CREDIT FACILITY DETAILS
    """
    for r in rows:
        acc = r["Account Number"]
        if not acc:
            continue

        block = extract(
            rf"A/C:\s*{re.escape(acc)}.*?(?=\n[A-Z][A-Z ]+ (Demand|Bank|Credit)|\Z)",
            full_text
        )

        if not block:
            continue

        r["Current Balance (₹)"] = clean_money(
            extract(r"OUTSTANDING BALANCE\s*₹\s*([0-9,]+)", block)
        )

        overdue = extract(r"OVERDUE\s+([0-9,]+)", block)
        r["Amount Overdue (₹)"] = clean_money(overdue) if overdue else "0"

        closed = extract(r"Date Closed\s*\n\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", block)
        if closed:
            r["Date Closed"] = closed
            r["Status of Loan"] = "Closed"

    return rows

# ---------------- Streamlit UI ----------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)

    text = normalize(raw)

    rows = parse_list_of_credit_facilities(text)
    rows = enrich_from_details(rows, text)

    df = pd.DataFrame(rows)

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
