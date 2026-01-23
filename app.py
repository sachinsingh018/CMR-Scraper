import streamlit as st
import pdfplumber
import pandas as pd
import re
import io

st.set_page_config(page_title="CIBIL Credit Facility Extractor", layout="wide")
st.title("📄 Company CIBIL – Credit Facility Extractor")

uploaded_file = st.file_uploader("Upload CIBIL PDF", type=["pdf"])


# -------------------------
# Helpers
# -------------------------

def normalize_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def extract(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def clean_amount(val):
    if not val:
        return ""
    return re.sub(r"[₹, ]", "", val)

def extract_bank_and_loan(block):
    """
    Bank + Loan Type always appear in the line immediately
    preceding the 'A/C:' line
    """
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        if line.startswith("A/C:") and i > 0:
            header = lines[i - 1]

            # split last word group as loan type, rest as bank
            m = re.match(r"(.+?)\s+(Overdraft|Property Loan|HealthCare Finance|GECL Loan|Auto Loan|"
                         r"Cash Credit|"
                         r"Long term loan .*?|"
                         r"Medium term loan .*?|"
                         r"Short term loan .*?|"
                         r"Equipment financing .*?)$", header)

            if m:
                return m.group(1).strip(), m.group(2).strip()

            return header, ""

    return "", ""


# -------------------------
# Main
# -------------------------

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            raw_text = "\n".join(
                page.extract_text() or ""
                for page in pdf.pages[6:]  # PAGE 7 ONWARDS (0-based)
            )

    full_text = normalize_text(raw_text)
    st.success(f"Characters extracted: {len(full_text)}")

    start = re.search(r"CREDIT FACILITY DETAILS", full_text, re.IGNORECASE)
    if not start:
        st.error("CREDIT FACILITY DETAILS section not found")
        st.stop()

    credit_text = full_text[start.start():]

    # One block per account
    blocks = re.split(r"\n(?=LOAN INFORMATION)", credit_text)
    st.caption(f"Facility blocks found: {len(blocks)}")

    records = []

    for block in blocks:
        if "A/C:" not in block:
            continue

        bank_name, loan_type = extract_bank_and_loan(block)

        records.append({
            "Bank Name": bank_name,
            "Loan Type": loan_type,
            "Account Number": extract(r"A/C:\s*([A-Z0-9/-]+)", block),
            "Account Status": extract(r"\b(OPEN|CLOSED)\b", block),
            "Days Past Due": extract(
                r"(\d+\s*DAY[S]?\s*PAST\s*DUE)",
                block
            ),
            "Asset Classification": extract(
                r"\b(STANDARD|SUB-STANDARD|DOUBTFUL|LOSS|NPA)\b",
                block
            ),

            "Sanctioned Amount": clean_amount(
                extract(r"SANCTIONED AMOUNT ₹([0-9,]+)", block)
            ),
            "Outstanding Balance": clean_amount(
                extract(r"OUTSTANDING BALANCE ₹([0-9,-]+)", block)
            ),
            "Sanctioned Date": extract(
                r"SANCTIONED DATE\s+([0-9A-Za-z ,]+)", block
            ),
            "Last Reported Date": extract(
                r"Last Reported[\s\S]{0,50}?([0-9]{1,2}\s+[A-Za-z]{3},?\s+[0-9]{4})",
                block
            ),
            "Repayment Frequency": extract(
                r"REPAYMENT FREQUENCY\s*([A-Za-z]+)", block
            ),
            "Loan Expiry / Maturity": extract(
                r"LOAN EXPIRY/MATURITY\s*([0-9A-Za-z -]+)", block
            ),
            "Suit Filed": "Yes" if "SUIT FILED" in block.upper() else "No",
            "Written Off": "Yes" if "WRITTEN OFF" in block.upper() else "No",
            "NPA": "Yes" if re.search(r"\bNPA\b", block.upper()) else "No",
        })

    df = pd.DataFrame(records)

    st.subheader("📊 Extracted Credit Facilities")
    st.dataframe(df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        df.to_excel(writer, index=False)

    st.download_button(
        "⬇️ Download Excel",
        buf.getvalue(),
        "company_cibil_credit_facilities.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
