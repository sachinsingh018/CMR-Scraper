import streamlit as st
import pdfplumber
import pandas as pd
import re
import io

st.set_page_config(page_title="CIBIL Credit Facility Extractor", layout="wide")
st.title("📄 Company CIBIL – Credit Facility Extractor")

uploaded_file = st.file_uploader("Upload CIBIL PDF", type=["pdf"])


# -----------------------------
# Helper functions
# -----------------------------

def normalize_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def extract(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_line(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""

def clean_amount(val):
    if not val:
        return ""
    return re.sub(r"[₹, ]", "", val)

def extract_bank_and_loan(header_text):
    bank = extract(r"([A-Z ]+BANK(?: LIMITED| LTD)?)", header_text)
    loan = extract(
        r"(Overdraft|Property Loan|Demand loan|GECL Loan|Cash Credit|"
        r"Long term loan|Medium term loan|Short term loan|Equipment financing|HealthCare Finance)",
        header_text
    )
    return bank, loan


# -----------------------------
# Main logic
# -----------------------------

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            raw_text = "\n".join(
                page.extract_text() or ""
                for page in pdf.pages[5:]  # page 6 onwards
            )

        full_text = normalize_text(raw_text)

    st.success(f"Total characters extracted: {len(full_text)}")

    start_match = re.search(r"CREDIT FACILITY DETAILS", full_text, re.IGNORECASE)
    if not start_match:
        st.error("CREDIT FACILITY DETAILS section not found")
        st.stop()

    credit_text = full_text[start_match.start():]
    st.info("Starting extraction from CREDIT FACILITY DETAILS")

    # 🔑 Correct split: one block per account
    blocks = re.split(r"\n(?=A/C:\s*)", credit_text)
    st.caption(f"Account blocks found: {len(blocks)}")

    records = []

    for block in blocks:
        if "A/C" not in block:
            continue

        # Look backward to capture bank + loan header
        prefix = credit_text[:credit_text.find(block)]
        header_lines = prefix.splitlines()[-6:]
        header_text = " ".join(header_lines)

        bank_name, loan_type = extract_bank_and_loan(header_text)

        record = {
            "Bank Name": bank_name,
            "Loan Type": loan_type,
            "Account Number": extract(r"A/C:\s*([A-Z0-9/-]+)", block),
            "Account Status": extract(r"\b(OPEN|CLOSED)\b", block),
            "Asset Classification": extract(
                r"(STANDARD|SUB-STANDARD|DOUBTFUL|NON PERFORMING ASSETS|"
                r"\d+\s*DAY[S]?\s*PAST\s*DUE|0\s*DAY\s*PAST\s*DUE)",
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
                r"Last Reported\s*\n?\s*([0-9A-Za-z ,]+)", block
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
        }

        records.append(record)

    df = pd.DataFrame(records)

    st.subheader("📊 Extracted Credit Facilities")
    st.dataframe(df, use_container_width=True)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer) as writer:
        df.to_excel(writer, index=False, sheet_name="Credit Facilities")

    st.download_button(
        label="⬇️ Download Excel",
        data=excel_buffer.getvalue(),
        file_name="company_cibil_credit_facilities.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
