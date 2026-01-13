import streamlit as st
import pdfplumber
import pandas as pd
import re
import io

st.set_page_config(page_title="CIBIL Credit Facility Extractor", layout="wide")

st.title("📄 Company CIBIL – Credit Facility Extractor")

uploaded_file = st.file_uploader("Upload CIBIL PDF", type=["pdf"])

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

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        full_text = normalize_text(raw_text)

    st.success(f"Total characters extracted: {len(full_text)}")

    start_match = re.search(r"CREDIT FACILITY DETAILS", full_text, re.IGNORECASE)

    if not start_match:
        st.error("CREDIT FACILITY DETAILS section not found")
        st.stop()

    credit_text = full_text[start_match.start():]
    st.info("Starting extraction from CREDIT FACILITY DETAILS")

    # EXACT same logic as your notebook
    facility_blocks = re.split(
        r"\n(?=[A-Z &]+ (Demand loan|Bank guarantee|Cash credit|Short term loan))",
        credit_text
    )

    raw_blocks = re.split(r"\n(?=A/C:\s*)", credit_text)

    st.caption(f"Facility blocks found: {len(facility_blocks)}")
    st.caption(f"Raw blocks found: {len(raw_blocks)}")

    records = []

    for block in raw_blocks:
        if not block.strip().startswith("A/C:"):
            continue

        record = {
            "Bank Name": extract(r"([A-Z &]{3,})\s+(Demand loan|Bank guarantee|Cash credit|Short term loan)", block),
            "Loan Type": extract(r"(Demand loan|Bank guarantee|Cash credit|Short term loan)", block),
            "Account Number": extract(r"A/C:\s*([A-Z0-9/-]+)", block),
            "Account Status": extract(r"\b(OPEN|CLOSED)\b", block),
            "Asset Classification": extract(
                r"(STANDARD|SUB-STANDARD|DOUBTFUL|NON PERFORMING ASSETS|0 DAY PAST DUE)", block
            ),
            "Sanctioned Amount": clean_amount(
                extract(r"SANCTIONED AMOUNT ₹([0-9,]+)", block)
            ),
            "Outstanding Balance": clean_amount(
                extract(r"OUTSTANDING BALANCE ₹([0-9,]+)", block)
            ),
            "Sanctioned Date": extract(r"SANCTIONED DATE\s*([0-9A-Za-z ]+)", block),
            "Last Reported Date": extract(r"Last Reported\s*([0-9A-Za-z ]+)", block),
            "Repayment Frequency": extract(r"REPAYMENT FREQUENCY\s*([A-Za-z]+)", block),
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
