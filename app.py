import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="CIBIL Company Credit Report Extractor", layout="wide")

st.title("🏢 CIBIL Company Credit Report Extractor")
st.write("Upload a CIBIL Company Credit Report PDF and download structured data as Excel/CSV.")

# --------------------------
# Helpers
# --------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def clean_money(value: str) -> str:
    """
    Keep numeric money values clean.
    Supports formats like:
      ₹ 1,15,50,057
      ₹39,40,994
      ₹ 87.9Lac (kept as-is if it contains Lac/Cr)
    """
    if not value:
        return ""
    v = value.strip()
    # If Lac/Cr present, keep human format (don’t destroy meaning)
    if re.search(r"\b(Lac|Cr)\b", v, flags=re.IGNORECASE):
        return v.replace("₹", "").strip()
    # Otherwise strip ₹, commas, spaces
    v = re.sub(r"[₹, ]", "", v)
    return v.strip()

def extract_one(pattern: str, text: str, flags=re.IGNORECASE | re.DOTALL) -> str:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""

def extract_date_like(pattern_label: str, text: str) -> str:
    """
    Matches dates like:
      05/01/2026
      31 Jul 2021
      31 Jul, 2021
      15 Dec 2025
    """
    pattern = rf"{pattern_label}\s*[:]*\s*([0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}|[0-9]{{2}}\s+[A-Za-z]{{3}},?\s+[0-9]{{4}}|-)"
    return extract_one(pattern, text)

def safe_int(s: str):
    try:
        return int(s)
    except:
        return None

# --------------------------
# Parsing: Report Summary / Adverse Info / Enquiries
# --------------------------
def parse_report_meta(full_text: str) -> dict:
    # Company name often appears as: "Dhruv Containers Pvt Ltd | Report Order Number : W-..."
    company_name = extract_one(r"^\s*([A-Za-z0-9 &()./-]+)\s*\|\s*Report Order Number\s*:\s*(W-[A-Za-z0-9-]+)", full_text, flags=re.MULTILINE)
    # If regex above returns only company, fallback
    if company_name:
        # regex returns only company; get order number separately
        pass

    meta = {
        "Company Name": extract_one(r"^\s*([A-Za-z0-9 &()./-]+)\s*\|\s*Report Order Number\s*:", full_text, flags=re.MULTILINE),
        "Report Order Number": extract_one(r"Report Order Number\s*:\s*(W-[A-Za-z0-9-]+)", full_text),
        "Report Order Date": extract_one(r"Report Order Date\s*:\s*([0-9]{2}/[0-9]{2}/[0-9]{4})", full_text),
        "CIBIL Rank (1-10)": extract_one(r"\bYOUR CIBIL RANK\b.*?\n\s*([0-9]{1,2})\s*$", full_text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE),
        "Entity Exposure": clean_money(extract_one(r"CURRENT EXPOSURE\s+Entity Exposure:\s*₹\s*([0-9.]+(?:\s*(?:Cr|Lac))?)", full_text)),
        "Total Exposure": clean_money(extract_one(r"TOTAL EXPOSURE\s*\n\s*₹\s*([0-9.]+(?:\s*(?:Cr|Lac))?)", full_text)),
        "Active Credit Facilities": extract_one(r"\bACTIVE CREDIT\s+FACILITIES\b\s*\n\s*([0-9]+)", full_text),
        "Banks": extract_one(r"\bBANKS\b\s*\n\s*([0-9]+)", full_text),
        "Working Capital Utilisation (%)": extract_one(r"\bWORKING CAPITAL UTILISATION\b\s*\n\s*([0-9]+%)", full_text),
    }
    # Clean empties
    for k, v in list(meta.items()):
        if v == "-" or v is None:
            meta[k] = ""
    return meta

def parse_adverse_summary(full_text: str) -> pd.DataFrame:
    """
    Extracts the Adverse Information block in the Report Summary page.
    We’ll return rows: Category, Value, Count/Notes (if present).
    """
    # Grab chunk after ADVERSE INFORMATION until PAYMENT REGULARITY (structure in this PDF)
    chunk = extract_one(r"ADVERSE INFORMATION.*?\n(.*?)\nPAYMENT REGULARITY", full_text)
    if not chunk:
        return pd.DataFrame(columns=["Category", "Value", "Notes"])

    rows = []
    # Lines like: "SUIT FILED ₹ 1,15,50,057 3 credit facility"
    # Or: "DISHONOURED CHEQUE 12 dishonoured cheque"
    for line in chunk.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match money lines
        m1 = re.match(r"^([A-Z /]+)\s+₹\s*([0-9,]+)\s*(.*)$", line)
        if m1:
            rows.append({
                "Category": m1.group(1).strip(),
                "Value": clean_money(m1.group(2)),
                "Notes": m1.group(3).strip()
            })
            continue

        # Match count-only lines (no ₹)
        m2 = re.match(r"^([A-Z /]+)\s+([0-9]+.*)$", line)
        if m2:
            rows.append({
                "Category": m2.group(1).strip(),
                "Value": "",
                "Notes": m2.group(2).strip()
            })
            continue

        # Fallback raw
        rows.append({"Category": line, "Value": "", "Notes": ""})

    return pd.DataFrame(rows)

def parse_enquiries(full_text: str) -> pd.DataFrame:
    """
    Extracts the enquiries table:
    INSTITUTION NAME DATE OF ENQUIRY CREDIT TYPE ENQUIRY AMOUNT
    """
    # Locate enquiries section
    section = extract_one(
        r"INSTITUTION NAME DATE OF ENQUIRY CREDIT TYPE ENQUIRY AMOUNT\n(.*?)(?:\nYOUR CIBIL RANK|\nENQUIRIES|\nCompany Credit Report)",
        full_text
    )
    if not section:
        return pd.DataFrame(columns=["Institution", "Date of Enquiry", "Credit Type", "Enquiry Amount"])

    rows = []
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Example:
        # Bank of India 28 Nov 2025 Others ₹ 10,000
        m = re.match(r"^(.*?)\s+([0-9]{2}\s+[A-Za-z]{3}\s+[0-9]{4})\s+([A-Za-z ]+)\s+₹\s*([0-9,]+)$", line)
        if m:
            rows.append({
                "Institution": m.group(1).strip(),
                "Date of Enquiry": m.group(2).strip(),
                "Credit Type": m.group(3).strip(),
                "Enquiry Amount": clean_money(m.group(4))
            })
        else:
            # Some PDFs wrap institution names; keep as raw if not matched
            rows.append({
                "Institution": line,
                "Date of Enquiry": "",
                "Credit Type": "",
                "Enquiry Amount": ""
            })

    return pd.DataFrame(rows)

# --------------------------
# Parsing: Credit Facility Details blocks (best structured section)
# --------------------------
def split_facility_blocks(full_text: str):
    """
    Each facility in this report’s details section starts like:
      BANK OF INDIA Demand loan
      A/C: 870020110000856 OPEN
      STANDARD Last Reported
      31 Jul 2021
      SANCTIONED AMOUNT ₹2,124 ...
    We’ll find these headers and slice blocks until next header.
    """
    # Very stable anchor in your PDF:
    # "\n[A-Z ...] <loan type>\nA/C: <...> (OPEN|CLOSED)\n"
    header_re = re.compile(
        r"\n(?P<bank>[A-Z][A-Z0-9 &()./-]+)\s+(?P<loanType>[A-Za-z /()-]+)\nA/C:\s*(?P<account>[^\n]+?)\s+(?P<status>OPEN|CLOSED)\n",
        flags=re.MULTILINE
    )

    matches = list(header_re.finditer(full_text))
    blocks = []
    for i, m in enumerate(matches):
        start = m.start() + 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        blocks.append((m.groupdict(), full_text[start:end].strip()))
    return blocks

def parse_facility_block(header: dict, block: str) -> dict:
    """
    Pull key fields from one facility block.
    """
    bank = header.get("bank", "").strip()
    loan_type = header.get("loanType", "").strip()
    account_raw = header.get("account", "").strip()
    status = header.get("status", "").strip()

    # Some account lines include trailing tokens; keep as-is but also store clean ID
    account_id = account_raw.split()[0].strip()

    # Asset classification line comes right after the A/C header in this report
    asset_class = extract_one(r"\n([A-Z][A-Z -]+)\s+Last Reported", block, flags=re.MULTILINE)

    last_reported = extract_one(r"Last Reported\s*\n\s*([0-9]{2}\s+[A-Za-z]{3},?\s+[0-9]{4})", block)

    # Loan information numeric fields
    sanctioned_amount = clean_money(extract_one(r"SANCTIONED AMOUNT\s+₹\s*([0-9,]+|[0-9.]+\s*(?:Lac|Cr))", block))
    drawing_power = clean_money(extract_one(r"DRAWING POWER\s+₹\s*([0-9,]+|[0-9.]+\s*(?:Lac|Cr))", block))
    installment_amount = clean_money(extract_one(r"INSTALLMENT AMOUNT\s+₹\s*([0-9,]+|-)", block))
    tenure = extract_one(r"\bTENURE\b\s*([0-9]+|-)", block)

    sanctioned_date = extract_date_like(r"SANCTIONED DATE", block)
    outstanding_balance = clean_money(extract_one(r"OUTSTANDING BALANCE\s+₹\s*([0-9,]+|[0-9.]+\s*(?:Lac|Cr)|-)", block))
    repayment_frequency = extract_one(r"REPAYMENT FREQUENCY\s+([A-Za-z]+|Others)", block)
    loan_expiry = extract_one(r"LOAN EXPIRY/MATURITY\s*([0-9]{2}\s+[A-Za-z]{3}\s+[0-9]{4}|-|[0-9]{2}\s+[A-Za-z]{3},\s+[0-9]{4})", block)
    high_credit = clean_money(extract_one(r"HIGH CREDIT\s+([0-9,]+|[0-9.]+\s*(?:Lac|Cr)|-)", block))
    last_repaid = clean_money(extract_one(r"LAST REPAID\s+([0-9,]+|[0-9.]+\s*(?:Lac|Cr)|-)", block))
    loan_renewal = extract_one(r"LOAN RENEWAL\s+([A-Za-z0-9-]+|-)", block)
    restructuring_reason = extract_one(r"RESTRUCTURING REASON\s+([^\n]+)", block)
    guarantee_invocation_date = extract_date_like(r"GUARANTEE INVOCATION DATE\*?", block)

    # Adverse info table in each block (if any)
    # Structure:
    # ADVERSE INFORMATION
    # AMOUNT REPORTED ON TYPE STATUS
    # 1,15,50,000 21 Mar 2023 SUIT FILED SUIT FILED
    adverse_chunk = extract_one(r"ADVERSE INFORMATION\s*\n(.*?)(?:\nSECURITY DETAILS|\nLOAN INFORMATION|Company Credit Report|\Z)", block)
    adverse_items = []
    if adverse_chunk:
        # find rows: amount + date + type + status
        for row in adverse_chunk.split("\n"):
            row = row.strip()
            if not row or row.upper().startswith("AMOUNT"):
                continue
            m = re.match(
                r"^([0-9,]+|-)\s+([0-9]{2}\s+[A-Za-z]{3}\s+[0-9]{4})\s+([A-Z /]+)\s+([A-Z /]+|NA)$",
                row
            )
            if m:
                adverse_items.append(f"{m.group(3).strip()}={m.group(1).strip()} ({m.group(2)}) [{m.group(4).strip()}]")

    # Security Details (often present)
    security_chunk = extract_one(r"SECURITY DETAILS\s*\n(.*?)(?:\nGUARANTOR DETAILS|\nLOAN INFORMATION|Company Credit Report|\Z)", block)
    security_summary = ""
    if security_chunk:
        # Keep first meaningful line(s) to stay deterministic
        lines = [ln.strip() for ln in security_chunk.split("\n") if ln.strip()]
        # if table-like: TYPE CLASSIFICATION VALUE ...
        if len(lines) >= 2 and "TYPE" in lines[0].upper():
            security_summary = " | ".join(lines[1:3])  # take first row-ish
        else:
            security_summary = " | ".join(lines[:2])

    out = {
        "Bank": bank,
        "Loan Type": loan_type,
        "Account Raw": account_raw,
        "Account Number": account_id,
        "Facility Status": status,
        "Asset Classification": asset_class,
        "Last Reported Date": last_reported,
        "Sanctioned Amount": sanctioned_amount,
        "Drawing Power": drawing_power,
        "Installment Amount": installment_amount,
        "Tenure": tenure if tenure != "-" else "",
        "Sanctioned Date": sanctioned_date,
        "Outstanding Balance": outstanding_balance,
        "Repayment Frequency": repayment_frequency,
        "Loan Expiry/Maturity": loan_expiry if loan_expiry != "-" else "",
        "High Credit": high_credit if high_credit != "-" else "",
        "Last Repaid": last_repaid if last_repaid != "-" else "",
        "Loan Renewal": loan_renewal if loan_renewal != "-" else "",
        "Restructuring Reason": restructuring_reason if restructuring_reason != "-" else "",
        "Guarantee Invocation Date": guarantee_invocation_date,
        "Adverse Items": " ; ".join(adverse_items),
        "Security Summary": security_summary,
    }

    # Normalize "-" to empty
    for k, v in list(out.items()):
        if isinstance(v, str) and v.strip() == "-":
            out[k] = ""
    return out

# --------------------------
# Streamlit UI
# --------------------------
uploaded_file = st.file_uploader("Upload CIBIL Company Credit Report PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Parsing PDF..."):
        with pdfplumber.open(uploaded_file) as pdf:
            raw_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        full_text = normalize_text(raw_text)

    # --- Extract tables/sections ---
    meta = parse_report_meta(full_text)
    df_meta = pd.DataFrame([meta])

    df_adverse = parse_adverse_summary(full_text)
    df_enquiries = parse_enquiries(full_text)

    # --- Facilities: best-structured section blocks ---
    facility_blocks = split_facility_blocks(full_text)
    facilities = []
    for header, block in facility_blocks:
        facilities.append(parse_facility_block(header, block))
    df_facilities = pd.DataFrame(facilities)

    # --- Display ---
    st.subheader("Company / Report Meta")
    st.dataframe(df_meta, use_container_width=True)

    st.subheader("Adverse Summary")
    st.dataframe(df_adverse, use_container_width=True)

    st.subheader("Enquiries")
    st.dataframe(df_enquiries, use_container_width=True)

    st.subheader("Credit Facilities (Details)")
    st.caption("One row per facility block in the 'CREDIT FACILITY DETAILS' section.")
    st.dataframe(df_facilities, use_container_width=True)

    # --- Downloads ---
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df_meta.to_excel(writer, index=False, sheet_name="Report_Meta")
        df_adverse.to_excel(writer, index=False, sheet_name="Adverse_Summary")
        df_enquiries.to_excel(writer, index=False, sheet_name="Enquiries")
        df_facilities.to_excel(writer, index=False, sheet_name="Credit_Facilities")

    st.download_button(
        "⬇️ Download Excel",
        xlsx_buffer.getvalue(),
        file_name="cibil_company_report_extracted.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Optional CSV for facilities only
    csv_buffer = BytesIO()
    df_facilities.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Download Credit Facilities CSV",
        csv_buffer.getvalue(),
        file_name="cibil_company_credit_facilities.csv",
        mime="text/csv",
    )
