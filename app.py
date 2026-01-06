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
    text = text.replace("SUB￾STANDARD", "SUB-STANDARD")
    text = text.replace("NON \nPERFORMING \nASSETS", "NON PERFORMING ASSETS")
    return text.strip()

def clean_money(val: str) -> str:
    if not val:
        return ""
    if "Lac" in val or "Cr" in val:
        return val.replace("₹", "").strip()
    return re.sub(r"[₹, ]", "", val).strip()

def extract(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def pick_first(*vals):
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""

# ---------------- Section extraction ----------------
def get_list_section(full_text: str) -> str:
    return extract(
        r"LIST OF CREDIT FACILITIES(.*?)LIST OF CREDIT FACILITIES AS GUARANTOR",
        full_text
    )

def get_details_section(full_text: str) -> str:
    return extract(r"CREDIT FACILITY DETAILS(.*)", full_text)

# ---------------- Parse LIST OF CREDIT FACILITIES ----------------
def parse_facilities_from_list_section(list_section: str):
    rows = []
    if not list_section:
        return rows

    s = normalize(list_section)

    # ✅ STABLE SECTION SPLIT (no hardcoded counts)
    open_chunk = extract(r"OPEN CREDIT FACILITIES(.*?)CLOSED CREDIT FACILITIES", s)
    closed_chunk = extract(r"CLOSED CREDIT FACILITIES(.*)", s)

    sections = [
        ("OPEN", open_chunk),
        ("CLOSED", closed_chunk)
    ]

    bank_block = r"(?P<bank>(?:[A-Za-z][A-Za-z &]+(?:\n[A-Za-z][A-Za-z &]+)*)?)"
    loan_type = r"(?P<loan>(Demand loan|Bank guarantee|Cash credit|Short term loan \(less than 1 year\)))"
    ac_opened = r"A/C\s+(?P<ac>[A-Z0-9/.\-]+)\s*\|\s*Opened:\s*(?P<opened>\d{1,2}\s+[A-Za-z]{3},\s+\d{4})"
    sanc = r"₹\s*(?P<sanc>[0-9,]+)"
    asset = r"(?P<asset>STANDARD|DOUBTFUL|SUB-STANDARD|NON PERFORMING ASSETS|0 DAY PAST DUE)"
    dt = r"\d{1,2}\s+[A-Za-z]{3},\s+\d{4}"

    open_row = re.compile(
        rf"{bank_block}\n{loan_type}\n{ac_opened}\s+{sanc}\s+{asset}\s+(?P<reported>{dt})",
        re.IGNORECASE
    )

    closed_row = re.compile(
        rf"{bank_block}\n{loan_type}\n{ac_opened}\s+{sanc}\s+(?P<closed>{dt})\s+{asset}(?:\s+(?P<reported>{dt}))?",
        re.IGNORECASE
    )

    def norm_bank(b):
        return re.sub(r"\s+", " ", (b or "")).strip()

    for status, chunk in sections:
        if not chunk:
            continue
        chunk = normalize(chunk)

        pattern = open_row if status == "OPEN" else closed_row

        for m in pattern.finditer(chunk):
            rows.append({
                "Member Name": norm_bank(m.group("bank")),
                "Account Type": m.group("loan"),
                "Account Number": m.group("ac"),
                "Ownership": "Company",
                "Sanctioned Amount (₹)": clean_money(m.group("sanc")),
                "Current Balance (₹)": "",
                "Amount Overdue (₹)": "",
                "Date Opened": m.group("opened"),
                "Date Closed": m.groupdict().get("closed", ""),
                "Date Reported": m.groupdict().get("reported", ""),
                "Status of Loan": "Closed" if status == "CLOSED" else m.group("asset"),
                "Facility Status": status
            })

    # De-dup by Account Number
    seen = set()
    final = []
    for r in rows:
        if r["Account Number"] not in seen:
            seen.add(r["Account Number"])
            final.append(r)

    return final

# ---------------- Parse CREDIT FACILITY DETAILS ----------------
def build_details_index(details_text: str):
    idx = {}
    if not details_text:
        return idx

    t = normalize(details_text)
    blocks = re.split(r"\n(?=A/C:\s*)", t)

    for b in blocks:
        acc = extract(r"A/C:\s*([A-Z0-9/.\-]+)", b)
        if not acc:
            continue

        idx[acc] = {
            "status": extract(rf"A/C:\s*{re.escape(acc)}\s*(OPEN|CLOSED)", b),
            "asset": extract(r"\n(STANDARD|DOUBTFUL|SUB-STANDARD|NON PERFORMING ASSETS|0 DAY PAST DUE)\s+Last Reported", b),
            "reported": extract(r"Last Reported\s*\n\s*([0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4})", b),
            "outstanding": clean_money(extract(r"OUTSTANDING BALANCE\s*₹\s*([0-9,]+)", b)),
            "overdue": clean_money(extract(r"([0-9,]+)\s+[0-9]{1,2}\s+[A-Za-z]{3}\s+[0-9]{4}\s+OVERDUE", b))
        }

    return idx

def enrich_from_details(rows, details_index):
    for r in rows:
        d = details_index.get(r["Account Number"])
        if not d:
            continue

        r["Current Balance (₹)"] = pick_first(d.get("outstanding"))
        r["Amount Overdue (₹)"] = pick_first(d.get("overdue"), "0")
        r["Date Reported"] = pick_first(d.get("reported"), r["Date Reported"])

        if d.get("status") == "CLOSED":
            r["Status of Loan"] = "Closed"
            if not r["Date Closed"]:
                r["Date Closed"] = d.get("reported", "")

        if r["Status of Loan"] != "Closed":
            r["Status of Loan"] = pick_first(d.get("asset"), r["Status of Loan"])

    return rows

# ---------------- Streamlit UI ----------------
uploaded = st.file_uploader("Upload Company CIBIL PDF", type=["pdf"])

if uploaded:
    with pdfplumber.open(uploaded) as pdf:
        raw = "\n".join(page.extract_text() or "" for page in pdf.pages)

    text = normalize(raw)

    rows = parse_facilities_from_list_section(get_list_section(text))
    rows = enrich_from_details(rows, build_details_index(get_details_section(text)))

    df = pd.DataFrame(rows)

    df = df[[
        "Member Name", "Account Type", "Account Number", "Ownership",
        "Sanctioned Amount (₹)", "Current Balance (₹)", "Amount Overdue (₹)",
        "Date Opened", "Date Closed", "Date Reported", "Status of Loan",
        "Facility Status"
    ]]

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
