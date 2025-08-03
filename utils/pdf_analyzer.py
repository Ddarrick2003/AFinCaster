import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pdfplumber
import pandas as pd

# Fallback OCR-based text extraction for scanned PDFs
def extract_text_with_ocr(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
    return text

# Extract text using PDFPlumber for digital PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or '' for page in pdf.pages)
        return text if text.strip() else extract_text_with_ocr(pdf_path)
    except:
        return extract_text_with_ocr(pdf_path)

# Helper to find numeric amounts in KSh
def find_amounts(text, keywords):
    lines = text.splitlines()
    results = []
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in keywords):
            match = re.findall(r"KSh[\s]*[\d,]+(?:\.\d+)?", line)
            if match:
                results.append((line.strip(), match[0]))
    return results

# Extract investor-centric financial summary
def extract_financial_summary(text):
    summary = {}

    summary["Revenue"] = find_amounts(text, ["revenue", "turnover"])
    summary["Net Income"] = find_amounts(text, ["net income", "profit after tax", "earnings"])
    summary["EPS"] = find_amounts(text, ["earnings per share", "eps"])
    summary["Total Assets"] = find_amounts(text, ["total assets"])
    summary["Total Liabilities"] = find_amounts(text, ["total liabilities"])
    summary["Cash Flow"] = find_amounts(text, ["cash flows from operating", "net cash"])
    summary["Debt"] = find_amounts(text, ["debt", "borrowings", "loans"])
    summary["Highlights"] = find_amounts(text, ["highlights", "milestones", "summary"])

    # Convert summary to a tabular DataFrame
    rows = []
    for key, values in summary.items():
        if values:
            for sentence, amount in values:
                rows.append({"Category": key, "Amount": amount, "Context": sentence})
    return pd.DataFrame(rows)
