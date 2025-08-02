import fitz  # PyMuPDF
import re
import pandas as pd

CATEGORIES = {
    "Revenue": r"\b(revenue|top line|sales)\b",
    "Cash Flow": r"\b(cash flow|operating activities|financing activities|investing activities)\b",
    "EPS": r"\b(EPS|earnings per share|diluted earnings)\b",
    "Debt": r"\b(debt|liabilities|subordinated debt|credit risk)\b",
    "Highlights": r"\b(highlight|summary|performance milestone)\b"
}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_sentences_by_category(text, categories=CATEGORIES):
    sentences = re.split(r'(?<=[.!?]) +', text)
    data = {cat: [] for cat in categories}

    for sentence in sentences:
        clean = sentence.strip().replace("\n", " ")
        for category, pattern in categories.items():
            if re.search(pattern, clean, flags=re.IGNORECASE):
                if len(clean.split()) >= 5:
                    data[category].append(clean)
                break
    return data
