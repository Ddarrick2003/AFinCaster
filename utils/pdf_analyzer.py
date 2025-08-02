# utils/pdf_analyzer.py

import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_sentences_by_category(text):
    categories = {
        "Revenue": [],
        "Cash Flow": [],
        "EPS": [],
        "Debt": [],
        "Highlights": []
    }

    # Normalize line breaks
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        for category in categories:
            if category.lower() in sentence.lower():
                categories[category].append(sentence.strip())
    
    return categories
