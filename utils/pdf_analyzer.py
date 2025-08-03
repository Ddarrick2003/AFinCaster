import re
from PyPDF2 import PdfReader

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_financial_summary(text: str) -> dict:
    categories = {
        "Revenue": ["total revenue", "revenues"],
        "Net Income": ["net income", "profit after tax"],
        "Assets": ["total assets", "consolidated assets"],
        "Equity": ["shareholder equity", "equity attributable"],
        "Cash Flow": ["operating cash flow", "net cash from operations"],
        "EPS": ["earnings per share", "basic eps"]
    }

    summary = {}
    for key, keywords in categories.items():
        for kw in keywords:
            if kw.lower() in text.lower():
                idx = text.lower().find(kw.lower())
                snippet = text[idx:idx+100]
                match = re.search(r"[\$€£]?\s?[\d.,]+[MB]?", snippet)
                if match:
                    summary[key] = match.group()
                    break
        if key not in summary:
            summary[key] = "N/A"
    return summary
