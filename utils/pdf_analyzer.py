# utils/pdf_analyzer.py
import fitz  # PyMuPDF
import re
import pandas as pd


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def clean_text(text):
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()


# Utility function to find the first occurrence of a number (KES or %)
def find_value(pattern, text, default=None):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else default


def extract_financial_summary(text):
    text = clean_text(text)
    summary = []

    summary.append({
        "Category": "📈 Revenue",
        "Metric": "Total Revenue",
        "Amount (KES)": find_value(r"total revenue[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "Overall income from all operations"
    })

    summary.append({
        "Category": "💰 Profitability",
        "Metric": "Net Profit After Tax",
        "Amount (KES)": find_value(r"net profit[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "Final earnings after tax deduction"
    })

    summary.append({
        "Category": "🧾 EPS",
        "Metric": "Earnings Per Share (EPS)",
        "Amount (KES)": find_value(r"earnings per share[^\d]+([\d,.]+)", text),
        "Comments": "Profit per outstanding share"
    })

    summary.append({
        "Category": "💳 Cash Flow",
        "Metric": "Net Cash from Operating Activities",
        "Amount (KES)": find_value(r"cash from operating activities[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "Cash generated from core business"
    })

    summary.append({
        "Category": "💵 Dividends",
        "Metric": "Total Dividends Paid",
        "Amount (KES)": find_value(r"dividends paid[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "Total payout to shareholders"
    })

    summary.append({
        "Category": "📉 Debt",
        "Metric": "Total Liabilities",
        "Amount (KES)": find_value(r"total liabilities[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "Obligations owed by the company"
    })

    summary.append({
        "Category": "💼 Assets",
        "Metric": "Total Assets",
        "Amount (KES)": find_value(r"total assets[^\d]+([\d,.]+\s*(?:billion|million)?)", text),
        "Comments": "All owned resources"
    })

    summary.append({
        "Category": "🧮 ROE",
        "Metric": "Return on Equity (ROE)",
        "Amount (KES)": find_value(r"return on equity[^\d]+([\d,.]+%)", text),
        "Comments": "Net income as % of shareholder equity"
    })

    summary.append({
        "Category": "🔐 Solvency Ratio",
        "Metric": "Regulatory Solvency Margin",
        "Amount (KES)": find_value(r"solvency margin[^\d]+([\d,.]+%)", text),
        "Comments": "Financial cushion over liabilities"
    })

    df = pd.DataFrame(summary)
    return df
