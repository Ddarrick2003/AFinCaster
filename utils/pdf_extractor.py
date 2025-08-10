import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    text = ""
    try:
        # Try extracting text directly
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # If no text found, try OCR
        if not text.strip():
            file.seek(0)  # reset pointer for OCR
            images = convert_from_bytes(file.read())
            for image in images:
                try:
                    text += pytesseract.image_to_string(image)
                except pytesseract.pytesseract.TesseractNotFoundError:
                    st.error("⚠️ Tesseract OCR is not installed. Please install it to process scanned PDFs.")
                    return ""
    except pytesseract.pytesseract.TesseractNotFoundError:
        st.error("⚠️ Tesseract OCR is not installed. Please install it to process scanned PDFs.")
        return ""
    except Exception as e:
        st.error(f"⚠️ PDF extraction error: {e}")
        return ""

    return text
