# 📁 utils/theme.py

import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="MDAnalist-Financial Forecasting",
        layout="wide",
        page_icon="📊"
    )

def inject_custom_css():
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #2E8B57;
            color: white;
        }
        .stRadio > div {
            flex-direction: row;
        }
        </style>
    """, unsafe_allow_html=True)
