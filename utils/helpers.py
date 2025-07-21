# 📁 utils/helpers.py

import streamlit as st
import plotly.graph_objects as go

def convert_currency(value, currency="KSh"):
    return value * 157 if currency == "KSh" else value

def display_mae_chart(mae):
    st.markdown("**Mean Absolute Error (MAE)**")
    st.metric("MAE", f"{mae:.2f}")
