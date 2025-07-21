import streamlit as st
import matplotlib.pyplot as plt

def convert_currency(value, currency):
    return value * 144 if currency == "KSh" else value

def display_mae_chart(mae):
    st.markdown("#### 📉 Mean Absolute Error (MAE)")
    st.metric("MAE", f"{mae:.2f}")
