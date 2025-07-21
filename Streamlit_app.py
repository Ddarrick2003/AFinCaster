# =========================
# 📁 FILE: streamlit_app.py
# =========================

import streamlit as st
import pandas as pd
from utils.helpers import convert_currency, display_mae_chart
from utils.plotting import plot_forecast_chart, plot_volatility_chart
from utils.theme import set_page_config, inject_custom_css

from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

# Page setup
set_page_config()
inject_custom_css()

# Hide the Streamlit default menu and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2E8B57;'>📊 FinCaster – Intelligent Financial Forecasting</h1>
        <p style='font-size: 18px;'>Configure, Upload & Visualize multi-model forecasts with confidence intervals.</p>
    </div>
""", unsafe_allow_html=True)

# Task Configuration Section
with st.expander("📝 Configure Analysis Task", expanded=True):
    task_name = st.text_input("Task Name", "My Forecast Task")
    selected_model = st.selectbox("Select Forecasting Model", ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"])
    forecast_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=30, value=10)
    currency = st.radio("Currency", ["KSh", "USD"], horizontal=True)
    run_all = st.checkbox("Run All Models", value=True)
    auto_clean = st.checkbox("Auto Clean Data (drop NaNs)", value=False)

# Upload data
st.subheader("📤 Upload Historical Price Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        if auto_clean:
            df.dropna(inplace=True)

        st.markdown(f"### 🧾 Preview of `{task_name}` Dataset")
        st.dataframe(df.tail(), use_container_width=True)

        # Run selected or all models
        with st.spinner("Running forecast(s)..."):
            models_to_run = [selected_model] if not run_all else ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"]

            for model in models_to_run:
                st.markdown(f"### 🔮 {model} Forecast")
                if model == "LSTM":
                    forecast, mae = run_lstm_forecast(df, forecast_days, currency)
                    plot_forecast_chart(forecast, model)
                    display_mae_chart(mae)
                elif model == "GARCH":
                    forecast, volatility = run_garch_forecast(df, forecast_days, currency)
                    plot_volatility_chart(forecast, volatility)
                elif model == "XGBoost":
                    forecast, mae, shap_plot = run_xgboost_with_shap(df, forecast_days, currency)
                    plot_forecast_chart(forecast, model)
                    display_mae_chart(mae)
                    st.pyplot(shap_plot)
                elif model == "Informer":
                    forecast = run_informer(df, forecast_days, currency)
                    plot_forecast_chart(forecast, model)
                elif model == "Autoformer":
                    forecast = run_autoformer(df, forecast_days, currency)
                    plot_forecast_chart(forecast, model)

    except Exception as e:
        st.error(f"Data processing error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
