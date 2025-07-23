# =========================
# 📁 FILE: streamlit_app.py
# =========================

import streamlit as st
import pandas as pd
from datetime import timedelta

from utils.helpers import convert_currency, display_mae_chart
from utils.plotting import plot_forecast_chart, plot_volatility_chart
from utils.theme import set_page_config, inject_custom_css

from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

# =========================
# 📅 Holiday & Weekend Logic
# =========================

# Customize this list with local/national stock exchange holidays
CUSTOM_HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-04-18", "2025-12-25",  # Sample holidays
    # Add more as needed
])

def get_next_trading_day(last_date, holidays=CUSTOM_HOLIDAYS):
    next_day = last_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day

# =========================
# Page & Theme Setup
# =========================
set_page_config()
inject_custom_css()

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

# =========================
# Task Configuration
# =========================
with st.expander("📝 Configure Analysis Task", expanded=True):
    task_name = st.text_input("Task Name", "My Forecast Task")
    selected_model = st.selectbox("Select Forecasting Model", ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"])
    forecast_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=30, value=10)
    currency = st.radio("Currency", ["KSh", "USD"], horizontal=True)
    run_all = st.checkbox("Run All Models", value=True)
    auto_clean = st.checkbox("Auto Clean Data (drop NaNs)", value=False)

# =========================
# Upload CSV Data
# =========================
st.subheader("📤 Upload Historical Price Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

export_data = []  # To hold model output for export

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # Convert numeric-like strings
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('-', ''), errors='coerce')
                except:
                    pass

        if auto_clean:
            df.dropna(inplace=True)

        # Filter out weekends and known holidays
        df = df[df['Date'].dt.weekday < 5]  # Remove weekends
        df = df[~df['Date'].isin(CUSTOM_HOLIDAYS)]  # Remove holidays

        st.markdown(f"### 🧾 Preview of `{task_name}` Dataset")
        st.dataframe(df.tail(), use_container_width=True)

        # =========================
        # Forecast Execution
        # =========================
        with st.spinner("Running forecast(s)..."):
            models_to_run = [selected_model] if not run_all else ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"]

            for model in models_to_run:
                st.markdown(f"---\n### 🔮 {model} Forecast")
                with st.container():
                    if model == "LSTM":
                        forecast_df, mae = run_lstm_forecast(df, forecast_days, currency)
                        plot_forecast_chart(forecast_df, model)
                        display_mae_chart(mae)

                    elif model == "GARCH":
                        forecast_df, volatility_df = run_garch_forecast(df, forecast_days, currency)
                        plot_volatility_chart(forecast_df, volatility_df)

                    elif model == "XGBoost":
                        forecast_df, mae, shap_plot = run_xgboost_with_shap(df, forecast_days, currency)
                        plot_forecast_chart(forecast_df, model)
                        display_mae_chart(mae)
                        st.pyplot(shap_plot)

                    elif model == "Informer":
                        forecast_df = run_informer(df, forecast_days, currency)
                        plot_forecast_chart(forecast_df, model)

                    elif model == "Autoformer":
                        forecast_df = run_autoformer(df, forecast_days, currency)
                        plot_forecast_chart(forecast_df, model)

                    # =========================
                    # 🔔 Alert, Signal, Export
                    # =========================
                    last_date = df['Date'].max()
                    next_trading_day = get_next_trading_day(last_date)

                    next_price = forecast_df.iloc[0]['Forecast']

                    if 'Date' not in forecast_df.columns:
                        forecast_df.insert(0, 'Date', pd.NaT)
                    forecast_df.at[0, 'Date'] = next_trading_day

                    last_close = df['Close'].iloc[-1]
                    change = next_price - last_close
                    percent = (change / last_close) * 100
                    direction = "📈 Increase" if change > 0 else "📉 Decrease"
                    signal = "✅ BUY Signal" if percent > 2 else "⚠️ SELL Signal" if percent < -2 else "🟡 HOLD"
                    alert_color = "green" if change > 0 else "red"

                    st.metric(
                        f"📌 {model} Forecast for {next_trading_day.strftime('%a, %b %d')}",
                        f"{currency} {next_price:,.2f}"
                    )
                    st.markdown(
                        f"<span style='color:{alert_color}; font-weight:bold;'>🔔 Alert: {direction} of {currency} {abs(change):,.2f} ({percent:.2f}%)</span><br>"
                        f"<span style='color:{'green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'orange'}; font-weight:bold;'>📢 {signal}</span>",
                        unsafe_allow_html=True
                    )

                    export_data.append({
                        "Model": model,
                        "Forecasted Price": next_price,
                        "Last Price": last_close,
                        "Change": change,
                        "Percent Change": percent,
                        "Direction": direction,
                        "Signal": signal,
                        "Next Trading Day": next_trading_day.strftime('%Y-%m-%d')
                    })

        # =========================
        # 📤 Export Forecast Summary
        # =========================
        if export_data:
            st.markdown("### 📄 Export Summary")
            export_df = pd.DataFrame(export_data)
            st.dataframe(export_df)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast Summary as CSV", csv, file_name="forecast_summary.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Data processing error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
