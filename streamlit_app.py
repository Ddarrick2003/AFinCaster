import re
import pytesseract
from PIL import Image
from io import BytesIO
import tempfile
import streamlit as st
import pandas as pd
from datetime import timedelta
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

from utils.helpers import convert_currency, display_mae_chart
from utils.plotting import plot_forecast_chart, plot_volatility_chart
from utils.theme import set_page_config, inject_custom_css
from utils.sentiment import fetch_twitter_sentiment, fetch_news_sentiment
from utils.pdf_extractor import extract_text_from_pdf

from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

from companies import companies

# === Page Setup & Styling ===
set_page_config()
inject_custom_css()

# === Custom CSS for polished look ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #f3f3f3;
    color: #1f1f1f;
}

.block-container {
    padding: 2rem 3rem;
}

h1, h2, h3, h4 {
    font-weight: 700;
    color: #121212;
}

.stButton>button {
    background-color: #121212;
    color: white;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: none;
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    background-color: #2e2e2e;
    transform: scale(1.02);
}

.stRadio > div {
    gap: 1rem;
}

.stDataFrame {
    background-color: white;
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# === Page Title ===
st.markdown("""
<div style='text-align:center; margin-bottom:1rem;'>
    <h1 style='color:#2E8B57;'>📊 MDAnalyst – Intelligent Financial Forecasting & Report Analysis</h1>
    <p style='font-size:18px;'>Configure, Upload & Visualize multi-model forecasts, sentiment & financial reports</p>
</div>
""", unsafe_allow_html=True)


# === NSE Company Selector (Grouped) ===
st.subheader("NSE-Listed Company Selector")
selected_sector = st.selectbox("Choose Sector", list(companies.keys()), key="sector_select")
company_options = companies[selected_sector]
company_names = [company["name"] for company in company_options]
selected_company_name = st.selectbox("Choose Company", company_names, key="company_select")
selected_company_symbol = next((c["symbol"] for c in company_options if c["name"] == selected_company_name), None)
st.markdown(f"✅ **Selected Company Symbol:** `{selected_company_symbol}`")


# === Financial Report Upload & Analysis (Grouped) ===
st.subheader("📄 Upload Financial Report (PDF)")

uploaded_file_main = st.file_uploader(
    "Upload Financial Report",
    type=["pdf"],
    key="financial_report_main"
)

uploaded_file_2 = st.file_uploader(
    "Upload Another Financial Report for Comparison (Optional)",
    type=["pdf"],
    key="financial_report_secondary"
)

def safe_extract_text(file):
    try:
        return extract_text_from_pdf(file)
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def extract_financial_metrics(text):
    metrics_patterns = {
        "Revenue (KES)": r"(?:Revenue|Sales|Turnover)[^\d]*(\d[\d,\.]*)",
        "Net Income (KES)": r"(?:Net Income|Profit after tax|Net profit)[^\d]*(\d[\d,\.]*)",
        "EPS (KES)": r"(?:EPS|Earnings per share)[^\d]*(\d[\d,\.]*)",
        "Debt (KES)": r"(?:Debt|Borrowings|Loans)[^\d]*(\d[\d,\.]*)",
        "Cash Flow (KES)": r"(?:Cash Flow from operations|Operating cash flow)[^\d]*(\d[\d,\.]*)",
        "Assets (KES)": r"(?:Total Assets)[^\d]*(\d[\d,\.]*)",
        "Equity (KES)": r"(?:Total Equity|Shareholder’s equity)[^\d]*(\d[\d,\.]*)"
    }
    
    extracted_data = {}
    for metric, pattern in metrics_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_data[metric] = match.group(1).replace(",", "")
        else:
            extracted_data[metric] = "N/A"
    return extracted_data


if uploaded_file_main is not None:
    st.markdown("### Report 1 Analysis")
    with st.spinner("Extracting and analyzing report 1..."):
        text1 = safe_extract_text(uploaded_file_main)
        if text1:
            metrics1 = extract_financial_metrics(text1)
            df_metrics1 = pd.DataFrame(metrics1.items(), columns=["Metric", "Value"])
            st.dataframe(df_metrics1)
            st.markdown("**Insights:**")
            for metric, val in metrics1.items():
                if val != "N/A":
                    st.write(f"✅ {metric}: {val}")

if uploaded_file_2 is not None:
    st.markdown("### Report 2 Analysis")
    with st.spinner("Extracting and analyzing report 2..."):
        text2 = safe_extract_text(uploaded_file_2)
        if text2:
            metrics2 = extract_financial_metrics(text2)
            df_metrics2 = pd.DataFrame(metrics2.items(), columns=["Metric", "Value"])
            st.dataframe(df_metrics2)
            st.markdown("**Insights:**")
            for metric, val in metrics2.items():
                if val != "N/A":
                    st.write(f"✅ {metric}: {val}")


# === Stock CSV Upload and Technical Indicators (Grouped) ===
st.subheader("📈 Upload Historical Stock Data (CSV)")

uploaded_file_csv = st.file_uploader("Upload your stock CSV", type=["csv"], key="stock_csv_uploader")
# =========================
# 🔕 Holiday & Weekend Logic
# =========================
CUSTOM_HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-04-18", "2025-12-25",
])

def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in CUSTOM_HOLIDAYS:
        next_day += pd.Timedelta(days=1)
    return next_day



if uploaded_file_csv:
    df = pd.read_csv(uploaded_file_csv)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date')

    # Clean numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('-', ''), errors='coerce')
            except:
                pass

    # Remove weekends & holidays
    df = df[df['Date'].dt.weekday < 5]  # exclude weekends
    df = df[~df['Date'].isin(CUSTOM_HOLIDAYS)]

    # Calculate Technical Indicators if Close column exists
    if 'Close' in df.columns:
        df['Daily Return'] = df['Close'].pct_change()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    with st.expander("🔍 Preview Data: Date, Close, Daily Return, RSI", expanded=True):
        st.dataframe(df[['Date', 'Close', 'Daily Return', 'RSI']].dropna().tail(10))


# === Forecast Task Configuration (Grouped) ===
st.subheader("⚙️ Configure Forecast Task")

task_name = st.text_input("Task Name", "My Forecast Task", key="task_name")
selected_model = st.selectbox("Select Forecasting Model", ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"], key="forecast_model")
forecast_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=30, value=10, key="forecast_days")
currency = st.radio("Currency", ["KSh", "USD"], horizontal=True, key="currency_select")
run_all = st.checkbox("Run All Models", value=True, key="run_all_models")
auto_clean = st.checkbox("Auto Clean Data (drop NaNs)", value=False, key="auto_clean_data")


# === Sentiment Analysis Section (Grouped) ===
st.subheader("🧠 Sentiment Analysis")

sentiment_symbol = selected_company_name or "Safaricom"
sentiment_run = st.checkbox("Include Sentiments", value=False, key="include_sentiments")

if sentiment_run:
    st.info(f"Fetching sentiment analysis for {sentiment_symbol}...")

    twitter_sentiment = fetch_twitter_sentiment(sentiment_symbol)
    news_sentiment = fetch_news_sentiment(sentiment_symbol)

    st.markdown("### Twitter Sentiment (VADER)")
    st.write(twitter_sentiment)

    st.markdown("### News Sentiment (FinBERT)")
    st.write(news_sentiment)


# =========================
# 📄 Upload CSV Data
# =========================
st.subheader("Historical Price Data")
uploaded_file_csv_2 = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader_2")

export_data = []

if uploaded_file_csv_2:
    try:
        df = pd.read_csv(uploaded_file_csv_2)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df.sort_values(by='Date')

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('-', ''), errors='coerce')
                except:
                    pass

        if auto_clean:
            df.dropna(inplace=True)

        df = df[df['Date'].dt.weekday < 5]
        df = df[~df['Date'].isin(CUSTOM_HOLIDAYS)]

        st.markdown(f"### 🧼 Preview of `{task_name}` Dataset")
        st.dataframe(df.tail(), use_container_width=True)

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

                        # Forecast Price Chart
                        st.markdown("### 📉 Forecasted Price")
                        plot_forecast_chart(forecast_df, model)

                        # Volatility Card & Chart
                        st.markdown("""
                            <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px;
                                        box-shadow:0 4px 14px rgba(0,0,0,0.05); margin-top:2rem;">
                                <h4 style="margin-bottom:0.5rem; color:#121212;">📊 Forecasted Volatility</h4>
                                <p style="font-size:14px; color:#666;">This chart shows the predicted volatility (standard deviation of returns) for each forecasted day using GARCH.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        plot_volatility_chart(forecast_df, volatility_df)


                        # Raw data
                        with st.expander("🔍 Raw Volatility Data"):
                            st.dataframe(volatility_df, use_container_width=True)

                        # Volatility Summary Cards
                        avg_vol = volatility_df['Volatility'].mean()
                        max_vol = volatility_df['Volatility'].max()
                        min_vol = volatility_df['Volatility'].min()

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown(f"""
                                <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px;
                                            box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                    <div style="font-size:14px; color:#888;">Average Volatility</div>
                                    <div style="font-size:22px; font-weight:700;">{avg_vol:.4f}</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                                <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px;
                                            box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                    <div style="font-size:14px; color:#888;">Max Volatility</div>
                                    <div style="font-size:22px; font-weight:700;">{max_vol:.4f}</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                                <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px;
                                            box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                    <div style="font-size:14px; color:#888;">Min Volatility</div>
                                    <div style="font-size:22px; font-weight:700;">{min_vol:.4f}</div>
                                </div>
                            """, unsafe_allow_html=True)

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
                    signal_color = "green" if "BUY" in signal else "red" if "SELL" in signal else "orange"

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                            <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px; box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                <div style="font-size:14px; color:#888;">Next Trading Day</div>
                                <div style="font-size:22px; font-weight:700;">{next_trading_day.strftime('%b %d, %Y')}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px; box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                <div style="font-size:14px; color:#888;">Forecasted Price</div>
                                <div style="font-size:22px; font-weight:700;">{currency} {next_price:,.2f}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                            <div style="background-color:#ffffff; padding:1.5rem; border-radius:20px; box-shadow:0 4px 14px rgba(0,0,0,0.05); text-align:center;">
                                <div style="font-size:14px; color:#888;">Forecast Signal</div>
                                <div style="font-size:20px; font-weight:700; color:{signal_color};">{signal}</div>
                                <div style="font-size:13px; color:#666;">{direction} of {currency} {abs(change):,.2f} ({percent:.2f}%)</div>
                            </div>
                        """, unsafe_allow_html=True)

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
        # 📊 Sentiment Analysis Section
        # =========================
        if sentiment_run:
            st.markdown("---\n### 🗾️ Sentiment Analysis Summary")
            twitter_df = fetch_twitter_sentiment(sentiment_symbol)
            news_df = fetch_news_sentiment(sentiment_symbol)

            st.markdown("#### 🔦 Twitter Sentiment")
            st.dataframe(twitter_df)
            st.markdown("#### 📰 News Sentiment")
            st.dataframe(news_df)

        # =========================
        # 📄 Export Forecast Summary
        # =========================
        if export_data:
            st.markdown("### 📄 Export Summary")
            export_df = pd.DataFrame(export_data)
            st.dataframe(export_df)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("📅 Download Forecast Summary as CSV", csv, file_name="forecast_summary.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Data processing error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
