# =========================
# 📁 FILE: streamlit_app.py (Enhanced with Sentiment Analysis + PDF Report Analysis)
# =========================

import streamlit as st
import pandas as pd
from datetime import timedelta
import fitz  # PyMuPDF

from utils.helpers import convert_currency, display_mae_chart
from utils.plotting import plot_forecast_chart, plot_volatility_chart
from utils.theme import set_page_config, inject_custom_css
from utils.sentiment import fetch_twitter_sentiment, fetch_news_sentiment

from model.lstm_model import run_lstm_forecast
from model.garch_model import run_garch_forecast
from model.xgboost_model import run_xgboost_with_shap
from model.transformer_models import run_informer, run_autoformer

from companies import companies

# =========================
# 🔕 Holiday & Weekend Logic
# =========================

CUSTOM_HOLIDAYS = pd.to_datetime([
    "2025-01-01", "2025-04-18", "2025-12-25",
])

def get_next_trading_day(last_date, holidays=CUSTOM_HOLIDAYS):
    next_day = last_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day

# =========================
# 🗄️ Page & Theme Setup
# =========================
set_page_config()
inject_custom_css()

# ✅ NSE Company Selector
st.subheader("🏢 Select NSE-Listed Company")
selected_sector = st.selectbox("Choose Sector", list(companies.keys()))
company_options = companies[selected_sector]
company_names = [company["name"] for company in company_options]
selected_company_name = st.selectbox("Choose Company", company_names)
selected_company_symbol = next((c["symbol"] for c in company_options if c["name"] == selected_company_name), None)
st.markdown(f"✅ **Selected Company Symbol:** `{selected_company_symbol}`")

# ✅ Header
st.markdown("""
    <div style='text-align: center; margin-top: -30px;'>
        <h1 style='color: #2E8B57;'>📊 MDAnalist – Report Analysis</h1>
        <p style='font-size: 18px;'>Configure, Upload & Visualize Financial Report .</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# 📊 Financial Report Summary Dashboard (with commentary)
# =========================

st.subheader("📥 Upload Company Financial Report (PDF)")
pdf_file = st.file_uploader("Upload Financial Report", type=["pdf"], key="pdf_report")

if pdf_file:
    from utils.pdf_analyzer import extract_text_from_pdf, extract_financial_summary

    with st.spinner("🔎 Analyzing report..."):
        text = extract_text_from_pdf(pdf_file)
        summary = extract_financial_summary(text)

    def generate_comment(metric, value):
        if "billion" in value.lower() or "million" in value.lower():
            num = float(''.join(filter(str.isdigit, value)))
        else:
            return "—"

        if metric == "Revenue":
            return "Strong topline performance" if num >= 20 else "Revenue growth potential exists"
        elif metric == "Net Income":
            return "Profitable bottom-line" if num >= 5 else "Below target net profit"
        elif metric == "EPS":
            return "Solid EPS value" if num > 30 else "EPS could improve"
        elif metric == "Cash Flow":
            return "Healthy cash generation" if num >= 3 else "Cash flow under pressure"
        elif metric == "Dividends":
            return "Consistent shareholder return" if num >= 1 else "Low dividend payout"
        elif metric == "Debt":
            return "Debt level manageable" if num < 130 else "Monitor rising debt levels"
        elif metric == "Assets":
            return "Strong asset base" if num > 150 else "Asset base modest"
        elif metric == "ROE":
            return "Excellent ROE" if num > 15 else "ROE may need improvement"
        elif metric == "Solvency":
            return "Above regulatory requirement" if num > 100 else "Solvency ratio needs attention"
        return "—"

    # Render Financial Summary Table
    st.markdown("### 📊 Investor Summary Table")

    financial_table = pd.DataFrame([
        {"Category": "📈 Revenue", "Metric": "Total Revenue", "Amount (KES)": summary.get("Revenue", "N/A"),
         "Comments": generate_comment("Revenue", summary.get("Revenue", "0"))},
        {"Category": "💰 Profitability", "Metric": "Net Profit After Tax", "Amount (KES)": summary.get("Net Income", "N/A"),
         "Comments": generate_comment("Net Income", summary.get("Net Income", "0"))},
        {"Category": "🧾 EPS", "Metric": "Earnings Per Share (EPS)", "Amount (KES)": summary.get("EPS", "N/A"),
         "Comments": generate_comment("EPS", summary.get("EPS", "0"))},
        {"Category": "💳 Cash Flow", "Metric": "Net Cash from Operating Activities", "Amount (KES)": summary.get("Cash Flow", "N/A"),
         "Comments": generate_comment("Cash Flow", summary.get("Cash Flow", "0"))},
        {"Category": "💵 Dividends", "Metric": "Total Dividends Paid", "Amount (KES)": summary.get("Dividends", "N/A"),
         "Comments": generate_comment("Dividends", summary.get("Dividends", "0"))},
        {"Category": "📉 Debt", "Metric": "Total Liabilities", "Amount (KES)": summary.get("Debt", "N/A"),
         "Comments": generate_comment("Debt", summary.get("Debt", "0"))},
        {"Category": "💼 Assets", "Metric": "Total Assets", "Amount (KES)": summary.get("Assets", "N/A"),
         "Comments": generate_comment("Assets", summary.get("Assets", "0"))},
        {"Category": "🧮 ROE", "Metric": "Return on Equity (ROE)", "Amount (KES)": summary.get("ROE", "N/A"),
         "Comments": generate_comment("ROE", summary.get("ROE", "0"))},
        {"Category": "🔐 Solvency Ratio", "Metric": "Regulatory Solvency Margin", "Amount (KES)": summary.get("Solvency", "N/A"),
         "Comments": generate_comment("Solvency", summary.get("Solvency", "0"))},
    ])

    st.dataframe(financial_table, use_container_width=True)






# ✅ Step 1: Inject Custom Modern UI CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space+Grotesk', sans-serif;
        background-color: #f3f3f3;
        color: #1f1f1f;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
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

# Original header block stays here
st.markdown("""
    <div style='text-align: center; margin-top: -30px;'>
        <h1 style='color: #2E8B57;'>📊 MDAnalist – Intelligent Financial Forecasting</h1>
        <p style='font-size: 18px;'>Configure, Upload & Visualize multi-model forecasts with confidence intervals.</p>
    </div>
""", unsafe_allow_html=True)


# ✅ Optional File Upload Handling
uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
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

    df = df[df['Date'].dt.weekday < 5]  # exclude weekends
    df = df[~df['Date'].isin(CUSTOM_HOLIDAYS)]

    # ✅ Add Technical Indicators
    if 'Close' in df.columns:
        df['Daily Return'] = df['Close'].pct_change()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # ✅ Show Expanded Data Preview
    st.markdown("### 🗰️ Enriched Data Preview")
    with st.expander("🔍 Return & RSI (Last 10 Days)", expanded=True):
        st.dataframe(df[['Date', 'Close', 'Daily Return', 'RSI']].dropna().tail(10), use_container_width=True)

# =========================
# 📒️ Task Configuration
# =========================
with st.expander("📜 Configure Analysis Task", expanded=True):
    task_name = st.text_input("Task Name", "My Forecast Task")
    selected_model = st.selectbox("Select Forecasting Model", ["LSTM", "GARCH", "XGBoost", "Informer", "Autoformer"])
    forecast_days = st.slider("Forecast Horizon (days)", min_value=5, max_value=30, value=10)
    currency = st.radio("Currency", ["KSh", "USD"], horizontal=True)
    run_all = st.checkbox("Run All Models", value=True)
    auto_clean = st.checkbox("Auto Clean Data (drop NaNs)", value=False)

# =========================
# 🧠 Optional Sentiment Modules
# =========================
st.subheader("💬 Optional Sentiment Analysis")
sentiment_symbol = selected_company_name or "Safaricom"

sentiment_run = st.checkbox("Include Sentiment Analysis", value=True)

# =========================
# 📄 Upload CSV Data
# =========================
st.subheader("📄 Upload Historical Price Data")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

export_data = []

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
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
