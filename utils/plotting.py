# 📁 utils/plotting.py

import streamlit as st
import plotly.graph_objects as go

def plot_forecast_chart(forecast_df, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'],
                             mode='lines+markers', name='Forecast', line=dict(color='green')))
    fig.update_layout(title=f"{model_name} Forecast", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_volatility_chart(forecast_df, volatility):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'],
                             mode='lines+markers', name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=volatility,
                             mode='lines+markers', name='Volatility', line=dict(color='red')))
    fig.update_layout(title="GARCH Forecast & Volatility", xaxis_title="Date", yaxis_title="Value",
                      template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)
