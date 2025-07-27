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

import plotly.graph_objects as go

def plot_volatility_chart(forecast_df, volatility_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility_df.index,
        y=volatility_df['Volatility'],
        mode='lines+markers',
        name='Volatility',
        line=dict(color='orange', width=2)
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Volatility",
    )
    st.plotly_chart(fig, use_container_width=True)

