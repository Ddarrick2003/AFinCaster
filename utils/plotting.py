import plotly.graph_objects as go

def plot_forecast_chart(forecast_df, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Price'], mode='lines+markers', name=f"{model_name} Forecast"))
    fig.update_layout(title=f"{model_name} Forecasted Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_volatility_chart(forecast_df, volatility):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Price'], name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=volatility, name="Volatility", yaxis="y2"))

    fig.update_layout(
        title="GARCH Forecast with Volatility",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volatility", overlaying="y", side="right"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
