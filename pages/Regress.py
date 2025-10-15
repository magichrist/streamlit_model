import streamlit as st
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import plotly.graph_objects as go
import ccxt
from warnings import filterwarnings

filterwarnings('ignore')

st.set_page_config(
    page_title="Regression",
    layout="centered",
)

models = ["Linear Regression", "Support Vector Regression",
          "Decision Tree Regression", "KNeighbors Regression",
          "Extra Trees Regression", "HistGradientBoosting Regression",
          "MLPRegressor", "Ridge", "ElasticNet"]

st.cache_data(show_spinner="Loading data ...")

exchanges_list=[
    "binance","bingx","kraken","kucoin","bybit","bitget","mexc","okx","coinex"
]

def ticker():
    exchange_choice=st.session_state.exchange
    try:
        exchange=getattr(ccxt,exchange_choice)
        exchange=exchange()
        data = exchange.fetch_ohlcv(symbol=st.session_state.ticker, timeframe="1d",limit=2000)
    except Exception:
        st.error(f"Error fetching data")
    data = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data=data.set_index(data["timestamp"])
    data["Mid"] = (data["high"] + data["low"]) / 2
    data = data.dropna()
    data = data[["volume", "Mid","timestamp"]]
    st.session_state.data_len=len(data)
    return data

if any(key not in st.session_state.keys() for key in ["preds", "ticker", "data", "batch_size", "tab3_seed","exchange"]):
    st.session_state.preds = None
    st.session_state.ticker = "ADA/USDT"
    st.session_state.data = 1
    st.session_state.batch_size = 30
    st.session_state.tab3_seed = None
    st.session_state.exchange="kraken"
    st.session_state.data = ticker()



# Map for coins with their emojis
options = {
    "ADA/USDT": "🌱 ADA",
    "BTC/USDT": "₿ BTC",
    "ETH/USDT": "💎 ETH",
    "XRP/USDT": "⚡ XRP",
    "XMR/USDT": "Ⓜ️ XMR",
    "LTC/USDT": "🌕 LTC",
    "DOGE/USDT": "🐕 DOGE",
    "DOT/USDT": "🌐 DOT",
    "BCH/USDT": "🍀 BCH",
    "SOL/USDT": "🌞 SOL",
    "BNB/USDT": "🪙 BNB",
    "UNI/USDT": "🦄 UNI",
    "LINK/USDT": "🔗 LINK",
    "AAVE/USDT": "🏦 AAVE",
    "XLM/USDT": "⭐ XLM",
    "MATIC/USDT": "🔹 MATIC",
    "MANA/USDT": "🕹️ MANA",
    "SHIB/USDT": "🐕‍🦺 SHIB",
    "CAKE/USDT": "🍰 CAKE",
    "AXS/USDT": "🛡️ AXS",
    "AVAX/USDT": "🔥 AVAX",
    "BUSD/USDT": "💵 BUSD",
    "DAI/USDT": "🏅 DAI",
    "USDT/USDT": "💲 USDT",
    "USDC/USDT": "💵 USDC",
}

st.cache_resource(show_spinner="Loading model ...")


def modeling():
    model_input = st.session_state.model
    data = st.session_state.data
    data = pd.DataFrame(data)
    data["x"] = None
    if model_input:
        if model_input == "Linear Regression":
            final_model = LinearRegression()
        elif model_input == "Support Vector Regression":
            final_model = SVR()
        elif model_input == "Decision Tree Regression":
            final_model = DecisionTreeRegressor()
        elif model_input == "KNeighbors Regression":
            final_model = KNeighborsRegressor()
        elif model_input == "Extra Trees Regression":
            final_model = ExtraTreesRegressor()
        elif model_input == "HistGradientBoosting Regression":
            final_model = HistGradientBoostingRegressor()
        elif model_input == "Ridge":
            final_model = Ridge()
        elif model_input == "ElasticNet":
            final_model = ElasticNet()
        elif model_input == "MLPRegressor":
            final_model = MLPRegressor()
        else:
            st.error("Unknow model or bad parameter!")
        whole = len(data)
        batch_size = st.session_state.batch_size
        x_values = []

        for i in range(batch_size, whole):
            y = data["Mid"].iloc[i - batch_size:i].to_numpy()
            x_values.append(y)  # Store as list

        data = data.iloc[batch_size:].copy()  # Remove first few NaN rows
        data["x"] = x_values  # Assign he collected sequences
        data = data.dropna()
        x_train, x_test, y_train, y_test = train_test_split(data["x"].tolist(), data["Mid"], shuffle=False,
                                                            test_size=0.1)
        if st.session_state.tab3_seed is not None:
            seed = st.session_state.tab3_seed
            final_model.random_state = seed
        final_model.fit(x_train, y_train)
        preds = final_model.predict(x_test)
        days_ahead = pd.date_range(start=data.index[-1], periods=st.session_state.days_ahead, freq="1D")
        last = x_test[-1].reshape(1, -1)
        days_ahead_prices = list()
        for i in range(st.session_state.days_ahead):
            p = final_model.predict(last)
            days_ahead_prices.append(p[0])
            p = np.array(p).reshape(1, 1)  # Ensure p is 2D (1 sample, 1 feature)
            last = np.append(last[:, 1:], p, axis=1)
        days_ahead_prices = pd.DataFrame(days_ahead_prices, index=days_ahead)
        preds = pd.DataFrame({"y": y_test, "preds": preds}, index=y_test.index)
        st.session_state.preds = preds
        st.session_state.days_ahead_prices = days_ahead_prices
        st.session_state.y_test = y_test

        return preds, days_ahead_prices


tab1, tab2, tab3 = st.tabs(["Data", "Model", "Environment"])
with tab1:
    with st.form("my_form"):
        ticker_choice = st.selectbox("Select Ticker", list(options.values()))
        exchanges_choice = st.selectbox("Select Ticker", exchanges_list)
        ticker_symbol = [key for key, value in options.items() if value == ticker_choice][0]
        st.session_state.ticker = ticker_symbol  # Set the ticker to session state
        st.session_state.exchange = exchanges_choice
        if st.form_submit_button("Submit"):
            st.session_state.data = ticker()
            if st.session_state.data.empty:
                st.error("Ticker not found!")
            else:
                st.success(f"Data Loaded Successfully! {st.session_state.data_len} DPs")

with tab2:
    with st.form("model_form"):
        col1, col2 = st.columns(2)
        col1.selectbox("Select Model", models, key="model")
        col2.number_input("Batch size", min_value=5, max_value=60, value=30, step=1, key="batch_size",
                          help="more stable predictions with higher batch size")
        col2.number_input("Days ahead", min_value=5, max_value=150, value=30, step=1, key="days_ahead",
                          help="The longer periods less accurate predictions")
        if st.form_submit_button("Submit", on_click=modeling):
            st.session_state.preds, st.session_state.days_ahead_prices = modeling()
    if st.session_state.preds is not None:
        with st.spinner("Plotting..."):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data["Mid"], mode="lines",
                                     name=f"{st.session_state.ticker}"))
            fig.add_trace(
                go.Scatter(x=st.session_state.preds.index, y=st.session_state.preds.preds, mode="lines",
                           name="Validation"))

            fig.add_trace(
                go.Scatter(x=st.session_state.days_ahead_prices.index, y=st.session_state.days_ahead_prices[0],
                           mode="lines",
                           name="Future Predictions", line=dict(color='cyan', width=3)))
            fig.update_layout(
                title=f"Data for {st.session_state.get('ticker', 'Unknown')}",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_white",
                showlegend=True,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            st.write("There is always chances of error in prediction.")
            st.write("Good Validation could be cuase of overfitting.")
            st.table(st.session_state.days_ahead_prices)

st.sidebar.button("clear cache", on_click=lambda: st.cache_data.clear())

with tab3:
    with st.form("env_form"):
        st.header("Environment", divider="grey")
        col1, col2 = st.columns(2)
        st.multiselect("Select the models", models,
                       default=["Linear Regression"], key="env")
        if col2.checkbox("Use Seed?"):
            col1.number_input("Seed", min_value=1, value=42, key="tab3_seed")
        st.form_submit_button("Submit")

    if st.session_state.env:
        df = pd.DataFrame()
        for model in st.session_state.env:
            del st.session_state.model
            st.session_state.model = model
            st.session_state.preds, st.session_state.days_ahead_prices = modeling()
            preds=st.session_state.preds
            y_test=st.session_state.y_test
            df[model] = st.session_state.days_ahead_prices
            st.title(model)
            st.dataframe(preds)
            df["MAPE"]=mape(y_test,preds["preds"])
            df["MSE"]=mse(y_test,preds["preds"])

        if df is not None:
            ex = st.expander("Prediction")
            df["mean"] = df.drop(columns=["MAPE","MSE"]).mean(axis=1)
            ex.table(df)
            fig = go.Figure()
            for i in df.drop(columns=["MAPE","MSE"]).columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[i], mode="lines", name=i))
            st.plotly_chart(fig)


