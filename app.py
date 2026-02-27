import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pyotp
from SmartApi.smartConnect import SmartConnect
import statsmodels.api as sm
import warnings
import time
import io
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(page_title="📈 State Space Trend Scanner", layout="wide")
st.title("📈 State Space Trend Scanner (Angel One)")

# ================= ANGEL LOGIN =================
api_key = "EKa93pFu"
client_id = "R59803990"
password = "1234"
totp_secret = "5W4MC6MMLANC3UYOAW2QDUIFEU"

@st.cache_resource
def angel_login():
    obj = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    obj.generateSession(client_id, password, totp)
    return obj

try:
    obj = angel_login()
    st.success("Login Successful")
except:
    st.error("Login Failed")
    st.stop()

# ================= STOCK LIST =================
from Stock_tokens import stock_list   # {"RELIANCE": 2885, ...}

# ================= FETCH DATA =================
def fetch_data(token, interval, lookback):

    params = {
        "exchange": "NSE",
        "symboltoken": str(token),
        "interval": interval,
        "fromdate": (dt.datetime.now() - dt.timedelta(days=lookback+50)).strftime("%Y-%m-%d 09:15"),
        "todate": dt.datetime.now().strftime("%Y-%m-%d 15:30"),
    }

    response = obj.getCandleData(params)

    if not response or response["status"] != True:
        return None

    df = pd.DataFrame(response["data"],
                      columns=["timestamp","open","high","low","close","volume"])

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)

    return df


# ================= STATE SPACE MODEL =================
def compute_state_space(df, lookback):

    available = len(df)

    # Minimum data requirement
    if available < 120:
        return None, None

    # Auto-adjust lookback
    if available < lookback:
        lookback = available - 5

    closes = df["close"].tail(lookback)

    if (closes <= 0).any():
        return None, None

    try:
        log_prices = np.log(closes)

        model = sm.tsa.UnobservedComponents(log_prices, level='lltrend')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method='lbfgs', disp=False)

        filtered_state = result.filter_results.filtered_state

        if filtered_state.shape[1] >= 2:

            slope = filtered_state[-1, 1]

            # Trend reconstruction
            trend_log = filtered_state[0]
            trend_series = np.exp(trend_log)

            return slope, trend_series
        else:
            return None, None

    except:
        return None, None


# ================= TABS =================
tab1, tab2 = st.tabs(["📦 Batch Scanner", "📊 Stock Analysis"])

# ======================================================
# ================= TAB 1 : BATCH SCANNER ==============
# ======================================================
with tab1:

    st.subheader("📦 Batch Trend Scanner")

    items = list(stock_list.items())
    batch_size = 100
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    batch_no = st.selectbox("Select Batch", list(range(1, len(batches) + 1)))
    selected_batch = batches[batch_no - 1]

    col1, col2, col3 = st.columns(3)

    with col1:
        interval = st.selectbox("Interval", ["ONE_DAY", "ONE_HOUR"])

    with col2:
        lookback = st.number_input("Lookback Period", 120, 500, 200)

    with col3:
        slope_threshold = st.number_input("Trend Slope Threshold", value=0.001)

    scan_button = st.button("🚀 Run Scan")

    if scan_button:

        progress = st.progress(0)
        results = []
        total = len(selected_batch)

        for i, (symbol, token) in enumerate(selected_batch):

            df = fetch_data(token, interval, lookback)

            if df is not None:

                slope, _ = compute_state_space(df, lookback)

                if slope is not None and slope > slope_threshold:
                    results.append({
                        "Symbol": symbol,
                        "Trend Slope": round(slope, 6),
                        "Last Close": df["close"].iloc[-1],
                        "Batch": batch_no
                    })

            progress.progress((i + 1) / total)
            time.sleep(0.2)

        if results:
            result_df = pd.DataFrame(results).sort_values("Trend Slope", ascending=False)
            st.dataframe(result_df, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False)

            st.download_button(
                "⬇️ Download Excel",
                data=buffer.getvalue(),
                file_name=f"StateSpace_Batch_{batch_no}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No Strong Trend Stocks Found")


# ======================================================
# ================= TAB 2 : SINGLE STOCK ===============
# ======================================================
with tab2:

    st.subheader("📊 Single Stock Trend Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_stock = st.selectbox("Select Stock", list(stock_list.keys()))

    with col2:
        interval_single = st.selectbox("Interval", ["ONE_DAY", "ONE_HOUR"], key="single_interval")

    with col3:
        lookback_single = st.number_input("Lookback", 120, 500, 200, key="single_lookback")

    analyze_button = st.button("Analyze Stock")

    if analyze_button:

        token = stock_list[selected_stock]
        df = fetch_data(token, interval_single, lookback_single)

        if df is None:
            st.error("Data not available")
        else:

            slope, trend_series = compute_state_space(df, lookback_single)

            if slope is None:
                st.warning("Not enough usable data for model.")
            else:

                last_close = df["close"].iloc[-1]
                trend_score = round(slope * 1000, 2)

                # Interpretation
                if slope > 0.002:
                    meaning = "🔥 Very Strong Uptrend"
                elif slope > 0.001:
                    meaning = "🟢 Strong Uptrend"
                elif slope > 0.0005:
                    meaning = "🟡 Mild Uptrend"
                elif slope > -0.0005:
                    meaning = "⚪ Sideways"
                elif slope > -0.001:
                    meaning = "🔴 Mild Downtrend"
                else:
                    meaning = "🚨 Strong Downtrend"

                colA, colB, colC = st.columns(3)
                colA.metric("Last Close", round(last_close, 2))
                colB.metric("Trend Slope", round(slope, 6))
                colC.metric("Trend Score", trend_score)

                st.success(f"Trend Meaning: {meaning}")

                # ===== Chart =====
                df_plot = df.tail(200).copy()

                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=df_plot["timestamp"],
                    open=df_plot["open"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                    close=df_plot["close"],
                    name="Price"
                ))

                if trend_series is not None:

                    trend_len = min(len(trend_series), len(df_plot))
                    trend_plot = trend_series[-trend_len:]

                    fig.add_trace(go.Scatter(
                        x=df_plot["timestamp"].tail(trend_len),
                        y=trend_plot,
                        mode='lines',
                        name='Kalman Trend',
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)
