import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler

# --- CONFIG ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
SLACK_WEBHOOK   = os.getenv("SLACK_WEBHOOK", "")
RSI_PERIOD      = 7
EMA_SHORT       = 5
EMA_LONG        = 20
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
WINDOW_DAYS     = 14
MAX_TICKERS     = 500  # reduce to avoid long runtimes

# --- LOGGING ---
logger = logging.getLogger("robinhood_signal")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=2)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# --- UTILITIES ---
@st.cache_data(ttl=3600)
def load_index_tickers():
    """Return list of tickers from S&P 500, NASDAQ, NYSE as baseline universe"""
    sp500 = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")["Symbol"].tolist()
    url = "https://api.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    nasdaq = pd.read_csv(url, sep="|")["Symbol"].dropna().tolist()
    url = "https://api.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    nyse = pd.read_csv(url, sep="|")["ACT Symbol"].dropna().tolist()
    return list(set(sp500 + nasdaq + nyse))

@st.cache_data(ttl=600)
def get_stock_data(ticker, start, end):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    try:
        res = requests.get(url); res.raise_for_status()
        bars = res.json().get("results", [])
    except Exception as e:
        bars = []
        logger.warning(f"{ticker} failed: {e}")
    return pd.DataFrame([{
        "date": datetime.fromtimestamp(bar["t"] / 1000).date(),
        "close": bar["c"],
        "volume": bar["v"]
    } for bar in bars])

def compute_indicators(df):
    delta = df["close"].diff()
    up   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    down = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + up/down))
    df["EMA5"]  = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
    fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = fast - slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    return df

def flag_triggers(df):
    latest = df.iloc[-1]
    pct_move = (latest["close"] - df["close"].iloc[0]) / df["close"].iloc[0]
    vol20 = df["volume"].rolling(20).mean().iloc[-1]
    golden = df["EMA5"].iloc[-2] < df["EMA20"].iloc[-2] and df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]
    vol_spike = latest["volume"] > 2 * (vol20 or 0)
    macd_cross = df["MACD"].iloc[-2] < df["MACD_SIGNAL"].iloc[-2] and df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]

    checks = {
        "RSI<30": latest["RSI"] < 30,
        "Vol>100k": latest["volume"] > 100_000,
        "Pct>5%": pct_move > 0.05,
        "GoldenCross": golden,
        "VolSpike": vol_spike,
        "MACD_Cross": macd_cross
    }
    triggers = [k for k, v in checks.items() if v]
    return triggers, pct_move, latest["RSI"], latest["volume"]

def notify_slack(message):
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": message})
        except: pass

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close, name="Close"))
    fig.add_trace(go.Scatter(x=df.date, y=df.RSI, name="RSI", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA5, name="EMA5"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA20, name="EMA20"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD, name="MACD"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD_SIGNAL, name="MACD_Signal"))
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="RSI"),
        title=f"{ticker} Indicators"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- APP ---
st.title("📊 Robinhood Scanner + MACD")
st.caption("Scans major exchange stocks for RSI<30, GoldenCross, Volume Spikes, MACD Crosses and more.")

with st.spinner("Loading tickers…"):
    tickers = load_index_tickers()

st.write(f"🎯 {len(tickers)} total tickers from S&P 500, NASDAQ, NYSE")

use_top = st.checkbox("Limit to top 500 tickers by yesterday's volume", value=True)

if use_top:
    st.info("Fetching volumes…")
    cutoff = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    vols = {}
    for t in tickers:
        df = get_stock_data(t, cutoff, cutoff)
        if not df.empty:
            vols[t] = df["volume"].iloc[-1]
    tickers = sorted(vols, key=vols.get, reverse=True)[:MAX_TICKERS]
    st.write(f"Scanning {len(tickers)} most active tickers")

if st.button("Run Scan"):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    results = []
    progress = st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(get_stock_data, t, start, end): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            t = futures[fut]
            df = fut.result()
            if df.shape[0] < RSI_PERIOD + 1:
                continue
            df = compute_indicators(df)
            triggers, pct, rsi, vol = flag_triggers(df)
            if triggers:
                results.append({
                    "ticker": t,
                    "pct_change": round(pct * 100, 2),
                    "RSI": round(rsi, 2),
                    "volume": int(vol),
                    "triggers": ", ".join(triggers)
                })
            progress.progress(int(i / len(tickers) * 100))

    if results:
        movers = pd.DataFrame(results)
        st.success(f"📈 Found {len(movers)} movers!")
        st.dataframe(movers)
        st.download_button("Export CSV", movers.to_csv(index=False), "signals.csv")
        notify_slack(f"Triggers: {movers['ticker'].tolist()}")
        for _, row in movers.iterrows():
            st.markdown(f"### {row['ticker']} – {row['triggers']}")
            chart_df = get_stock_data(row["ticker"], start, end)
            if not chart_df.empty:
                chart_df = compute_indicators(chart_df)
                plot_chart(chart_df, row["ticker"])
    else:
        st.warning("No triggers found.")