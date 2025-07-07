import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from fuzzywuzzy import process
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler

# --- CONFIG ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
SLACK_WEBHOOK   = os.getenv("SLACK_WEBHOOK", "")
WINDOW_DAYS     = 14
RSI_PERIOD      = 7    # ~7 trading days
EMA_SHORT       = 5
EMA_LONG        = 20
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9

# --- LOGGING SETUP ---
logger = logging.getLogger("signal_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=2)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# --- UTILITIES ---
@st.cache_data(ttl=3600)
def get_robinhood_tickers():
    """Fetch all stock tickers from Robinhood instruments endpoint."""
    url = "https://api.robinhood.com/instruments/?type=stock&active=true&limit=100"
    tickers = []
    while url:
        res = requests.get(url); res.raise_for_status()
        data = res.json()
        tickers += [i["symbol"] for i in data.get("results", [])]
        url = data.get("next")
    return tickers

def compute_rsi(df):
    delta = df["close"].diff()
    up   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    down = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + up/down))
    return df

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_macd(df):
    fast = compute_ema(df["close"], MACD_FAST)
    slow = compute_ema(df["close"], MACD_SLOW)
    df["MACD"]       = fast - slow
    df["MACD_SIGNAL"]= df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    return df

@st.cache_data(ttl=600)
def get_stock_data(ticker, start, end):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url); res.raise_for_status()
        bars = res.json().get("results", [])
        logger.info("Fetched %d bars for %s", len(bars), ticker)
    except Exception as e:
        bars = []
        logger.error("Polygon error for %s: %s", ticker, e)
    return pd.DataFrame([{
        "date":   datetime.fromtimestamp(b["t"]/1000).date(),
        "close":  b["c"],
        "volume": b["v"]
    } for b in bars])

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close, name="Close"))
    fig.add_trace(go.Scatter(x=df.date, y=df.RSI,   name="RSI",   yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA5,  name=f"EMA{EMA_SHORT}"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA20, name=f"EMA{EMA_LONG}"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD,       name="MACD"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD_SIGNAL,name="Signal"))
    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        yaxis2=dict(overlaying="y", side="right", title="RSI")
    )
    st.plotly_chart(fig, use_container_width=True)

def flag_signals(tickers):
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    results = []

    # fetch in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(get_stock_data, t, start, end): t for t in tickers}

        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            df = fut.result()
            if df.shape[0] < WINDOW_DAYS//2:  # ensure enough data
                continue

            df = compute_rsi(df)
            df["EMA5"]  = compute_ema(df["close"], EMA_SHORT)
            df["EMA20"] = compute_ema(df["close"], EMA_LONG)
            df = compute_macd(df)

            latest = df.iloc[-1]
            pct_move = (latest["close"] - df["close"].iloc[0]) / df["close"].iloc[0]
            vol20 = df["volume"].rolling(20).mean().iloc[-1]
            golden = df["EMA5"].iloc[-2] < df["EMA20"].iloc[-2] and df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]
            vol_spike = latest["volume"] > 2 * (vol20 or 0)
            macd_cross = df["MACD"].iloc[-2] < df["MACD_SIGNAL"].iloc[-2] and df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]

            conditions = {
                "RSI<30":        latest["RSI"] < 30,
                "Vol>100k":      latest["volume"] > 100_000,
                "Pct>5%":        pct_move > 0.05,
                "GoldenCross":   golden,
                "VolSpike":      vol_spike,
                "MACD_Cross":    macd_cross
            }
            triggered = [k for k,v in conditions.items() if v]
            if triggered:
                results.append({
                    "ticker":     t,
                    "pct_change": round(pct_move*100,2),
                    "RSI":        round(latest["RSI"],2),
                    "volume":     int(latest["volume"]),
                    "triggers":   ", ".join(triggered)
                })

    return pd.DataFrame(results)

# --- STREAMLIT UI ---
st.title("📈 Robinhood Universe Signal Scanner")
st.write("Flags any stock meeting: RSI<30, Vol>100k, Pct>5%, GoldenCross, VolSpike, MACD_Cross")

if st.button("Run Full Scan"):
    symbols = get_robinhood_tickers()
    st.write(f"Scanning {len(symbols)} tickers…")
    signals = flag_signals(symbols)
    if not signals.empty:
        st.success(f"Found {len(signals)} candidates!")
        st.dataframe(signals)
        csv = signals.to_csv(index=False)
        st.download_button("Export CSV", csv, "signals.csv")
        for _, row in signals.iterrows():
            st.markdown(f"### {row['ticker']} — Triggers: {row['triggers']}")
            df = get_stock_data(row["ticker"],
                                (datetime.today()-timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d"),
                                datetime.today().strftime("%Y-%m-%d"))
            df = compute_rsi(df); df["EMA5"] = compute_ema(df["close"],EMA_SHORT)
            df["EMA20"] = compute_ema(df["close"],EMA_LONG); df = compute_macd(df)
            plot_chart(df, row["ticker"])
    else:
        st.warning("No signals detected in the current universe.")