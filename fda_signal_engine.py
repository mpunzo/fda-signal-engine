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
MAX_TICKERS     = 500

# --- LOGGING SETUP ---
logger = logging.getLogger("universe_scanner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=2)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# --- UTILITIES ---
@st.cache_data(ttl=3600)
def load_index_tickers():
    """
    Load tickers from S&P 500, NASDAQ & NYSE via remote endpoints.
    Fail quietly (empty list) if any endpoint errors.
    """
    tickers = set()

    # S&P 500
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        sp500 = pd.read_csv(url)["Symbol"].dropna().tolist()
        tickers.update(sp500)
    except Exception as e:
        logger.warning("S&P500 load failed: %s", e)

    # NASDAQ-listed
    try:
        url = "https://api.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        nasdaq = pd.read_csv(url, sep="|")["Symbol"].dropna().tolist()
        tickers.update(nasdaq)
    except Exception as e:
        logger.warning("NASDAQ load failed: %s", e)

    # NYSE / otherlisted
    try:
        url = "https://api.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        nyse = pd.read_csv(url, sep="|")["ACT Symbol"].dropna().tolist()
        tickers.update(nyse)
    except Exception as e:
        logger.warning("NYSE load failed: %s", e)

    tickers = list(tickers)
    if not tickers:
        st.error("Failed to load any index tickers from remote sources.")
    return tickers

@st.cache_data(ttl=600)
def get_stock_data(ticker, start, end):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url); res.raise_for_status()
        bars = res.json().get("results", [])
    except Exception as e:
        bars = []
        logger.warning(f"{ticker} fetch failed: {e}")
    return pd.DataFrame([{
        "date":   datetime.fromtimestamp(b["t"]/1000).date(),
        "close":  b["c"],
        "volume": b["v"]
    } for b in bars])

def compute_indicators(df):
    delta = df["close"].diff()
    up   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    down = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + up/down))

    df["EMA5"]  = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["EMA20"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"]        = fast - slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    return df

def flag_triggers(df):
    latest = df.iloc[-1]
    pct_move = (latest["close"] - df["close"].iloc[0]) / df["close"].iloc[0]
    vol20    = df["volume"].rolling(20).mean().iloc[-1]
    golden   = (
        df["EMA5"].iloc[-2] < df["EMA20"].iloc[-2] and
        df["EMA5"].iloc[-1] > df["EMA20"].iloc[-1]
    )
    vol_spike = latest["volume"] > 2 * (vol20 or 0)
    macd_cross = (
        df["MACD"].iloc[-2] < df["MACD_SIGNAL"].iloc[-2] and
        df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]
    )

    conditions = {
        "RSI<30":      latest["RSI"] < 30,
        "Vol>100k":    latest["volume"] > 100_000,
        "Pct>5%":      pct_move > 0.05,
        "GoldenCross": golden,
        "VolSpike":    vol_spike,
        "MACD_Cross":  macd_cross
    }
    triggered = [k for k, v in conditions.items() if v]
    return triggered, pct_move, latest["RSI"], latest["volume"]

def plot_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close,  name="Close"))
    fig.add_trace(go.Scatter(x=df.date, y=df.RSI,    name="RSI",    yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA5,   name="EMA5"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA20,  name="EMA20"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD,       name="MACD"))
    fig.add_trace(go.Scatter(x=df.date, y=df.MACD_SIGNAL,name="Signal"))
    fig.update_layout(
        title=f"{ticker} Indicators",
        yaxis2=dict(overlaying="y", side="right", title="RSI")
    )
    st.plotly_chart(fig, use_container_width=True)

def notify_slack(msg):
    if SLACK_WEBHOOK:
        try: requests.post(SLACK_WEBHOOK, json={"text": msg})
        except: pass

# --- APP UI ---
st.title("📊 Universe Scanner + MACD")
st.caption("S&P500, NASDAQ, NYSE tickers | top500 vol filter | RSI/EMA/MACD triggers")

with st.spinner("Loading tickers…"):
    tickers = load_index_tickers()

st.write(f"🌐 Universe: {len(tickers)} tickers")

# liquidity filter
if st.checkbox("Limit to top 500 by yesterday’s volume", value=True):
    cutoff = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    vols = {}
    for t in tickers:
        df = get_stock_data(t, cutoff, cutoff)
        if not df.empty: vols[t] = df["volume"].iloc[-1]
    tickers = sorted(vols, key=vols.get, reverse=True)[:MAX_TICKERS]
    st.write(f"🔎 Scanning top {len(tickers)} by volume")

if st.button("Run Scan"):
    start = (datetime.today() - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    progress = st.progress(0)
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(get_stock_data, t, start, end): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            t = futures[fut]
            df = fut.result()
            if df.shape[0] < RSI_PERIOD+1: continue

            df = compute_indicators(df)
            triggers, pct, rsi, vol = flag_triggers(df)
            if triggers:
                results.append({
                    "ticker":     t,
                    "pct_change": round(pct*100,2),
                    "RSI":        round(rsi,2),
                    "volume":     int(vol),
                    "triggers":   ", ".join(triggers)
                })

            progress.progress(int((i+1)/len(tickers)*100))

    if results:
        movers = pd.DataFrame(results)
        st.success(f"📈 Found {len(movers)} movers!")
        st.dataframe(movers)
        st.download_button("Export CSV", movers.to_csv(index=False), "signals.csv")
        notify_slack(f"Signals: {movers['ticker'].tolist()}")
        for _, row in movers.iterrows():
            st.markdown(f"### {row['ticker']} — {row['triggers']}")
            chart = get_stock_data(row["ticker"], start, end)
            if not chart.empty:
                chart = compute_indicators(chart)
                plot_chart(chart, row["ticker"])
    else:
        st.warning("No triggers found.")