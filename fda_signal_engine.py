import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler

# --- CONFIG ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
SLACK_WEBHOOK   = os.getenv("SLACK_WEBHOOK", "")
FDA_LIMIT       = 50
TICKER_MAP_CSV  = "sponsor_ticker_list.csv"
RSI_PERIOD      = 7    # match ±5-day window (~7 trading days)
EMA_SHORT       = 5
EMA_LONG        = 20

# --- LOGGING SETUP ---
logger = logging.getLogger("fda_signal_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# --- UTILITIES ---
@st.cache_data(ttl=600)
def load_ticker_map():
    if os.path.exists(TICKER_MAP_CSV):
        df = pd.read_csv(TICKER_MAP_CSV)
        return dict(zip(df["sponsor"], df["ticker"]))
    # fallback map
    return {
        "Pfizer": "PFE", "Moderna": "MRNA", "Gilead Sciences": "GILD",
        "Amgen": "AMGN", "Biogen": "BIIB", "Regeneron": "REGN",
        "Vertex": "VRTX", "Alnylam": "ALNY", "BioMarin": "BMRN",
        "Sarepta": "SRPT", "Roche": "RHHBY", "Novartis": "NVS"
    }

def guess_ticker(sponsor: str, ticker_map: dict, threshold: int = 80) -> str | None:
    keys = list(ticker_map.keys())
    match, score = process.extractOne(sponsor, keys)
    return ticker_map[match] if score >= threshold else None

def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["RSI"] = 100 - (100 / (1 + up.rolling(period).mean() / down.rolling(period).mean()))
    return df

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

@st.cache_data(ttl=600)
def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url); res.raise_for_status()
        bars = res.json().get("results", [])
        logger.info("Fetched %s bars for %s", len(bars), ticker)
    except Exception as e:
        bars = []
        logger.error("Error fetching %s data: %s", ticker, e)
        # no st.error here in worker thread
    rows = [{
        "date":   datetime.fromtimestamp(b["t"] / 1000).date(),
        "close":  b["c"],
        "volume": b["v"]
    } for b in bars]
    return pd.DataFrame(rows)

def get_fda_approvals(limit: int = FDA_LIMIT) -> pd.DataFrame:
    url = f"https://api.fda.gov/drug/drugsfda.json?limit={limit}"
    try:
        res = requests.get(url); res.raise_for_status()
        payload = res.json()
        logger.info("Fetched FDA approvals: %d", limit)
    except Exception as e:
        logger.error("Error fetching FDA data: %s", e)
        st.error(f"Failed to fetch FDA data: {e}")
        return pd.DataFrame()
    records = []
    for item in payload.get("results", []):
        sponsor = item.get("sponsor_name", "").strip()
        date_str = item.get("approval_date", "")
        for prod in item.get("products", []):
            records.append({
                "sponsor":       sponsor,
                "drug_name":     prod.get("brand_name", "Unknown"),
                "approval_date": date_str
            })
    return pd.DataFrame(records)

def notify_slack(message: str):
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": message})
            logger.info("Sent Slack notification: %s", message)
        except Exception as e:
            logger.error("Slack notification failed: %s", e)

def plot_full_chart(df: pd.DataFrame, ticker: str):
    df["EMA5"]  = compute_ema(df["close"], EMA_SHORT)
    df["EMA20"] = compute_ema(df["close"], EMA_LONG)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close,  name="Close"))
    fig.add_trace(go.Scatter(x=df.date, y=df.RSI,    name="RSI",    yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA5,   name=f"EMA{EMA_SHORT}"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA20,  name=f"EMA{EMA_LONG}"))
    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        yaxis2=dict(overlaying="y", side="right", title="RSI")
    )
    st.plotly_chart(fig, use_container_width=True)

def flag_movers(df: pd.DataFrame, ticker_map: dict) -> pd.DataFrame:
    results = []
    df = df.assign(
        date_obj = pd.to_datetime(df["approval_date"]),
        start    = (pd.to_datetime(df["approval_date"]) - timedelta(days=5)).dt.strftime("%Y-%m-%d"),
        end      = (pd.to_datetime(df["approval_date"]) + timedelta(days=5)).dt.strftime("%Y-%m-%d")
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
        futures = {}
        for idx, row in df.iterrows():
            sponsor, start, end = row["sponsor"], row["start"], row["end"]
            ticker = ticker_map.get(sponsor) or guess_ticker(sponsor, ticker_map)
            if not ticker:
                continue
            futures[exe.submit(get_stock_data, ticker, start, end)] = (idx, ticker, row)

        for fut in concurrent.futures.as_completed(futures):
            idx, ticker, row = futures[fut]
            price_df = fut.result()
            if price_df.shape[0] < RSI_PERIOD + 1:
                continue
            price_df = compute_rsi(price_df)
            latest = price_df.iloc[-1]
            pct_move = (latest["close"] - price_df["close"].iloc[0]) / price_df["close"].iloc[0]

            # volume spike & EMA crossover
            df20 = price_df["volume"].rolling(20).mean().iloc[-1]
            ema_short = compute_ema(price_df["close"], EMA_SHORT)
            ema_long  = compute_ema(price_df["close"], EMA_LONG)
            golden    = ema_short.iloc[-2] < ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]
            volume_spike = latest["volume"] > 2 * (df20 or 0)

            # check any conditions and record which
            conditions = {
                "RSI<30":      latest["RSI"] < 30,
                "Vol>100k":    latest["volume"] > 100_000,
                "Pct>5%":      pct_move > 0.05,
                "GoldenCross": golden,
                "VolSpike":    volume_spike
            }
            triggered = [name for name, ok in conditions.items() if ok]
            if triggered:
                results.append({
                    "ticker":        ticker,
                    "drug":          row["drug_name"],
                    "approval_date": row["approval_date"],
                    "pct_change":    round(pct_move * 100, 2),
                    "RSI":           round(latest["RSI"], 2),
                    "volume":        int(latest["volume"]),
                    "triggers":      ", ".join(triggered)
                })

    return pd.DataFrame(results)

# --- STREAMLIT UI ---
st.title("💊 FDA Catalyst Trading Signal Engine")
st.write("Flags biotech stocks that meet ANY of: RSI<30, volume>100k, golden-cross, >5% price move.")

if st.button("Run Screener"):
    ticker_map = load_ticker_map()
    fda_df     = get_fda_approvals()
    if fda_df.empty:
        st.warning("No FDA data to screen.")
    else:
        movers = flag_movers(fda_df, ticker_map)
        if not movers.empty:
            st.success(f"📈 {len(movers)} candidate(s) found!")
            st.dataframe(movers)
            notify_slack(f"FDA signals: {movers['ticker'].tolist()}")
            csv = movers.to_csv(index=False)
            st.download_button("Download signals as CSV", csv, "signals.csv")
            for _, m in movers.iterrows():
                st.markdown(f"### {m['ticker']} — {m['drug']} (±5d around {m['approval_date']})")
                start = (datetime.fromisoformat(m["approval_date"]) - timedelta(days=5)).strftime("%Y-%m-%d")
                end   = (datetime.fromisoformat(m["approval_date"]) + timedelta(days=5)).strftime("%Y-%m-%d")
                price_df = get_stock_data(m["ticker"], start, end)
                if not price_df.empty:
                    price_df = compute_rsi(price_df)
                    plot_full_chart(price_df, m["ticker"])
        else:
            st.warning("No significant movers found today.")