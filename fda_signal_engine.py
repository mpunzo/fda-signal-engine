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
POLYGON_API_KEY  = os.getenv("POLYGON_API_KEY", "")
OPENFIGI_API_KEY = os.getenv("OPENFIGI_API_KEY", "")
SLACK_WEBHOOK    = os.getenv("SLACK_WEBHOOK", "")
FDA_LIMIT        = 50         # items per page
FDA_PAGES        = 10         # pages of FDA data (50×10=500 approvals)
TICKER_MAP_CSV   = "sponsor_ticker_list.csv"
RSI_PERIOD       = 7          # ~7 trading days
EMA_SHORT        = 5
EMA_LONG         = 20

# --- LOGGING SETUP ---
logger = logging.getLogger("fda_signal_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=2)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)

# --- UTILITIES ---
@st.cache_data(ttl=3600)
def load_ticker_map():
    if os.path.exists(TICKER_MAP_CSV):
        df = pd.read_csv(TICKER_MAP_CSV)
        return dict(zip(df["sponsor"], df["ticker"]))
    return {
        "Pfizer":"PFE","Moderna":"MRNA","Gilead Sciences":"GILD",
        "Amgen":"AMGN","Biogen":"BIIB","Regeneron":"REGN",
        "Vertex":"VRTX","Alnylam":"ALNY","BioMarin":"BMRN",
        "Sarepta":"SRPT","Roche":"RHHBY","Novartis":"NVS"
    }

def query_openfigi(sponsor: str) -> str | None:
    """
    Fallback: query OpenFIGI to map sponsor name → ticker.
    Requires OPENFIGI_API_KEY env var.
    """
    url = "https://api.openfigi.com/v3/mapping"
    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": OPENFIGI_API_KEY
    }
    payload = [{"query": sponsor}]
    try:
        res = requests.post(url, json=payload, headers=headers)
        res.raise_for_status()
        data = res.json()[0].get("data", [])
        if data:
            return data[0].get("ticker")
    except Exception as e:
        logger.error("OpenFIGI error for %s: %s", sponsor, e)
    return None

def guess_ticker(sponsor: str, ticker_map: dict, threshold: int = 80) -> str | None:
    keys = list(ticker_map.keys())
    match, score = process.extractOne(sponsor, keys)
    return ticker_map[match] if score >= threshold else None

def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    up   = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    df["RSI"] = 100 - (100 / (1 + up/down))
    return df

def compute_ema(df: pd.DataFrame, span: int) -> pd.Series:
    return df["close"].ewm(span=span, adjust=False).mean()

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
        logger.error("Polygon error for %s: %s", ticker, e)
        bars = []
    rows = [{
        "date":   datetime.fromtimestamp(b["t"]/1000).date(),
        "close":  b["c"],
        "volume": b["v"]
    } for b in bars]
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def get_fda_approvals(limit: int = FDA_LIMIT, pages: int = FDA_PAGES) -> pd.DataFrame:
    records = []
    for page in range(pages):
        skip = page * limit
        url = f"https://api.fda.gov/drug/drugsfda.json?limit={limit}&skip={skip}"
        try:
            res = requests.get(url); res.raise_for_status()
            items = res.json().get("results", [])
            logger.info("Fetched FDA page %d: %d items", page+1, len(items))
        except Exception as e:
            logger.error("FDA page %d error: %s", page+1, e)
            break
        if not items:
            break
        for item in items:
            sponsor  = item.get("sponsor_name", "").strip()
            appr_date = item.get("approval_date", "")
            for prod in item.get("products", []):
                records.append({
                    "sponsor":       sponsor,
                    "drug_name":     prod.get("brand_name", "Unknown"),
                    "approval_date": appr_date
                })
    return pd.DataFrame(records)

def notify_slack(message: str):
    if SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={"text": message})
            logger.info("Slack notification sent.")
        except Exception as e:
            logger.error("Slack notify failed: %s", e)

def plot_full_chart(df: pd.DataFrame, ticker: str):
    df["EMA5"]  = compute_ema(df, EMA_SHORT)
    df["EMA20"] = compute_ema(df, EMA_LONG)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.close,  name="Close"))
    fig.add_trace(go.Scatter(x=df.date, y=df.RSI,    name="RSI", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA5,   name=f"EMA{EMA_SHORT}"))
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA20,  name=f"EMA{EMA_LONG}"))
    fig.update_layout(
        title=f"{ticker} Price & Indicators",
        yaxis2=dict(overlaying="y", side="right", title="RSI")
    )
    st.plotly_chart(fig, use_container_width=True)

def flag_movers(approvals: pd.DataFrame, ticker_map: dict) -> pd.DataFrame:
    results = []
    df = approvals.assign(
        date_obj = pd.to_datetime(approvals["approval_date"]),
        start    = (pd.to_datetime(approvals["approval_date"]) - timedelta(days=5)).dt.strftime("%Y-%m-%d"),
        end      = (pd.to_datetime(approvals["approval_date"]) + timedelta(days=5)).dt.strftime("%Y-%m-%d")
    )

    # fetch each ticker once
    tasks = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
        for _, row in df.iterrows():
            sponsor, start, end = row["sponsor"], row["start"], row["end"]
            ticker = ticker_map.get(sponsor) or guess_ticker(sponsor, ticker_map) or query_openfigi(sponsor)
            if not ticker:
                continue
            tasks[ticker] = exe.submit(get_stock_data, ticker, start, end)

    for ticker, fut in tasks.items():
        price_df = fut.result()
        if price_df.shape[0] < RSI_PERIOD+1:
            continue
        price_df = compute_rsi(price_df)
        latest   = price_df.iloc[-1]
        pct_move = (latest["close"] - price_df["close"].iloc[0]) / price_df["close"].iloc[0]
        df20     = price_df["volume"].rolling(20).mean().iloc[-1]
        ema_short = compute_ema(price_df, EMA_SHORT)
        ema_long  = compute_ema(price_df, EMA_LONG)
        golden    = ema_short.iloc[-2] < ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]
        volume_spike = latest["volume"] > 2 * (df20 or 0)

        # record any triggers
        conditions = {
            "RSI<30":      latest["RSI"] < 30,
            "Vol>100k":    latest["volume"] > 100_000,
            "Pct>5%":      pct_move > 0.05,
            "GoldenCross": golden,
            "VolSpike":    volume_spike
        }
        triggered = [name for name, ok in conditions.items() if ok]
        if triggered:
            # find the matching approval rows for this ticker
            for _, ap in approvals[approvals["sponsor_map"] == ticker].iterrows():
                results.append({
                    "ticker":        ticker,
                    "drug":          ap["drug_name"],
                    "approval_date": ap["approval_date"],
                    "pct_change":    round(pct_move*100, 2),
                    "RSI":           round(latest["RSI"], 2),
                    "volume":        int(latest["volume"]),
                    "triggers":      ", ".join(triggered)
                })

    return pd.DataFrame(results)

# --- STREAMLIT UI ---
st.title("💊 FDA Signal Engine (500 Approvals)")
st.write("Scans up to 500 approvals & flags ANY of: RSI<30, Vol>100k, Pct>5%, GoldenCross, VolSpike")

if st.button("Run Screener"):
    ticker_map = load_ticker_map()
    fda_df     = get_fda_approvals()
    if fda_df.empty:
        st.warning("No FDA data.")
    else:
        movers = flag_movers(fda_df, ticker_map)
        if not movers.empty:
            st.success(f"Found {len(movers)} candidate(s)!")
            st.dataframe(movers)
            notify_slack(f"Signals: {movers['ticker'].tolist()}")
            csv = movers.to_csv(index=False)
            st.download_button("Download CSV", csv, "signals.csv")
            for _, m in movers.iterrows():
                st.markdown(f"### {m['ticker']} — {m['drug']} (±5d @{m['approval_date']})")
                chart = get_stock_data(m["ticker"],
                                       (datetime.fromisoformat(m["approval_date"])-timedelta(5)).strftime("%Y-%m-%d"),
                                       (datetime.fromisoformat(m["approval_date"])+timedelta(5)).strftime("%Y-%m-%d"))
                if not chart.empty:
                    plot_full_chart(chart, m["ticker"])
        else:
            st.warning("No movers found.")