import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px

# --- CONFIG ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FDA_LIMIT = 50
TICKER_MAP_CSV = "sponsor_ticker_list.csv"

# --- UTILITIES ---
@st.cache_data
def load_ticker_map():
    """
    Load sponsor→ticker mapping from CSV, else use fallback dict.
    CSV should have headers: sponsor,ticker
    """
    if os.path.exists(TICKER_MAP_CSV):
        df = pd.read_csv(TICKER_MAP_CSV)
        return dict(zip(df["sponsor"], df["ticker"]))
    # Fallback: common biotech
    return {
        "Pfizer": "PFE", "Moderna": "MRNA", "Gilead Sciences": "GILD",
        "Amgen": "AMGN", "Biogen": "BIIB", "Regeneron": "REGN",
        "Vertex": "VRTX", "Alnylam": "ALNY", "BioMarin": "BMRN",
        "Sarepta": "SRPT", "Roche": "RHHBY", "Novartis": "NVS"
    }

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI and append as 'RSI' column."""
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def plot_price_chart(df: pd.DataFrame, ticker: str):
    """Render an interactive close-price chart for the given ticker."""
    fig = px.line(df, x="date", y="close", title=f"{ticker} Price")
    st.plotly_chart(fig, use_container_width=True)

# --- DATA FETCHERS ---
def get_fda_approvals(limit: int = FDA_LIMIT) -> pd.DataFrame:
    """Fetch recent FDA drug approvals and log the raw JSON."""
    url = f"https://api.fda.gov/drug/drugsfda.json?limit={limit}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        payload = res.json()
        st.write("🔎 FDA API response:", payload.get("meta", {}))
        items = payload.get("results", [])
    except Exception as e:
        st.error(f"Failed to fetch FDA data: {e}")
        return pd.DataFrame()
    records = []
    for item in items:
        sponsor = item.get("sponsor_name", "Unknown")
        for prod in item.get("products", []):
            records.append({
                "drug_name": prod.get("brand_name", "Unknown"),
                "sponsor": sponsor,
                "approval_date": item.get("approval_date", "")
            })
    return pd.DataFrame(records)

def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars for ticker, log JSON or errors, return DataFrame with date, close, volume."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url)
        res.raise_for_status()
        payload = res.json()
        st.write(f"🔎 Polygon data for {ticker}:", payload.get("results", []))
        bars = payload.get("results", [])
    except Exception as e:
        st.error(f"Error fetching {ticker} data: {e}")
        return pd.DataFrame()
    rows = [
        {
            "date": datetime.fromtimestamp(bar["t"] / 1000).date(),
            "close": bar["c"],
            "volume": bar["v"]
        }
        for bar in bars
    ]
    return pd.DataFrame(rows)

# --- SCREENING LOGIC ---
def flag_movers(approvals: pd.DataFrame, ticker_map: dict) -> pd.DataFrame:
    """
    For each approval:
      - Map sponsor→ticker
      - Pull 5 days before/after prices
      - Calc RSI, filter RSI<30 & latest volume>100k
      - Calc pct change >5%
    """
    flagged = []
    for _, row in approvals.iterrows():
        sponsor = row["sponsor"]
        ticker = ticker_map.get(sponsor)
        if not ticker:
            continue
        date_str = row["approval_date"]
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        start = (d - timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (d + timedelta(days=5)).strftime("%Y-%m-%d")
        price_df = get_stock_data(ticker, start, end)
        if price_df.shape[0] < 15:
            continue
        price_df = compute_rsi(price_df)
        latest = price_df.iloc[-1]
        # apply RSI & volume & pct filters
        pct_move = (latest["close"] - price_df["close"].iloc[0]) / price_df["close"].iloc[0]
        if latest["RSI"] < 30 and latest["volume"] > 100_000 and pct_move > 0.05:
            flagged.append({
                "ticker":        ticker,
                "drug":          row["drug_name"],
                "approval_date": date_str,
                "pct_change":    round(pct_move * 100, 2),
                "RSI":           round(latest["RSI"], 2),
                "volume":        int(latest["volume"])
            })
    return pd.DataFrame(flagged)

# --- STREAMLIT UI ---
st.title("💊 FDA Catalyst Trading Signal Engine")
st.write("Flags biotech stocks with RSI<30, volume>100k & >5% price move around approval.")

if st.button("Run Screener"):
    # 1. load & fetch data
    ticker_map = load_ticker_map()
    fda_df     = get_fda_approvals()
    if fda_df.empty:
        st.warning("No FDA data to screen.")
    else:
        movers = flag_movers(fda_df, ticker_map)

        if not movers.empty:
            st.success(f"📈 Found {len(movers)} candidate(s)!")
            st.dataframe(movers)

            # visualize each
            for _, m in movers.iterrows():
                st.markdown(f"### {m['ticker']} — {m['drug']} (±5d around {m['approval_date']})")
                # re-fetch prices for chart
                d = datetime.strptime(m["approval_date"], "%Y-%m-%d")
                start = (d - timedelta(days=5)).strftime("%Y-%m-%d")
                end   = (d + timedelta(days=5)).strftime("%Y-%m-%d")
                chart_df = get_stock_data(m["ticker"], start, end)
                plot_price_chart(chart_df, m["ticker"])
        else:
            st.warning("No significant movers found today.")