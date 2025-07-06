import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FDA_LIMIT = 50
TICKER_MAP = {
    "Pfizer": "PFE",
    "Moderna": "MRNA",
    "Gilead Sciences": "GILD",
    "Amgen": "AMGN",
    "Biogen": "BIIB"
}

# --- FUNCTIONS ---
def get_fda_approvals(limit=FDA_LIMIT):
    url = f"https://api.fda.gov/drug/drugsfda.json?limit={limit}"
    res = requests.get(url)
    data = res.json().get("results", [])
    approvals = []
    for item in data:
        sponsor = item.get("sponsor_name", "Unknown")
        for product in item.get("products", []):
            approvals.append({
                "drug_name": product.get("brand_name", "Unknown"),
                "sponsor": sponsor,
                "approval_date": item.get("approval_date", "")
            })
    return pd.DataFrame(approvals)

def map_tickers(df):
    df["ticker"] = df["sponsor"].map(TICKER_MAP)
    return df.dropna(subset=["ticker"])

def get_stock_data(ticker, start_date, end_date):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/"
        f"range/1/day/{start_date}/{end_date}"
        f"?adjusted=true&sort=asc&limit=30&apiKey={POLYGON_API_KEY}"
    )
    res = requests.get(url).json().get("results", [])
    prices = [
        {"date": datetime.fromtimestamp(r["t"]/1000).date(), "close": r["c"]}
        for r in res
    ]
    return pd.DataFrame(prices)

def flag_movers(df):
    flagged = []
    for _, row in df.iterrows():
        date_str = row["approval_date"]
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            continue
        start = (d - timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (d + timedelta(days=5)).strftime("%Y-%m-%d")
        price_df = get_stock_data(row["ticker"], start, end)
        if price_df.shape[0] >= 2:
            pct = (price_df["close"].iloc[-1] - price_df["close"].iloc[0]) / price_df["close"].iloc[0]
            if pct > 0.05:
                flagged.append({
                    "ticker":       row["ticker"],
                    "drug":         row["drug_name"],
                    "approval_date": date_str,
                    "pct_change":   round(pct*100, 2)
                })
    return pd.DataFrame(flagged)

# --- STREAMLIT UI ---
st.title("💊 FDA Catalyst Trading Signal Engine")
st.write("Flags biotech stocks around FDA events with >5% price moves.")

if st.button("Run Screener"):
    fda_df    = get_fda_approvals()
    mapped_df = map_tickers(fda_df)
    movers    = flag_movers(mapped_df)

    if not movers.empty:
        st.success("📈 Potential Movers Found!")
        st.dataframe(movers)
    else:
        st.warning("No significant movers in the current FDA dataset.")