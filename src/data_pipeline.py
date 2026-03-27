"""
data_pipeline.py
----------------
1. Download OHLCV price data via yfinance.
2. Compute price-based technical features.
3. Load and align news headlines (Benzinga CSV from Kaggle).
4. Construct the binary target (next-day return direction).
5. Temporal 80/10/10 train/val/test split.
6. Save: data/processed/features.parquet, data/splits/{train,val,test}.pt

Leakage guard: news from day t-1 is used to predict return on day t.
No shuffling — all splits respect temporal order.

Usage:
    python src/data_pipeline.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = ["MRK", "JNJ", "KO", "NVDA", "QCOM"]
START_DATE = "2013-01-01"
END_DATE = "2020-06-30"

# Kaggle Benzinga — analyst_ratings_processed.csv
# Columns: title (headline), date (minute-precision, UTC-4), stock (ticker)
NEWS_CSV = Path("data/raw/news/analyst_ratings_processed.csv")
NEWS_COL_MAP = {
    "title": "headline",
    "stock": "ticker",
    # "date" matches directly
}

PRICE_DIR = Path("data/raw/prices")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

# Temporal split: train 2013-2017, val 2018, test 2019-Jun 2020
TRAIN_END = "2017-12-31"
VAL_END   = "2018-12-31"
# test = 2019-01-01 → 2020-06-30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirs():
    for d in [PRICE_DIR, PROCESSED_DIR, SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for all tickers, cache to CSV, return combined df."""
    frames = []
    for ticker in tickers:
        cache = PRICE_DIR / f"{ticker}.csv"
        if cache.exists():
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            print(f"  {ticker}: loaded from cache ({len(df)} rows)")
        else:
            print(f"  {ticker}: downloading from Yahoo Finance...")
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  WARNING: no data for {ticker}, skipping.")
                continue
            # flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            df.to_csv(cache)
            print(f"  {ticker}: saved {len(df)} rows to {cache}")
        df["Ticker"] = ticker
        frames.append(df)
    return pd.concat(frames).sort_index()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical price features from OHLCV.
    All rolling windows use only past data (.shift(1) where needed).
    Returns a DataFrame with one row per (Date, Ticker).
    """
    results = []
    for ticker, g in df.groupby("Ticker"):
        g = g.copy().sort_index()
        c = g["Close"]
        v = g["Volume"]

        g["return_1d"] = c.pct_change()
        g["return_5d"] = c.pct_change(5)
        g["return_20d"] = c.pct_change(20)

        g["vol_20d"] = g["return_1d"].rolling(20).std()

        g["ma5"] = c.rolling(5).mean()
        g["ma20"] = c.rolling(20).mean()
        g["ma_ratio"] = g["ma5"] / g["ma20"]  # momentum: fast/slow MA

        # RSI (14-day)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g["rsi"] = 100 - (100 / (1 + rs))

        # MACD signal line
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        g["macd_signal"] = macd.ewm(span=9, adjust=False).mean()

        # Volume features
        g["vol_change"] = v.pct_change()
        g["vol_ma20"] = v.rolling(20).mean()
        g["turnover_ratio"] = v / g["vol_ma20"]

        # OHLCV-derived candlestick features
        o, h, l = g["Open"], g["High"], g["Low"]
        hl = (h - l).replace(0, np.nan)           # avoid divide-by-zero
        g["intraday_ret"]   = (c - o) / o                        # open-to-close direction
        g["hl_range"]       = hl / c                             # intraday volatility
        g["upper_shadow"]   = (h - c.combine(o, max)) / hl      # bearish rejection
        g["lower_shadow"]   = (c.combine(o, min) - l) / hl      # bullish support

        # Target: 1 if next-day close > today's close, else 0
        # .shift(-1): next day's close; drop last row (no target available)
        g["target"] = (c.shift(-1) > c).astype(int)

        results.append(g)

    out = pd.concat(results).sort_index().reset_index()  # Date → column
    return out


FEATURE_COLS = [
    # Returns & momentum
    "return_1d", "return_5d", "return_20d",
    # Volatility
    "vol_20d",
    # Trend
    "ma_ratio",
    # Oscillators
    "rsi", "macd_signal",
    # Volume
    "vol_change", "turnover_ratio",
    # OHLCV candlestick
    "intraday_ret", "hl_range", "upper_shadow", "lower_shadow",
]


def load_news(path: Path) -> pd.DataFrame:
    """
    Load analyst_ratings_processed.csv from Kaggle Benzinga.
    Columns: Unnamed:0, title, date (UTC-4 minute precision), stock.
    Returns DataFrame with columns: date (datetime.date), ticker (str), headline (str).
    """
    if not path.exists():
        print(f"  WARNING: news file not found at {path}. Text features will be zeros.")
        return pd.DataFrame(columns=["date", "ticker", "headline"])

    print(f"  Loading news from {path} ...")
    news = pd.read_csv(path, usecols=["title", "date", "stock"], low_memory=False)
    news = news.rename(columns={"title": "headline", "stock": "ticker"})

    # Parse timestamps — stored as UTC-4; convert to UTC then extract date
    news["date"] = pd.to_datetime(news["date"], utc=True, errors="coerce").dt.date
    news = news.dropna(subset=["date", "ticker", "headline"])

    news["ticker"] = news["ticker"].str.upper().str.strip()
    news = news[news["ticker"].isin(TICKERS)]

    # Filter to experiment window
    news = news[
        (news["date"] >= pd.Timestamp(START_DATE).date()) &
        (news["date"] <= pd.Timestamp(END_DATE).date())
    ]

    # Aggregate all headlines per (date, ticker) into one string
    news = (
        news.groupby(["date", "ticker"])["headline"]
        .apply(lambda x: " | ".join(x.dropna().astype(str)))
        .reset_index()
    )
    print(f"  News: {len(news)} (date, ticker) rows after aggregation.")
    return news


def align_features_and_news(features: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price features with news headlines.
    Leakage guard: news[date t-1] is aligned to price row[date t].
    Days with no news get headline = "" (embed_news.py handles → zero vector).
    """
    features["date"] = pd.to_datetime(features["Date"]).dt.date

    if news.empty:
        features["headline"] = ""
        return features

    # Shift news forward by one trading day:
    # For each ticker, map news date → the *next* trading date for that ticker.
    aligned_frames = []
    for ticker, g in features.groupby("Ticker"):
        g = g.sort_values("date").copy()
        dates = g["date"].tolist()
        # Build map: news_date → next_trading_date
        next_date_map = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}

        ticker_news = news[news["ticker"] == ticker].copy()
        ticker_news["aligned_date"] = ticker_news["date"].map(next_date_map)
        ticker_news = ticker_news.dropna(subset=["aligned_date"])
        ticker_news = ticker_news.drop(columns=["date"]).rename(columns={"aligned_date": "date"})

        g = g.merge(ticker_news[["date", "headline"]], on="date", how="left")
        g["headline"] = g["headline"].fillna("")
        aligned_frames.append(g)

    return pd.concat(aligned_frames).sort_values(["date", "Ticker"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    """
    Fixed-date temporal split:
      Train: 2013-2017
      Val:   2018
      Test:  2019-Jun 2020
    """
    train_end = pd.Timestamp(TRAIN_END).date()
    val_end   = pd.Timestamp(VAL_END).date()

    train = df[df["date"] <= train_end]
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test  = df[df["date"] > val_end]

    print(f"  Train: {len(train)} rows ({train['date'].min()} → {train['date'].max()})")
    print(f"  Val:   {len(val)} rows ({val['date'].min()} → {val['date'].max()})")
    print(f"  Test:  {len(test)} rows ({test['date'].min()} → {test['date'].max()})")
    return train, val, test


def check_class_balance(df: pd.DataFrame):
    counts = df["target"].value_counts(normalize=True)
    print(f"  Class balance — up: {counts.get(1, 0):.1%}, down: {counts.get(0, 0):.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _make_dirs()

    print("\n=== 1. Downloading prices ===")
    prices = download_prices(TICKERS, START_DATE, END_DATE)

    print("\n=== 2. Computing features ===")
    features = compute_features(prices)
    # Drop rows with NaN features (first ~26 rows per ticker for rolling windows)
    features = features.dropna(subset=FEATURE_COLS + ["target"])
    # Drop last row per ticker (no next-day target)
    features = features.groupby("Ticker").apply(lambda g: g.iloc[:-1]).reset_index(drop=True)
    print(f"  Feature rows after cleaning: {len(features)}")

    print("\n=== 3. Loading and aligning news ===")
    news = load_news(NEWS_CSV)
    aligned = align_features_and_news(features, news)

    print("\n=== 4. Class balance (full dataset) ===")
    check_class_balance(aligned)

    print("\n=== 5. Saving processed features ===")
    save_cols = ["Date", "Ticker", "date", "headline", "target"] + FEATURE_COLS + [
        "Open", "High", "Low", "Close", "Volume"
    ]
    aligned = aligned[[c for c in save_cols if c in aligned.columns]]
    out_path = PROCESSED_DIR / "features.parquet"
    aligned.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

    print("\n=== 6. Temporal split ===")
    train, val, test = temporal_split(aligned)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        torch.save(split_df.reset_index(drop=True), SPLITS_DIR / f"{split_name}.pt")
    print("  Splits saved to data/splits/")

    print("\nDone. Next step: python src/embed_news.py")


if __name__ == "__main__":
    main()
