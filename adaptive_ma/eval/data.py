"""Cached Binance data loader for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import requests

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "adaptive-ma")


def resolve_cache_dir() -> str:
    """Cache directory: ``$ADAPTIVE_MA_DATA_DIR`` if set, else ``~/.cache/adaptive-ma``.

    Never resolves inside the installed package directory.
    """
    env_dir = os.environ.get("ADAPTIVE_MA_DATA_DIR")
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return DEFAULT_CACHE_DIR


CACHE_DIR = resolve_cache_dir()
CACHE_FILE = os.path.join(CACHE_DIR, "binance_BTCUSDT_1m_sample.csv")
MANIFEST_FILE = os.path.join(CACHE_DIR, "manifest.json")


def ensure_btc_data_cached(n_bars: int = 1500) -> pd.DataFrame:
    """Fetch or load cached Binance BTC/USDT 1m klines."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE) and os.path.exists(MANIFEST_FILE):
        df = pd.read_csv(CACHE_FILE)
        return df

    print(f"Fetching {n_bars} bars of BTCUSDT 1m from Binance API to cache...")
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    # Binance max limit per call is 1000
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": min(n_bars, 1000)}
    r1 = requests.get(url, params=params, timeout=15).json()
    all_rows.extend(r1)

    if n_bars > 1000:
        first_open_time = r1[0][0]
        params2 = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": n_bars - 1000,
            "endTime": first_open_time - 1,
        }
        r2 = requests.get(url, params=params2, timeout=15).json()
        all_rows = r2 + all_rows

    df = pd.DataFrame(
        all_rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_vol",
            "taker_buy_quote_vol",
            "ignore",
        ],
    )
    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[col] = df[col].astype(float)

    # Compute order flow imbalance
    sell_vol = df["volume"] - df["taker_buy_vol"]
    imbalance = (df["taker_buy_vol"] - sell_vol) / (df["volume"] + 1e-8)
    df["imbalance"] = imbalance.clip(-1.0, 1.0)

    # Save CSV
    df.to_csv(CACHE_FILE, index=False)

    # Compute SHA256
    with open(CACHE_FILE, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "rows": len(df),
        "sha256": file_hash,
        "first_timestamp": int(df["open_time"].iloc[0]),
        "last_timestamp": int(df["open_time"].iloc[-1]),
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Cached {len(df)} rows to {CACHE_FILE} (sha256: {file_hash[:10]}...)")
    return df


def load_btc_dataset() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, str | int]]:
    """Load cached dataset with 4 order flow arms (contemp, lagged, zero, permuted)."""
    df = ensure_btc_data_cached()
    price = df["close"].values
    u_contemp = df["imbalance"].values

    # Lagged order flow u_{t-1}
    u_lagged = np.zeros_like(u_contemp)
    u_lagged[1:] = u_contemp[:-1]

    # Zero order flow u == 0
    u_zero = np.zeros_like(u_contemp)

    # Permuted order flow (negative control)
    rng = np.random.default_rng(42)
    u_permuted = rng.permutation(u_contemp)

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    arms = {
        "contemp": u_contemp,
        "lagged": u_lagged,
        "zero": u_zero,
        "permuted": u_permuted,
    }
    return price, arms, manifest
