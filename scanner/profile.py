"""Stock profile — float, short interest, 21-day ABR, sector.

ABR is computed from existing price data (fast).
Float/SI/sector are fetched from yfinance Ticker.info (slow, cached 30 days).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as dt_date

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .cache import Cache
from .config import ABR_PERIOD

# How many top stocks to enrich with yfinance profile data
PROFILE_FETCH_LIMIT = 50
PROFILE_CACHE_TTL_HOURS = 30 * 24  # 30 days


def compute_abr(df: pd.DataFrame, period: int = ABR_PERIOD) -> float | None:
    """21-day Average Body Range — bodies only, gaps excluded.

    ABR = mean(|Close - Open|) over the last *period* days.
    Used to normalise distances and position sizing.
    """
    if len(df) < period:
        return None
    body = (df["Close"] - df["Open"]).abs().tail(period)
    val = body.mean()
    return round(val, 4) if pd.notna(val) else None


def compute_abr_universe(
    universe: list[str], price_data: dict[str, pd.DataFrame]
) -> dict[str, float]:
    """Compute ABR for every stock in the universe (fast — no API calls)."""
    result: dict[str, float] = {}
    for ticker in universe:
        df = price_data.get(ticker)
        if df is None:
            continue
        abr = compute_abr(df)
        if abr is not None:
            result[ticker] = abr
    return result


def fetch_stock_profiles(
    tickers: list[str],
    cache: Cache,
    limit: int = PROFILE_FETCH_LIMIT,
) -> dict[str, dict]:
    """Fetch float, short interest, and sector from yfinance.

    Only fetches the first *limit* tickers whose cached entry is missing,
    failed, older than 30 days, or past its next earnings date. The file's
    own mtime can't age entries out — it is rewritten on every fetch.
    """
    cached = cache.get_json("profiles", max_age_hours=PROFILE_CACHE_TTL_HOURS)
    profiles: dict[str, dict] = cached if cached else {}

    today = dt_date.today()
    to_fetch = [t for t in tickers[:limit] if _needs_refresh(profiles.get(t, {}), today)]
    if not to_fetch:
        return profiles

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in to_fetch}
        with tqdm(total=len(to_fetch), desc="  Fetching profiles", unit="stock") as pbar:
            for fut in as_completed(futures):
                ticker = futures[fut]
                fresh = fut.result()
                # A failed or stub fetch must not wipe a good cached entry;
                # the old one stays due for refresh, so the next run retries.
                if (fresh.get("quote_type") is not None
                        or profiles.get(ticker, {}).get("quote_type") is None):
                    profiles[ticker] = fresh
                pbar.update(1)

    cache.set_json("profiles", profiles)
    return profiles


def earnings_days_ago(profile: dict, today: dt_date | None = None) -> int | None:
    """Days since the last earnings, aged by the days since the profile
    was fetched. No earnings can fall in between: an entry is refetched
    once its next earnings date has passed."""
    days = profile.get("earnings_days_ago")
    fetched = profile.get("fetched_at")
    if days is None or fetched is None:
        return days
    return days + ((today or dt_date.today()) - dt_date.fromisoformat(fetched)).days


def _needs_refresh(entry: dict, today: dt_date) -> bool:
    """True if a cached profile should be fetched again."""
    if entry.get("quote_type") is None:
        return True  # never fetched, failed ({}), or a stub info response
    fetched = entry.get("fetched_at")
    if fetched is None:
        return True  # cached before entries carried a fetch date
    if (today - dt_date.fromisoformat(fetched)).days * 24 > PROFILE_CACHE_TTL_HOURS:
        return True
    next_earnings = entry.get("next_earnings")
    return next_earnings is not None and dt_date.fromisoformat(next_earnings) < today


def _fetch_one(ticker: str) -> dict:
    """Fetch profile data + earnings dates for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        float_shares = info.get("floatShares")

        # Earnings dates — most recent past and next upcoming
        earnings = _parse_earnings(t)

        return {
            "float_shares": float_shares,
            "float_label": _float_label(float_shares),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "short_ratio": info.get("shortRatio"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "quote_type": info.get("quoteType"),
            "fetched_at": dt_date.today().isoformat(),
            **earnings,
        }
    except Exception:
        return {}


def _parse_earnings(t: yf.Ticker) -> dict:
    """Extract next upcoming earnings date from yfinance calendar."""
    try:
        cal = t.calendar
        if not cal or "Earnings Date" not in cal:
            return {"next_earnings": None, "earnings_days_ago": None}

        dates = cal["Earnings Date"]
        if not dates:
            return {"next_earnings": None, "earnings_days_ago": None}

        # calendar returns a list of dates (usually 1-2)
        import datetime
        today = datetime.date.today()

        # Find next future and most recent past earnings
        next_dt = None
        last_dt = None
        for d in (dates if isinstance(dates, list) else [dates]):
            if isinstance(d, datetime.datetime):
                d = d.date()
            if d >= today and (next_dt is None or d < next_dt):
                next_dt = d
            if d < today and (last_dt is None or d > last_dt):
                last_dt = d

        days_ago = (today - last_dt).days if last_dt is not None else None

        return {
            "next_earnings": str(next_dt) if next_dt else None,
            "earnings_days_ago": days_ago,
        }
    except Exception:
        return {"next_earnings": None, "earnings_days_ago": None}


def _float_label(shares: int | None) -> str:
    if shares is None:
        return "unknown"
    millions = shares / 1_000_000
    if millions < 10:
        return "micro"
    if millions < 50:
        return "small"
    if millions < 100:
        return "medium"
    return "large"
