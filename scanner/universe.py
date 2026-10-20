"""Fetch and filter the US stock universe (NYSE + NASDAQ common stocks)."""

import json
import re
import urllib.request
import urllib.error

from .cache import Cache
from .config import (
    CACHE_UNIVERSE_TTL_HOURS,
    FUND_NAME_KEYWORDS,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    VOLUME_AVG_PERIOD,
)

# SEC EDGAR provides all SEC-registered companies with exchange info.
_SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_NASDAQ_TRADER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
_TARGET_EXCHANGES = {"NYSE", "Nasdaq"}


def fetch_ticker_list(cache: Cache) -> list[str]:
    """Return sorted list of NYSE/NASDAQ common-stock tickers.

    Uses SEC EDGAR's company_tickers_exchange.json which includes
    exchange information. Results cached for 7 days.
    """
    cached = cache.get_json("ticker_list_v3", max_age_hours=CACHE_UNIVERSE_TTL_HOURS)
    if cached is not None:
        return cached

    tickers = _fetch_sec_tickers(cache)

    result = sorted(tickers)
    if not result:
        raise RuntimeError(
            "Failed to fetch any tickers. Check your internet connection."
        )

    cache.set_json("ticker_list_v3", result)
    return result


def fetch_ticker_info(cache: Cache) -> dict[str, str]:
    """Return {ticker: company_name} for all NYSE/NASDAQ stocks.

    Uses the same SEC EDGAR data as fetch_ticker_list. Cached separately
    so the ticker list can be a simple JSON array.
    """
    cached = cache.get_json(
        "ticker_names", max_age_hours=CACHE_UNIVERSE_TTL_HOURS
    )
    if cached is not None:
        return cached

    info = _fetch_sec_ticker_names()
    if info:
        cache.set_json("ticker_names", info)
    return info


def filter_universe(tickers, price_data):
    """Keep only tickers where last close >= MIN_PRICE and
    20-day average volume >= MIN_AVG_VOLUME.

    Parameters
    ----------
    tickers : list[str]
    price_data : dict[str, DataFrame]  — keyed by ticker, each DF has
        columns including 'Close' and 'Volume' with a DatetimeIndex.

    Returns
    -------
    list[str] — tickers that pass filters, sorted alphabetically.
    """
    passed = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or len(df) < VOLUME_AVG_PERIOD:
            continue
        recent = df.tail(VOLUME_AVG_PERIOD)
        avg_vol = recent["Volume"].mean()
        last_close = df["Close"].iloc[-1]
        if last_close >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME:
            passed.append(ticker)
    return sorted(passed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _is_common_stock_ticker(symbol: str) -> bool:
    """Heuristic: common stocks have 1-5 uppercase-letter tickers."""
    return bool(_TICKER_RE.match(symbol))

def _is_fund_name(name: str) -> bool:
    """True if the company name carries a fund/ETF/trust word."""
    words = set(re.findall(r"[A-Z]+", name.upper()))
    return bool(words & FUND_NAME_KEYWORDS)


def _is_fund(name: str, ticker: str, etf_flagged: set[str]) -> bool:
    """True if a ticker should be excluded as a fund, by either signal.

    Keyword match catches closed-end funds (USA, IGR, ACP, ADX -- all
    ETF=N in Nasdaq Trader's own data). The flag catches true ETFs with
    a keyword-free name (WEAT: "Teucrium Commodity Trust", CORN, CANE).
    Neither signal alone is sufficient.
    """
    if ticker in etf_flagged:
        return True
    return isinstance(name, str) and _is_fund_name(name)


def _download_sec_data() -> list[list]:
    """Download SEC EDGAR company tickers with exchange info.

    Returns list of [cik, name, ticker, exchange] rows.
    """
    req = urllib.request.Request(
        _SEC_URL,
        headers={
            "User-Agent": "qullamaggie-scanner vlad@example.com",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
            return payload.get("data", [])
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"  Warning: could not fetch SEC data: {exc}")
        return []


def _download_nasdaq_traded() -> list[list[str]]:
    """Download the Nasdaq Trader symbol directory (nasdaqtraded.txt).

    A true open-end ETF (WEAT: "Teucrium Commodity Trust", CORN, CANE)
    often carries no fund keyword in its SEC-registered name, so
    _is_fund_name misses it. This file's own ETF column flags it
    directly, Y or N, independent of naming. It does NOT flag closed-end
    funds (USA, IGR, ACP, ADX all show ETF=N here) -- those stay caught
    by the keyword filter, so this supplements it rather than replacing
    it. Also missing some real NYSE tickers entirely (confirmed: BLFS,
    DBRG, DRCT), so SEC EDGAR stays the base universe source.
    """
    req = urllib.request.Request(
        _NASDAQ_TRADER_URL,
        headers={"User-Agent": "qullamaggie-scanner vlad@example.com"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode()
    except urllib.error.URLError as exc:
        print(f"  Warning: could not fetch Nasdaq Trader symbol directory: {exc}")
        return []

    lines = text.strip().split("\n")
    # Last line is "File Creation Time: ..." footer, not data.
    return [row for line in lines[1:-1] if len(row := line.split("|")) >= 8]


def _fetch_etf_flagged_tickers(cache: Cache | None = None) -> set[str]:
    """Return tickers Nasdaq Trader's own data flags as ETF=Y.

    Cached 7 days like the ticker list, since this changes rarely.
    cache is optional so this still works standalone (tests, ad hoc
    checks); without one the flag set is fetched fresh, uncached.
    """
    if cache is not None:
        cached = cache.get_json("etf_flags", max_age_hours=CACHE_UNIVERSE_TTL_HOURS)
        if cached is not None:
            return set(cached)

    rows = _download_nasdaq_traded()
    flagged = {row[1] for row in rows if row[5] == "Y" and row[7] == "N"}

    if cache is not None and flagged:
        cache.set_json("etf_flags", sorted(flagged))

    return flagged


def _fetch_sec_tickers(cache: Cache | None = None) -> set[str]:
    """Return set of common-stock tickers on NYSE/NASDAQ from SEC EDGAR."""
    rows = _download_sec_data()
    etf_flagged = _fetch_etf_flagged_tickers(cache)
    tickers: set[str] = set()
    for row in rows:
        if len(row) < 4:
            continue
        name = row[1]
        ticker = row[2]
        exchange = row[3]
        if exchange not in _TARGET_EXCHANGES:
            continue
        if not isinstance(ticker, str) or not _is_common_stock_ticker(ticker):
            continue
        if _is_fund(name, ticker, etf_flagged):
            continue
        tickers.add(ticker)
    return tickers

def _fetch_sec_ticker_names() -> dict[str, str]:
    """Return {ticker: company_name} for NYSE/NASDAQ stocks."""
    rows = _download_sec_data()
    names: dict[str, str] = {}
    for row in rows:
        if len(row) < 4:
            continue
        ticker = row[2]
        name = row[1]
        exchange = row[3]
        if exchange not in _TARGET_EXCHANGES:
            continue
        if isinstance(ticker, str) and _is_common_stock_ticker(ticker):
            names[ticker] = name
    return names