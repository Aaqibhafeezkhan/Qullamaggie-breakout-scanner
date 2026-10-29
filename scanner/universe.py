"""Fetch and filter the US stock universe (NYSE + NASDAQ common stocks)."""

import hashlib
import http.client
import json
import re
import urllib.request

from .cache import Cache
from .config import (
    CACHE_UNIVERSE_TTL_HOURS,
    FUND_NAME_KEYWORDS,
    FUND_NAME_PHRASES,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    VOLUME_AVG_PERIOD,
)

# SEC EDGAR provides all SEC-registered companies with exchange info.
_SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
_NASDAQ_TRADER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
_TARGET_EXCHANGES = {"NYSE", "Nasdaq"}
_TICKER_LIST_KEY = "ticker_list_v3"


def fetch_ticker_list(cache: Cache) -> list[str]:
    """Return sorted list of NYSE/NASDAQ common-stock tickers.

    Uses SEC EDGAR's company_tickers_exchange.json which includes
    exchange information, minus funds (name filter + Nasdaq Trader ETF
    flag). Results cached for 7 days; the cache records the fund-filter
    config, so editing FUND_NAME_KEYWORDS/PHRASES rebuilds it next run.
    """
    signature = _fund_filter_signature()
    cached = cache.get_json(_TICKER_LIST_KEY, max_age_hours=CACHE_UNIVERSE_TTL_HOURS)
    if isinstance(cached, dict) and cached.get("filter") == signature:
        return cached["tickers"]

    etf_flagged = _fetch_etf_flagged_tickers(cache)
    result = sorted(_fetch_sec_tickers(etf_flagged))
    if not result:
        raise RuntimeError(
            "Failed to fetch any tickers. Check your internet connection."
        )

    # Without ETF flags, keyword-free ETFs (WEAT, SLV, ...) stay in the
    # list — use it for this run only so the next run retries the download.
    if etf_flagged:
        cache.set_json(_TICKER_LIST_KEY, {"filter": signature, "tickers": result})
    else:
        print("  Warning: no Nasdaq ETF flags — ticker list not cached, "
              "keyword-free ETFs may pass this run")
    return result


def fetch_ticker_info(cache: Cache) -> dict[str, str]:
    """Return {ticker: company_name} for all NYSE/NASDAQ stocks.

    Uses the same SEC EDGAR data as fetch_ticker_list. Cached separately
    from the ticker list.
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
    """True if the company name carries a fund keyword or fund phrase."""
    words = re.findall(r"[A-Z]+", name.upper())
    if set(words) & FUND_NAME_KEYWORDS:
        return True
    padded = f" {' '.join(words)} "
    return any(f" {phrase} " in padded for phrase in FUND_NAME_PHRASES)


def _is_fund(name: str, ticker: str, etf_flagged: set[str]) -> bool:
    """True if a ticker should be excluded as a fund, by either signal.

    The name filter catches closed-end funds (USA, IGR, GAB, BTX, NXP --
    all ETF=N in Nasdaq Trader's own data). The flag catches true ETFs
    with a keyword-free name (WEAT: "Teucrium Commodity Trust", CORN,
    CANE). Neither signal alone is sufficient.
    """
    if ticker in etf_flagged:
        return True
    return isinstance(name, str) and _is_fund_name(name)


def _fund_filter_signature() -> str:
    """Short hash of the fund-filter config stored with the ticker list."""
    config = json.dumps([sorted(FUND_NAME_KEYWORDS), list(FUND_NAME_PHRASES)])
    return hashlib.sha1(config.encode()).hexdigest()[:12]


def _http_get(url: str, what: str, accept: str | None = None) -> str | None:
    """GET *url* as text, or None (with a warning) on any fetch error.

    Catches more than URLError: a read timeout (TimeoutError), a dropped
    connection (IncompleteRead) or a bad byte (UnicodeDecodeError) must
    not crash step 1 of the scan.
    """
    headers = {"User-Agent": "qullamaggie-scanner vlad@example.com"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()
    except (OSError, http.client.HTTPException, UnicodeDecodeError) as exc:
        print(f"  Warning: could not fetch {what}: {exc}")
        return None


def _download_sec_data() -> list[list]:
    """Download SEC EDGAR company tickers with exchange info.

    Returns list of [cik, name, ticker, exchange] rows.
    """
    text = _http_get(_SEC_URL, "SEC data", accept="application/json")
    if text is None:
        return []
    try:
        return json.loads(text).get("data", [])
    except json.JSONDecodeError as exc:
        print(f"  Warning: could not fetch SEC data: {exc}")
        return []


def _parse_etf_flags(text: str) -> set[str]:
    """Return symbols the Nasdaq Trader symbol directory flags ETF=Y.

    A true open-end ETF (WEAT: "Teucrium Commodity Trust", CORN, CANE)
    often carries no fund keyword in its SEC-registered name, so
    _is_fund_name misses it. This file's own ETF column flags it
    directly, Y or N, independent of naming. It does NOT flag closed-end
    funds (USA, IGR, GAB, BTX all show ETF=N here) -- those stay caught
    by the name filter, so this supplements it rather than replacing
    it. Also missing some real NYSE tickers entirely (confirmed: BLFS,
    DBRG, DRCT), so SEC EDGAR stays the base universe source.

    Columns are located by header name, so a changed layout yields an
    empty set (which the caller treats as a failed download) instead of
    silently reading the wrong column. Test issues are excluded.
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return set()
    header = lines[0].split("|")
    try:
        sym, etf, test = [header.index(c) for c in ("Symbol", "ETF", "Test Issue")]
    except ValueError:
        print(f"  Warning: unexpected Nasdaq Trader header: {lines[0][:120]}")
        return set()
    width = max(sym, etf, test) + 1
    flagged: set[str] = set()
    # Last line is "File Creation Time: ..." footer, not data.
    for line in lines[1:-1]:
        row = line.split("|")
        if len(row) >= width and row[etf] == "Y" and row[test] == "N":
            flagged.add(row[sym])
    return flagged


def _fetch_etf_flagged_tickers(cache: Cache | None = None) -> set[str]:
    """Return tickers Nasdaq Trader's own data flags as ETF=Y.

    Cached 7 days like the ticker list, since this changes rarely. If the
    download fails or parses to nothing, falls back to the last cached
    flag set regardless of age. cache is optional so this still works
    standalone (tests, ad hoc checks); without one the flag set is
    fetched fresh, uncached.
    """
    if cache is not None:
        cached = cache.get_json("etf_flags", max_age_hours=CACHE_UNIVERSE_TTL_HOURS)
        if cached is not None:
            return set(cached)

    text = _http_get(_NASDAQ_TRADER_URL, "Nasdaq Trader symbol directory")
    flagged = _parse_etf_flags(text) if text is not None else set()

    if cache is not None:
        if flagged:
            cache.set_json("etf_flags", sorted(flagged))
        else:
            stale = cache.get_json("etf_flags", max_age_hours=float("inf"))
            if stale:
                print("  Warning: using stale Nasdaq ETF flags")
                return set(stale)

    return flagged


def _fetch_sec_tickers(etf_flagged: set[str] = frozenset()) -> set[str]:
    """Return set of common-stock tickers on NYSE/NASDAQ from SEC EDGAR,
    excluding funds (name filter, plus *etf_flagged* symbols)."""
    rows = _download_sec_data()
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
