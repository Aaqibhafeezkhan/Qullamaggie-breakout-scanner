import http.client
import os
import time

from scanner import universe
from scanner.cache import Cache
from scanner.universe import _is_fund, _is_fund_name, _parse_etf_flags


def test_fund_keywords_catch_real_funds():
    """Names carrying a fund/ETF word are caught."""
    assert _is_fund_name("SPDR S&P 500 ETF TRUST")
    assert _is_fund_name("INVESCO DB AGRICULTURE FUND")
    assert _is_fund_name("ProShares Trust II")
    assert _is_fund_name("Hashdex Nasdaq CME Crypto Index ETF")
    assert _is_fund_name("21Shares Dogecoin ETF")


def test_real_companies_are_not_caught():
    """Names that use a fund-adjacent word for an unrelated reason pass through."""
    assert not _is_fund_name("NORTHERN TRUST CORP")
    assert not _is_fund_name("ALTISOURCE PORTFOLIO SOLUTIONS S.A.")
    assert not _is_fund_name("CONSUMER PORTFOLIO SERVICES, INC.")
    # REITs: "...Realty Trust" / "...Income Trust" are operating companies
    assert not _is_fund_name("ACADIA REALTY TRUST")
    assert not _is_fund_name("UNIVERSAL HEALTH REALTY INCOME TRUST")
    assert not _is_fund_name("OFFICE PROPERTIES INCOME TRUST")


def test_trust_named_closed_end_funds_caught_by_phrase():
    """Closed-end funds named "...Trust"/"...Portfolio" carry no keyword and
    are ETF=N in Nasdaq Trader's data; Yahoo types them EQUITY, so the
    quote_type backstop can't catch them either. The phrase list does."""
    assert _is_fund_name("GABELLI EQUITY TRUST INC")
    assert _is_fund_name("BlackRock Technology & Private Equity Term Trust")
    assert _is_fund_name("NUVEEN SELECT TAX FREE INCOME PORTFOLIO")
    assert _is_fund_name("Sprott Physical Gold & Silver Trust")
    assert _is_fund_name("Teucrium Commodity Trust")


def test_closed_end_fund_caught_by_keyword_not_flag():
    """USA, IGR, ACP, ADX: real closed-end funds, confirmed ETF=N in
    Nasdaq Trader's own data. Caught by the keyword signal alone."""
    assert _is_fund("Liberty All-Star Equity Fund Common Stock", "USA", etf_flagged=set())
    assert _is_fund("CBRE Global Real Estate Income Fund", "IGR", etf_flagged=set())


def test_keyword_free_etf_caught_by_flag_not_keyword():
    """A keyword- and phrase-free ETF name is caught by the ETF flag alone."""
    assert _is_fund("Example Commodity Pool LP", "EXMP", etf_flagged={"EXMP"})
    assert not _is_fund("Example Commodity Pool LP", "EXMP", etf_flagged=set())


def test_real_company_caught_by_neither_signal():
    """A real stock, no keyword match and not ETF-flagged, passes."""
    assert not _is_fund("Northern Trust Corporation", "NTRS", etf_flagged=set())


# ── Nasdaq Trader parsing ────────────────────────────────────────────────

_HEADER = (
    "Nasdaq Traded|Symbol|Security Name|Listing Exchange|Market Category|ETF|"
    "Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares"
)
_NASDAQ_TEXT = "\r\n".join([
    _HEADER,
    "Y|WEAT|Teucrium Wheat Fund ETV|P| |Y|100|N||WEAT|WEAT|N",
    "Y|GAB|Gabelli Equity Trust, Inc. (The) Common Stock|N| |N|100|N||GAB|GAB|N",
    "Y|ZXZZT|NASDAQ TEST STOCK|Q|G|Y|100|Y|N||ZXZZT|N",
    "Y|BAD|truncated row",
    "File Creation Time: 1028202618:02|||||",
])


def test_parse_etf_flags_reads_columns_by_header():
    """ETF=Y rows are flagged; ETF=N, test issues, short rows and the
    footer are not."""
    assert _parse_etf_flags(_NASDAQ_TEXT) == {"WEAT"}


def test_parse_etf_flags_rejects_unknown_layout():
    """A header without the expected columns (or an HTML error page) yields
    nothing rather than reading whatever sits at the old positions."""
    assert _parse_etf_flags("<html>Service unavailable</html>") == set()
    renamed = _NASDAQ_TEXT.replace("|ETF|", "|IsETF|", 1)
    assert _parse_etf_flags(renamed) == set()


def _sec_rows():
    return [
        [1, "APPLE INC", "AAPL", "Nasdaq"],
        [2, "Example Commodity Pool LP", "EXMP", "NYSE"],
        [3, "GABELLI EQUITY TRUST INC", "GAB", "NYSE"],
    ]


def test_ticker_list_not_cached_when_flags_unavailable(tmp_path, monkeypatch):
    """No flags (download failed, no stale copy) → the list is used for this
    run only, so a keyword-free ETF can't sit in the cache for 7 days."""
    cache = Cache(tmp_path)
    monkeypatch.setattr(universe, "_download_sec_data", _sec_rows)
    monkeypatch.setattr(universe, "_http_get", lambda *a, **k: None)

    assert universe.fetch_ticker_list(cache) == ["AAPL", "EXMP"]
    assert not (tmp_path / "ticker_list_v3.json").exists()


def test_stale_flags_used_when_download_fails(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    cache.set_json("etf_flags", ["EXMP"])
    old = time.time() - 30 * 86400
    os.utime(tmp_path / "etf_flags.json", (old, old))
    monkeypatch.setattr(universe, "_download_sec_data", _sec_rows)
    monkeypatch.setattr(universe, "_http_get", lambda *a, **k: None)

    assert universe.fetch_ticker_list(cache) == ["AAPL"]
    assert (tmp_path / "ticker_list_v3.json").exists()


def test_ticker_list_cache_tracks_filter_config(tmp_path, monkeypatch):
    """A cached list built under a different filter config (or the old
    plain-list format) is rebuilt instead of reused."""
    cache = Cache(tmp_path)
    cache.set_json("etf_flags", ["EXMP"])
    monkeypatch.setattr(universe, "_download_sec_data", _sec_rows)

    cache.set_json("ticker_list_v3", ["AAPL", "EXMP", "GAB"])
    assert universe.fetch_ticker_list(cache) == ["AAPL"]

    cache.set_json("ticker_list_v3", {"filter": "stale", "tickers": ["GAB"]})
    assert universe.fetch_ticker_list(cache) == ["AAPL"]

    monkeypatch.setattr(universe, "_download_sec_data", lambda: [])
    assert universe.fetch_ticker_list(cache) == ["AAPL"]  # served from cache


def test_http_get_returns_none_on_read_errors(monkeypatch):
    """Read timeouts and dropped connections aren't URLErrors; they must
    still degrade to a warning instead of crashing step 1."""
    for exc in (TimeoutError("timed out"), http.client.IncompleteRead(b"")):
        def boom(*a, exc=exc, **k):
            raise exc
        monkeypatch.setattr(universe.urllib.request, "urlopen", boom)
        assert universe._http_get("https://example.invalid", "test") is None
