from scanner.universe import _is_fund_name
from scanner.universe import _is_fund


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


def test_known_filter_gap_nxp_passes_through():
    """NXP (Nuveen Select Tax Free Income Portfolio) is a real closed-end
    fund that the name filter does not catch, since PORTFOLIO was dropped
    from FUND_NAME_KEYWORDS to avoid false positives on real companies
    (Altisource Portfolio Solutions, Consumer Portfolio Services). 
    This is a known, accepted gap — covered by the quote_type backstop in
    signals.py if the ticker ever reaches the top 50."""
    assert not _is_fund_name("NUVEEN SELECT TAX FREE INCOME PORTFOLIO")


def test_closed_end_fund_caught_by_keyword_not_flag():
    """USA, IGR, ACP, ADX: real closed-end funds, confirmed ETF=N in
    Nasdaq Trader's own data. Caught by the keyword signal alone."""
    assert _is_fund("Liberty All-Star Equity Fund Common Stock", "USA", etf_flagged=set())
    assert _is_fund("CBRE Global Real Estate Income Fund", "IGR", etf_flagged=set())


def test_keyword_free_etf_caught_by_flag_not_keyword():
    """WEAT: real ETF, SEC name is "Teucrium Commodity Trust" -- no
    fund keyword present. Caught by the ETF flag signal alone."""
    assert _is_fund("Teucrium Commodity Trust", "WEAT", etf_flagged={"WEAT"})
    assert not _is_fund("Teucrium Commodity Trust", "WEAT", etf_flagged=set())


def test_real_company_caught_by_neither_signal():
    """A real stock, no keyword match and not ETF-flagged, passes."""
    assert not _is_fund("Northern Trust Corporation", "NTRS", etf_flagged=set())