from datetime import date

from scanner import profile
from scanner.cache import Cache
from scanner.profile import _needs_refresh, earnings_days_ago

TODAY = date(2026, 10, 28)


def _entry(**overrides):
    base = {
        "sector": "Technology",
        "quote_type": "EQUITY",
        "fetched_at": "2026-10-20",
        "next_earnings": "2026-11-05",
        "earnings_days_ago": 80,
    }
    return {**base, **overrides}


def test_needs_refresh():
    assert _needs_refresh({}, TODAY)                                # never fetched / failed
    assert _needs_refresh(_entry(quote_type=None), TODAY)           # stub info response
    assert _needs_refresh(_entry(fetched_at=None), TODAY)           # pre-timestamp entry
    assert _needs_refresh(_entry(fetched_at="2026-09-01"), TODAY)   # older than 30 days
    assert _needs_refresh(_entry(next_earnings="2026-10-27"), TODAY)  # earnings passed
    assert not _needs_refresh(_entry(), TODAY)
    assert not _needs_refresh(_entry(next_earnings=None), TODAY)


def test_failed_refetch_keeps_good_entry(tmp_path, monkeypatch):
    """A refetch that fails ({}) or returns a stub must not wipe sector/float
    data already cached; a good refetch replaces it."""
    cache = Cache(tmp_path)
    old = _entry(fetched_at=None)  # due for refresh
    cache.set_json("profiles", {"AAA": old, "BBB": old, "CCC": old})
    results = {
        "AAA": {},
        "BBB": {"sector": None, "quote_type": None},
        "CCC": {"sector": "Energy", "quote_type": "EQUITY", "fetched_at": "2026-10-28"},
    }
    monkeypatch.setattr(profile, "_fetch_one", lambda t: results[t])

    profiles = profile.fetch_stock_profiles(["AAA", "BBB", "CCC"], cache)

    assert profiles["AAA"] == old
    assert profiles["BBB"] == old
    assert profiles["CCC"]["sector"] == "Energy"


def test_earnings_days_ago_ages_from_fetch_date():
    assert earnings_days_ago(_entry(), TODAY) == 88
    assert earnings_days_ago(_entry(fetched_at=None), TODAY) == 80
    assert earnings_days_ago(_entry(earnings_days_ago=None), TODAY) is None
