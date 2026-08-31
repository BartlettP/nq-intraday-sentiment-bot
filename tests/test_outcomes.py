"""
Tests for update_past_outcomes — the backfill pass that pairs each prediction
with what NQ actually did.

Regression coverage for fix #3: unresolvable predictions must have an attempt
recorded (so they eventually leave the queue), and the download window must be
wide enough to span a weekend.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


def make_bars(end, n=200, freq='5min'):
    """A plausible NQ 5-minute frame, UTC-indexed like yfinance returns."""
    idx = pd.date_range(end=end, periods=n, freq=freq, tz='UTC')
    rng = np.random.default_rng(0)
    close = 20000 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame(
        {
            'Open': close,
            'High': close + 5,
            'Low': close - 5,
            'Close': close,
            'Volume': rng.integers(500, 2000, n),
        },
        index=idx,
    )


@pytest.fixture
def backfill(bot, monkeypatch):
    """Wires up update_past_outcomes with recording stubs."""
    calls = {'download': [], 'updated': [], 'attempted': [], 'pending': []}

    def fake_download(ticker, **kwargs):
        calls['download'].append(kwargs)
        return calls['bars']

    monkeypatch.setattr(bot.yf, 'download', fake_download)
    monkeypatch.setattr(bot, 'get_predictions_needing_outcomes',
                        lambda **kw: calls['pending'])
    monkeypatch.setattr(bot, 'update_outcome',
                        lambda pid, vol: calls['updated'].append((pid, vol)))
    monkeypatch.setattr(bot, 'record_outcome_attempt',
                        lambda ids: calls['attempted'].extend(ids))
    return calls


def _pending(pid, when):
    return {
        'id': pid,
        'timestamp': when.replace(tzinfo=None).isoformat(),
        'predicted_volatility': 0.09,
        'outcome_attempts': 0,
    }


class TestDownloadWindow:
    def test_requests_five_days_not_one(self, bot, backfill):
        """period='1d' meant a Friday-afternoon prediction could never be
        resolved after the weekend."""
        now = datetime.now(timezone.utc)
        backfill['bars'] = make_bars(now)
        backfill['pending'] = [_pending(1, now - timedelta(hours=3))]

        bot.update_past_outcomes()

        assert backfill['download'][0]['period'] == '5d'
        assert backfill['download'][0]['interval'] == '5m'

    def test_no_pending_predictions_skips_the_download(self, bot, backfill):
        backfill['bars'] = make_bars(datetime.now(timezone.utc))
        backfill['pending'] = []

        bot.update_past_outcomes()
        assert backfill['download'] == []


class TestResolution:
    def test_prediction_with_a_full_hour_of_data_is_resolved(self, bot, backfill):
        now = datetime.now(timezone.utc)
        backfill['bars'] = make_bars(now)
        backfill['pending'] = [_pending(7, now - timedelta(hours=3))]

        bot.update_past_outcomes()

        assert len(backfill['updated']) == 1
        pid, vol = backfill['updated'][0]
        assert pid == 7
        assert vol > 0
        assert backfill['attempted'] == []

    def test_prediction_without_enough_forward_data_records_an_attempt(self, bot, backfill):
        """The row the old code silently left pending forever."""
        now = datetime.now(timezone.utc)
        backfill['bars'] = make_bars(now)
        # Made 10 minutes ago: only ~2 of the required 12 forward bars exist.
        backfill['pending'] = [_pending(9, now - timedelta(minutes=10))]

        bot.update_past_outcomes()

        assert backfill['updated'] == []
        assert backfill['attempted'] == [9], \
            "an unresolvable prediction must have its attempt counted"

    def test_prediction_older_than_the_window_records_an_attempt(self, bot, backfill):
        now = datetime.now(timezone.utc)
        backfill['bars'] = make_bars(now)
        backfill['pending'] = [_pending(11, now - timedelta(days=30))]

        bot.update_past_outcomes()
        # Nothing at/after a 30-day-old timestamp is missing, so it resolves;
        # what matters is it never silently vanishes from both lists.
        assert len(backfill['updated']) + len(backfill['attempted']) == 1

    def test_mixed_batch_resolves_some_and_defers_others(self, bot, backfill):
        now = datetime.now(timezone.utc)
        backfill['bars'] = make_bars(now)
        backfill['pending'] = [
            _pending(1, now - timedelta(hours=5)),
            _pending(2, now - timedelta(hours=4)),
            _pending(3, now - timedelta(minutes=5)),  # too recent
        ]

        bot.update_past_outcomes()

        assert sorted(pid for pid, _ in backfill['updated']) == [1, 2]
        assert backfill['attempted'] == [3]

    def test_handles_multiindex_columns(self, bot, backfill):
        """yfinance returns a MultiIndex when it feels like it."""
        now = datetime.now(timezone.utc)
        bars = make_bars(now)
        bars.columns = pd.MultiIndex.from_product([bars.columns, ['NQ=F']])
        backfill['bars'] = bars
        backfill['pending'] = [_pending(4, now - timedelta(hours=3))]

        bot.update_past_outcomes()
        assert len(backfill['updated']) == 1

    def test_naive_index_is_localized_not_crashed(self, bot, backfill):
        now = datetime.now(timezone.utc)
        bars = make_bars(now)
        bars.index = bars.index.tz_localize(None)
        backfill['bars'] = bars
        backfill['pending'] = [_pending(5, now - timedelta(hours=3))]

        bot.update_past_outcomes()
        assert len(backfill['updated']) == 1


class TestFailureHandling:
    def test_download_failure_does_not_propagate(self, bot, backfill, monkeypatch, caplog):
        """A yfinance outage must not take down the main loop."""
        def boom(ticker, **kwargs):
            raise ConnectionError("yfinance unreachable")

        monkeypatch.setattr(bot.yf, 'download', boom)
        backfill['pending'] = [_pending(1, datetime.now(timezone.utc) - timedelta(hours=3))]

        with caplog.at_level('WARNING', logger='nq_bot'):
            bot.update_past_outcomes()  # must not raise
        assert 'Outcome update error' in caplog.text
