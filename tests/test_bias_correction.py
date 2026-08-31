"""
Tests for the per-hour volatility bias correction.

The v2 model predicts decay at nearly every hour and can't represent the
volatility ramp across the 9:30 open (training was rth_only). These cover the
correction that supplies the missing diurnal component — and in particular the
property that keeps it from compounding on itself.
"""
from datetime import datetime, timedelta, timezone

import pytest
import pytz

ET = pytz.timezone('US/Eastern')


def insert_resolved(db, et_hour, current, predicted, actual, days_ago=1, raw=None,
                    correction=None):
    """Insert a resolved prediction stamped at a given ET hour."""
    when_et = ET.localize(
        datetime(2026, 6, 10, et_hour, 0) - timedelta(days=days_ago)
    )
    when_utc = when_et.astimezone(timezone.utc).replace(tzinfo=None)
    with db.get_connection() as conn:
        cur = conn.execute(
            'INSERT INTO predictions (timestamp, predicted_volatility, '
            'current_volatility, actual_volatility, predicted_volatility_raw, '
            'bias_correction) VALUES (?, ?, ?, ?, ?, ?)',
            (when_utc.isoformat(), predicted, current, actual, raw, correction)
        )
        conn.commit()
        return cur.lastrowid


class TestInsufficientHistory:
    def test_no_history_gives_no_corrections(self, db):
        assert db.get_hourly_bias_corrections() == {}

    def test_single_hour_correction_defaults_to_zero(self, db):
        assert db.get_bias_correction_for_hour(9) == 0.0

    def test_below_min_samples_the_hour_is_omitted(self, db):
        for i in range(db.MIN_BIAS_SAMPLES - 1):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11, days_ago=i + 1)
        assert 9 not in db.get_hourly_bias_corrections()
        assert db.get_bias_correction_for_hour(9) == 0.0

    def test_at_min_samples_the_hour_appears(self, db):
        for i in range(db.MIN_BIAS_SAMPLES):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11, days_ago=i + 1)
        assert 9 in db.get_hourly_bias_corrections()

    def test_unresolved_rows_do_not_count(self, db):
        """A prediction with no outcome carries no information about bias."""
        for i in range(db.MIN_BIAS_SAMPLES + 4):
            with db.get_connection() as conn:
                when = ET.localize(datetime(2026, 6, 10, 9, 0) - timedelta(days=i + 1))
                conn.execute(
                    'INSERT INTO predictions (timestamp, predicted_volatility, '
                    'current_volatility, actual_volatility) VALUES (?, ?, ?, NULL)',
                    (when.astimezone(timezone.utc).replace(tzinfo=None).isoformat(),
                     0.05, 0.06)
                )
                conn.commit()
        assert db.get_hourly_bias_corrections() == {}


class TestCorrectionValue:
    def test_recovers_a_known_underprediction(self, db):
        """Model says -0.01, reality is +0.05 => correction should be +0.06."""
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11, days_ago=i + 1)
        # predicted delta = -0.01, actual delta = +0.05, residual = +0.06
        assert db.get_bias_correction_for_hour(9) == pytest.approx(0.06, abs=1e-9)

    def test_recovers_a_known_overprediction(self, db):
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            insert_resolved(db, 14, current=0.10, predicted=0.12, actual=0.08, days_ago=i + 1)
        # predicted delta +0.02, actual delta -0.02, residual -0.04
        assert db.get_bias_correction_for_hour(14) == pytest.approx(-0.04, abs=1e-9)

    def test_hours_are_independent(self, db):
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11, days_ago=i + 1)
            insert_resolved(db, 15, current=0.10, predicted=0.10, actual=0.06, days_ago=i + 1)

        corrections = db.get_hourly_bias_corrections()
        assert corrections[9] > 0, "morning ramp should correct upward"
        assert corrections[15] < 0, "afternoon decay should correct downward"

    def test_averages_across_observations(self, db):
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            actual = 0.11 if i % 2 == 0 else 0.07   # residual +0.06 / +0.02
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=actual, days_ago=i + 1)
        assert db.get_bias_correction_for_hour(9) == pytest.approx(0.04, abs=1e-9)

    def test_uses_eastern_hour_not_utc(self, db):
        """Timestamps are stored UTC; corrections must key off the ET hour."""
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11, days_ago=i + 1)

        corrections = db.get_hourly_bias_corrections()
        assert 9 in corrections, "should be keyed by ET hour"
        assert 13 not in corrections and 14 not in corrections, \
            "must not be keyed by the raw UTC hour"


class TestNoCompounding:
    """The correction is learned from the RAW model output. Learning from
    already-corrected predictions would fold each correction into the next and
    diverge."""

    def test_learns_from_raw_when_a_correction_was_applied(self, db):
        n = db.MIN_BIAS_SAMPLES
        # Published 0.11 = raw 0.05 + correction 0.06. Reality also 0.11, so the
        # correction was exactly right and the next one must be unchanged.
        for i in range(n):
            insert_resolved(db, 9, current=0.06, predicted=0.11, actual=0.11,
                            raw=0.05, correction=0.06, days_ago=i + 1)

        assert db.get_bias_correction_for_hour(9) == pytest.approx(0.06, abs=1e-9), \
            "a correct correction must reproduce itself, not double"

    def test_repeated_application_converges_rather_than_diverging(self, db):
        """Simulate several cycles of applying and then re-learning."""
        n = db.MIN_BIAS_SAMPLES
        current, raw_pred, actual = 0.06, 0.05, 0.11

        history = []
        for _cycle in range(5):
            correction = db.get_bias_correction_for_hour(9)
            published = raw_pred + correction
            history.append(correction)
            for i in range(n):
                insert_resolved(db, 9, current=current, predicted=published,
                                actual=actual, raw=raw_pred, correction=correction,
                                days_ago=len(history) * 20 + i + 1)

        assert history[-1] == pytest.approx(0.06, abs=1e-9)
        assert max(history) <= 0.06 + 1e-9, f"correction diverged: {history}"

    def test_legacy_rows_without_raw_fall_back_to_published(self, db):
        """Rows written before the correction existed have NULL raw; for those
        the published value IS the raw model output."""
        n = db.MIN_BIAS_SAMPLES
        for i in range(n):
            insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11,
                            raw=None, days_ago=i + 1)
        assert db.get_bias_correction_for_hour(9) == pytest.approx(0.06, abs=1e-9)


class TestPersistence:
    def test_insert_prediction_stores_raw_and_correction(self, db, sentiment):
        vol = {
            'predicted_next_hour': 0.11,
            'predicted_next_hour_raw': 0.05,
            'bias_correction': 0.06,
            'current': 0.06,
            'last_4h_avg': 0.06,
            'change': 0.05,
            'nq_price': 20000.0,
        }
        pid = db.insert_prediction(sentiment(20.0), vol, "TEST")
        with db.get_connection() as conn:
            row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pid,)).fetchone()

        assert row['predicted_volatility'] == 0.11, "published value goes in the main column"
        assert row['predicted_volatility_raw'] == 0.05
        assert row['bias_correction'] == 0.06

    def test_volatility_without_correction_keys_still_inserts(self, db, sentiment, volatility):
        """The plain fixture dict has no raw/correction keys."""
        pid = db.insert_prediction(sentiment(10.0), volatility(0.09), "TEST")
        with db.get_connection() as conn:
            row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pid,)).fetchone()
        assert row['predicted_volatility_raw'] is None
        assert row['bias_correction'] is None

    def test_migration_adds_columns_without_losing_rows(self, db):
        insert_resolved(db, 9, current=0.06, predicted=0.05, actual=0.11)
        db.init_db()   # re-run migration
        with db.get_connection() as conn:
            cols = {r['name'] for r in conn.execute('PRAGMA table_info(predictions)')}
            n = conn.execute('SELECT COUNT(*) AS n FROM predictions').fetchone()['n']
        assert {'predicted_volatility_raw', 'bias_correction'} <= cols
        assert n == 1


class TestBotApplication:
    """The correction has to actually reach the published prediction."""

    @pytest.fixture
    def predict(self, bot, monkeypatch):
        """predict_intraday_volatility with the model and market data stubbed."""
        import numpy as np
        import pandas as pd

        idx = pd.date_range(end='2026-06-10 18:00', periods=300, freq='5min', tz='UTC')
        rng = np.random.default_rng(1)
        close = 20000 + np.cumsum(rng.normal(0, 6, len(idx)))
        bars = pd.DataFrame(
            {'Open': close, 'High': close + 5, 'Low': close - 5, 'Close': close,
             'Volume': rng.integers(500, 2000, len(idx))}, index=idx)

        monkeypatch.setattr(bot.yf, 'download', lambda *a, **k: bars)

        class FakeScalerX:
            def transform(self, X):
                return X

        class FakeScalerY:
            def inverse_transform(self, y):
                return np.array([[-0.01]])   # model always predicts -0.01

        class FakeModel:
            def predict(self, X, verbose=0):
                return np.array([[0.0]])

        monkeypatch.setattr(bot, 'scaler_X', FakeScalerX())
        monkeypatch.setattr(bot, 'scaler_y', FakeScalerY())
        monkeypatch.setattr(bot, 'volatility_model', FakeModel())
        return bot

    def test_correction_is_added_to_the_published_value(self, predict, monkeypatch):
        monkeypatch.setattr(predict, 'APPLY_BIAS_CORRECTION', True)
        monkeypatch.setattr(predict, 'get_bias_correction_for_hour', lambda h: 0.05)

        result = predict.predict_intraday_volatility()
        assert result is not None
        assert result['bias_correction'] == pytest.approx(0.05)
        assert result['predicted_next_hour'] == pytest.approx(
            result['predicted_next_hour_raw'] + 0.05, abs=1e-9)
        assert result['change'] == pytest.approx(result['change_raw'] + 0.05, abs=1e-9)

    def test_flag_off_publishes_the_raw_value(self, predict, monkeypatch):
        monkeypatch.setattr(predict, 'APPLY_BIAS_CORRECTION', False)
        monkeypatch.setattr(predict, 'get_bias_correction_for_hour',
                            lambda h: pytest.fail("must not be consulted when disabled"))

        result = predict.predict_intraday_volatility()
        assert result['bias_correction'] == 0.0
        assert result['predicted_next_hour'] == pytest.approx(result['predicted_next_hour_raw'])

    def test_a_failing_lookup_does_not_stop_the_prediction(self, predict, monkeypatch):
        """A database problem must never block a signal going out."""
        def boom(h):
            raise RuntimeError("database locked")

        monkeypatch.setattr(predict, 'APPLY_BIAS_CORRECTION', True)
        monkeypatch.setattr(predict, 'get_bias_correction_for_hour', boom)

        result = predict.predict_intraday_volatility()
        assert result is not None
        assert result['bias_correction'] == 0.0
        assert result['predicted_next_hour'] == pytest.approx(result['predicted_next_hour_raw'])

    def test_published_value_is_clamped_at_zero(self, predict, monkeypatch):
        """A large negative correction must not produce negative volatility."""
        monkeypatch.setattr(predict, 'APPLY_BIAS_CORRECTION', True)
        monkeypatch.setattr(predict, 'get_bias_correction_for_hour', lambda h: -99.0)

        result = predict.predict_intraday_volatility()
        assert result['predicted_next_hour'] == 0.0
