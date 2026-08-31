"""
Tests for the shared config (single DB path, one volatility scale) and the
liveness reporting added for monitoring.
"""
from datetime import datetime, timedelta, timezone

import pytest


class TestVolatilityScale:
    def test_boundaries_are_ascending(self):
        import config
        bounds = [b for b, _, _ in config.VOLATILITY_LEVELS]
        assert bounds == sorted(bounds)
        assert bounds[-1] == float('inf'), "the top bucket must be open-ended"

    @pytest.mark.parametrize("vol,expected", [
        (0.01, "Very Low"),
        (0.059, "Very Low"),
        (0.06, "Low"),
        (0.099, "Low"),
        (0.10, "Normal"),
        (0.149, "Normal"),
        (0.15, "High"),
        (0.249, "High"),
        (0.25, "Very High"),
        (5.0, "Very High"),
    ])
    def test_bucket_assignment(self, vol, expected):
        import config
        label, _ = config.volatility_level(vol)
        assert expected in label

    def test_every_bucket_returns_a_description(self):
        import config
        for vol in (0.01, 0.07, 0.12, 0.2, 0.9):
            label, desc = config.volatility_level(vol)
            assert label and desc

    def test_zero_and_negative_do_not_fall_through(self):
        """A clamped-at-zero prediction must still get a label."""
        import config
        assert config.volatility_level(0.0)[0]
        assert config.volatility_level(-1.0)[0]


class TestScaleConsistency:
    """#3: three separate threshold tables used to disagree, so the same
    reading was labelled differently depending on the code path."""

    def test_signal_wording_uses_the_shared_boundaries(self, bot):
        import config
        assert bot.config.VOL_QUIET_MAX == config.VOLATILITY_LEVELS[0][0]
        assert bot.config.VOL_ACTIVE_MIN == config.VOLATILITY_LEVELS[1][0]

    def test_get_volatility_context_agrees_with_the_embed_label(self, bot):
        """These two functions returned different labels for the same input."""
        import config
        for vol in (0.02, 0.05, 0.08, 0.12, 0.2, 0.4):
            context_label, _ = bot.get_volatility_context(vol)
            embed_label, _ = config.volatility_level(vol)
            assert context_label == embed_label, f"disagreement at vol={vol}"

    def test_quiet_and_active_do_not_overlap(self):
        import config
        assert config.VOL_QUIET_MAX < config.VOL_ACTIVE_MIN, \
            "a volatility can't be both 'small moves' and 'big moves'"


class TestDatabasePath:
    def test_database_resolves_through_config(self):
        import config
        import database
        assert database.DB_PATH == config.DB_PATH

    def test_env_var_overrides_the_default(self, monkeypatch, tmp_path):
        """NQ_DB_PATH lets a script point at a downloaded production snapshot."""
        import importlib
        import config

        custom = str(tmp_path / 'custom.db')
        monkeypatch.setenv('NQ_DB_PATH', custom)
        try:
            reloaded = importlib.reload(config)
            assert reloaded.DB_PATH == custom
        finally:
            # Restore the real config for every subsequent test.
            monkeypatch.undo()
            importlib.reload(config)

    def test_default_path_lives_under_data(self, monkeypatch):
        import config
        monkeypatch.delitem(config._ENV, 'NQ_DB_PATH', raising=False)
        assert config.DB_PATH.endswith('predictions.db')
        assert 'data' in config.DB_PATH

    def test_is_production_respects_explicit_env(self, monkeypatch):
        import config
        monkeypatch.setitem(config._ENV, 'NQ_ENV', 'production')
        assert config.is_production() is True
        monkeypatch.setitem(config._ENV, 'NQ_ENV', 'development')
        assert config.is_production() is False


class TestHealthSummary:
    def _insert(self, db, minutes_ago, actual=None):
        when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
        with db.get_connection() as conn:
            cur = conn.execute(
                'INSERT INTO predictions (timestamp, predicted_volatility, '
                'current_volatility, actual_volatility) VALUES (?, ?, ?, ?)',
                (when.replace(tzinfo=None).isoformat(), 0.09, 0.08, actual)
            )
            conn.commit()
            return cur.lastrowid

    def test_empty_database_is_stale(self, db):
        h = db.health_summary()
        assert h['total_predictions'] == 0
        assert h['last_prediction_utc'] is None
        assert h['is_stale'] is True, "an empty database is not healthy"

    def test_recent_prediction_is_fresh(self, db):
        self._insert(db, minutes_ago=10)
        h = db.health_summary(stale_after_minutes=90)
        assert h['is_stale'] is False
        assert h['minutes_since_last'] == pytest.approx(10, abs=2)

    def test_old_prediction_is_stale(self, db):
        self._insert(db, minutes_ago=48 * 60)
        h = db.health_summary(stale_after_minutes=90)
        assert h['is_stale'] is True

    def test_uses_the_newest_row_not_the_first(self, db):
        self._insert(db, minutes_ago=5000)
        self._insert(db, minutes_ago=15)
        h = db.health_summary()
        assert h['minutes_since_last'] == pytest.approx(15, abs=2)
        assert h['is_stale'] is False

    def test_counts_resolved_pending_and_abandoned(self, db):
        self._insert(db, minutes_ago=200, actual=0.07)   # resolved
        pending = self._insert(db, minutes_ago=200)      # pending
        gone = self._insert(db, minutes_ago=9000)        # will be abandoned
        for _ in range(db.MAX_OUTCOME_ATTEMPTS):
            db.record_outcome_attempt([gone])

        h = db.health_summary()
        assert h['total_predictions'] == 3
        assert h['resolved_outcomes'] == 1
        assert h['abandoned_outcomes'] == 1
        assert h['pending_outcomes'] == 1
        assert pending is not None

    def test_naive_stored_timestamp_is_read_as_utc(self, db):
        """Timestamps are stored without an offset; reading them as local time
        would make a fresh prediction look hours stale."""
        self._insert(db, minutes_ago=5)
        last = db.get_last_prediction_time()
        assert last.tzinfo is not None
        assert db.minutes_since_last_prediction() == pytest.approx(5, abs=2)
