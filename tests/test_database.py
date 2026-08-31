"""
Persistence layer tests, focused on the outcome-backfill queue — the part that
silently accumulated unfillable rows before the outcome_attempts change.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _insert_at(db, when, actual=None, attempts=0):
    """Insert a prediction with an explicit timestamp (bypasses insert_prediction,
    which always stamps 'now')."""
    with db.get_connection() as conn:
        cur = conn.execute(
            'INSERT INTO predictions (timestamp, predicted_volatility, '
            'current_volatility, actual_volatility, outcome_attempts) '
            'VALUES (?, ?, ?, ?, ?)',
            (when.replace(tzinfo=None).isoformat(), 0.09, 0.08, actual, attempts)
        )
        conn.commit()
        return cur.lastrowid


def _hours_ago(n):
    return datetime.now(timezone.utc) - timedelta(hours=n)


class TestSchema:
    def test_init_db_is_idempotent(self, db):
        db.init_db()
        db.init_db()  # must not raise on re-run

    def test_outcome_attempts_column_exists(self, db):
        with db.get_connection() as conn:
            cols = {r['name'] for r in conn.execute('PRAGMA table_info(predictions)')}
        assert 'outcome_attempts' in cols

    def test_migrates_a_pre_existing_database(self, db, tmp_path, monkeypatch):
        """A database created before outcome_attempts existed must gain the
        column without losing rows — this is what runs against production."""
        legacy_path = str(tmp_path / 'legacy.db')
        conn = sqlite3.connect(legacy_path)
        conn.execute(
            'CREATE TABLE predictions ('
            '  id INTEGER PRIMARY KEY AUTOINCREMENT,'
            '  timestamp TEXT NOT NULL,'
            '  sentiment_score REAL,'
            '  sentiment_bullish INTEGER,'
            '  sentiment_bearish INTEGER,'
            '  sentiment_total INTEGER,'
            '  predicted_volatility REAL,'
            '  current_volatility REAL,'
            '  last_4h_avg_volatility REAL,'
            '  nq_price REAL,'
            '  signal TEXT,'
            '  actual_volatility REAL DEFAULT NULL,'
            '  outcome_recorded_at TEXT DEFAULT NULL'
            ')'
        )
        conn.execute(
            'INSERT INTO predictions (timestamp, predicted_volatility) VALUES (?, ?)',
            ('2026-05-11T12:00:00', 0.05)
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(db, 'DB_PATH', legacy_path)
        db.init_db()

        with db.get_connection() as c:
            cols = {r['name'] for r in c.execute('PRAGMA table_info(predictions)')}
            rows = c.execute('SELECT * FROM predictions').fetchall()

        assert 'outcome_attempts' in cols
        assert len(rows) == 1, "migration must not drop existing predictions"
        assert rows[0]['outcome_attempts'] == 0
        assert rows[0]['predicted_volatility'] == 0.05


class TestInsertPrediction:
    def test_returns_row_id_and_persists(self, db, sentiment, volatility):
        pred_id = db.insert_prediction(sentiment(25.0), volatility(0.12), "STRONG BULLISH")
        assert pred_id > 0

        with db.get_connection() as conn:
            row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,)).fetchone()

        assert row['sentiment_score'] == 25.0
        assert row['predicted_volatility'] == 0.12
        assert row['signal'] == "STRONG BULLISH"
        assert row['actual_volatility'] is None
        assert row['outcome_attempts'] == 0

    def test_handles_none_sentiment_and_volatility(self, db):
        """The docstring promises this; nothing exercised it before."""
        pred_id = db.insert_prediction(None, None, "Insufficient data")
        with db.get_connection() as conn:
            row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,)).fetchone()
        assert row['sentiment_score'] is None
        assert row['predicted_volatility'] is None
        assert row['signal'] == "Insufficient data"

    def test_timestamp_is_utc_and_naive(self, db, sentiment, volatility):
        """Stored without an offset, matching the existing production rows —
        update_past_outcomes() does tz_localize('UTC') and would raise on an
        already-aware timestamp."""
        pred_id = db.insert_prediction(sentiment(0.0), volatility(0.08), "neutral")
        with db.get_connection() as conn:
            ts = conn.execute(
                'SELECT timestamp FROM predictions WHERE id = ?', (pred_id,)
            ).fetchone()['timestamp']

        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is None
        drift = abs((datetime.now(timezone.utc).replace(tzinfo=None) - parsed).total_seconds())
        assert drift < 60, f"timestamp is not UTC-now (drift {drift}s)"


class TestBackfillQueue:
    def test_returns_only_predictions_old_enough(self, db):
        recent = _insert_at(db, _hours_ago(0.5))
        ripe = _insert_at(db, _hours_ago(3))

        ids = [p['id'] for p in db.get_predictions_needing_outcomes(min_hours_old=1)]
        assert ripe in ids
        assert recent not in ids

    def test_excludes_predictions_that_already_have_outcomes(self, db):
        done = _insert_at(db, _hours_ago(3), actual=0.07)
        pending = _insert_at(db, _hours_ago(3))

        ids = [p['id'] for p in db.get_predictions_needing_outcomes()]
        assert pending in ids
        assert done not in ids

    def test_gives_up_after_max_attempts(self, db):
        """The core of the leak fix: a row that can never be resolved must
        eventually leave the queue."""
        exhausted = _insert_at(db, _hours_ago(48), attempts=db.MAX_OUTCOME_ATTEMPTS)
        still_trying = _insert_at(db, _hours_ago(48), attempts=db.MAX_OUTCOME_ATTEMPTS - 1)

        ids = [p['id'] for p in db.get_predictions_needing_outcomes()]
        assert still_trying in ids
        assert exhausted not in ids

    def test_orphans_do_not_starve_newer_predictions(self, db):
        """Regression test for the actual production bug.

        The queue is ORDER BY timestamp ASC LIMIT n. Unfillable old rows used
        to hold the front of it forever, so once enough accumulated, newer
        predictions would never be looked at.
        """
        limit = 5
        # More permanently-stuck old rows than the queue can hold...
        for _ in range(limit + 3):
            _insert_at(db, _hours_ago(200), attempts=db.MAX_OUTCOME_ATTEMPTS)
        # ...plus one fresh prediction that genuinely needs an outcome.
        newest = _insert_at(db, _hours_ago(2))

        ids = [p['id'] for p in db.get_predictions_needing_outcomes(max_results=limit)]
        assert newest in ids, "old abandoned rows are starving the queue again"

    def test_respects_max_results(self, db):
        for _ in range(10):
            _insert_at(db, _hours_ago(5))
        assert len(db.get_predictions_needing_outcomes(max_results=4)) == 4

    def test_ordered_oldest_first(self, db):
        newer = _insert_at(db, _hours_ago(2))
        older = _insert_at(db, _hours_ago(10))
        ids = [p['id'] for p in db.get_predictions_needing_outcomes()]
        assert ids.index(older) < ids.index(newer)


class TestRecordOutcomeAttempt:
    def test_increments_attempts(self, db):
        pid = _insert_at(db, _hours_ago(3))
        db.record_outcome_attempt([pid])
        db.record_outcome_attempt([pid])

        with db.get_connection() as conn:
            n = conn.execute(
                'SELECT outcome_attempts FROM predictions WHERE id = ?', (pid,)
            ).fetchone()['outcome_attempts']
        assert n == 2

    def test_empty_list_is_a_noop(self, db):
        db.record_outcome_attempt([])  # must not raise

    def test_repeated_attempts_eventually_drop_the_row(self, db):
        pid = _insert_at(db, _hours_ago(72))
        for _ in range(db.MAX_OUTCOME_ATTEMPTS):
            assert pid in [p['id'] for p in db.get_predictions_needing_outcomes()]
            db.record_outcome_attempt([pid])

        assert pid not in [p['id'] for p in db.get_predictions_needing_outcomes()]
        assert db.count_abandoned_outcomes() == 1


class TestUpdateOutcome:
    def test_fills_actual_and_stamps_recorded_at(self, db):
        pid = _insert_at(db, _hours_ago(3))
        db.update_outcome(pid, 0.0731)

        with db.get_connection() as conn:
            row = conn.execute('SELECT * FROM predictions WHERE id = ?', (pid,)).fetchone()

        assert row['actual_volatility'] == pytest.approx(0.0731)
        assert row['outcome_recorded_at'] is not None

    def test_resolved_prediction_leaves_the_queue(self, db):
        pid = _insert_at(db, _hours_ago(3))
        db.update_outcome(pid, 0.07)
        assert pid not in [p['id'] for p in db.get_predictions_needing_outcomes()]
