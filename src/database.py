"""
SQLite persistence for the NQ bot.
Records every prediction and (later) the actual outcome.
"""
import sqlite3
import os
from collections import defaultdict
from datetime import datetime, timezone
from contextlib import contextmanager

import pytz

import config

ET = pytz.timezone('US/Eastern')

# Minimum same-hour observations before that hour's bias correction is trusted.
# Below this the correction is treated as zero rather than fitted to noise.
MIN_BIAS_SAMPLES = 8

# Resolved centrally (config.DB_PATH, overridable via NQ_DB_PATH) so every
# script in the project reads and writes the same file.
DB_PATH = config.DB_PATH
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

# A prediction whose outcome window has passed unrecorded this many times is
# given up on. Without this, rows that can never be filled (bot was down over
# the window, yfinance gap, holiday) sit at the head of the ASC-ordered backfill
# queue forever and eventually starve out newer predictions.
MAX_OUTCOME_ATTEMPTS = 6


def _utc_now_iso():
    """Timezone-aware UTC timestamp. Stored without offset for continuity with
    the existing rows, which were written by the naive datetime.utcnow()."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@contextmanager
def get_connection():
    """Context manager so connections always get closed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the predictions table if it doesn't exist, and apply migrations."""
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sentiment_score REAL,
                sentiment_bullish INTEGER,
                sentiment_bearish INTEGER,
                sentiment_total INTEGER,
                predicted_volatility REAL,
                current_volatility REAL,
                last_4h_avg_volatility REAL,
                nq_price REAL,
                signal TEXT,
                actual_volatility REAL DEFAULT NULL,
                outcome_recorded_at TEXT DEFAULT NULL,
                outcome_attempts INTEGER NOT NULL DEFAULT 0,
                predicted_volatility_raw REAL DEFAULT NULL,
                bias_correction REAL DEFAULT NULL
            )
        ''')

        # Migrations for databases created before these columns existed.
        existing = {row['name'] for row in conn.execute('PRAGMA table_info(predictions)')}
        if 'outcome_attempts' not in existing:
            conn.execute(
                'ALTER TABLE predictions '
                'ADD COLUMN outcome_attempts INTEGER NOT NULL DEFAULT 0'
            )
        # predicted_volatility always holds what the bot published, so the
        # backtest keeps measuring the deployed system. The uncorrected model
        # output is kept alongside it; NULL on old rows means "no correction
        # was applied, raw == published".
        if 'predicted_volatility_raw' not in existing:
            conn.execute(
                'ALTER TABLE predictions ADD COLUMN predicted_volatility_raw REAL DEFAULT NULL'
            )
        if 'bias_correction' not in existing:
            conn.execute(
                'ALTER TABLE predictions ADD COLUMN bias_correction REAL DEFAULT NULL'
            )

        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)'
        )
        # The backfill query filters on these two columns on every cycle.
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_pending_outcomes '
            'ON predictions(actual_volatility, outcome_attempts, timestamp)'
        )
        conn.commit()


def insert_prediction(sentiment, volatility, signal):
    """
    Insert a new prediction row.
    Both sentiment and volatility may be None — handled gracefully.
    Returns the new row's ID.
    """
    with get_connection() as conn:
        cursor = conn.execute('''
            INSERT INTO predictions (
                timestamp, sentiment_score, sentiment_bullish, sentiment_bearish,
                sentiment_total, predicted_volatility, current_volatility,
                last_4h_avg_volatility, nq_price, signal,
                predicted_volatility_raw, bias_correction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            _utc_now_iso(),
            sentiment.get('score') if sentiment else None,
            sentiment.get('bullish') if sentiment else None,
            sentiment.get('bearish') if sentiment else None,
            sentiment.get('total') if sentiment else None,
            volatility.get('predicted_next_hour') if volatility else None,
            volatility.get('current') if volatility else None,
            volatility.get('last_4h_avg') if volatility else None,
            volatility.get('nq_price') if volatility else None,
            signal,
            volatility.get('predicted_next_hour_raw') if volatility else None,
            volatility.get('bias_correction') if volatility else None,
        ))
        conn.commit()
        return cursor.lastrowid


def get_predictions_needing_outcomes(min_hours_old=1, max_results=50,
                                     max_attempts=MAX_OUTCOME_ATTEMPTS):
    """
    Find predictions made at least min_hours_old hours ago that don't yet
    have an actual_volatility recorded and haven't been given up on.
    """
    with get_connection() as conn:
        cursor = conn.execute('''
            SELECT id, timestamp, predicted_volatility, outcome_attempts
            FROM predictions
            WHERE actual_volatility IS NULL
              AND outcome_attempts < ?
              AND datetime(timestamp) <= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (max_attempts, min_hours_old, max_results))
        return [dict(row) for row in cursor.fetchall()]


def update_outcome(prediction_id, actual_volatility):
    """Fill in the actual realized volatility for a past prediction."""
    with get_connection() as conn:
        conn.execute('''
            UPDATE predictions
            SET actual_volatility = ?, outcome_recorded_at = ?
            WHERE id = ?
        ''', (actual_volatility, _utc_now_iso(), prediction_id))
        conn.commit()


def record_outcome_attempt(prediction_ids):
    """
    Mark that we tried and failed to resolve these predictions' outcomes.
    Once a row reaches MAX_OUTCOME_ATTEMPTS it drops out of the backfill queue
    permanently instead of being re-fetched on every cycle forever.
    """
    ids = list(prediction_ids)
    if not ids:
        return
    with get_connection() as conn:
        conn.executemany(
            'UPDATE predictions SET outcome_attempts = outcome_attempts + 1 WHERE id = ?',
            [(pid,) for pid in ids]
        )
        conn.commit()


def count_abandoned_outcomes(max_attempts=MAX_OUTCOME_ATTEMPTS):
    """How many predictions were given up on. Useful for monitoring."""
    with get_connection() as conn:
        row = conn.execute(
            'SELECT COUNT(*) AS n FROM predictions '
            'WHERE actual_volatility IS NULL AND outcome_attempts >= ?',
            (max_attempts,)
        ).fetchone()
        return row['n']


def _et_hour(timestamp_str):
    """ET hour-of-day for a stored (naive UTC) timestamp.

    Done in Python rather than SQL because the UTC->ET offset is DST-dependent
    and SQLite has no timezone database.
    """
    parsed = datetime.fromisoformat(timestamp_str)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(ET).hour


def get_hourly_bias_corrections(min_samples=MIN_BIAS_SAMPLES, lookback_days=None):
    """
    Per-hour (ET) correction to add to the model's predicted volatility delta.

    The v2 model predicts a decrease at nearly every hour — its predicted
    deltas have ~38% of the standard deviation of actual ones and are negative
    ~71% of the time. That's fine in the afternoon, when volatility genuinely
    decays, but badly wrong across the 9:30 open, where realized volatility
    rises on ~97% of days. Training used rth_only=True, so the model never saw
    the pre-open -> open ramp and cannot represent it.

    This measures that miss per hour and hands back an additive correction:

        correction[hour] = mean(actual_delta - RAW_predicted_delta)

    Returns {et_hour: correction}, omitting hours with too few samples.

    The residual is always measured against the *raw* model output, never
    against an already-corrected prediction. Learning from corrected values
    would fold each correction back into the next one and compound without
    limit.
    """
    query = '''
        SELECT timestamp, predicted_volatility, predicted_volatility_raw,
               current_volatility, actual_volatility
        FROM predictions
        WHERE actual_volatility IS NOT NULL
          AND current_volatility IS NOT NULL
          AND predicted_volatility IS NOT NULL
    '''
    params = []
    if lookback_days is not None:
        query += " AND datetime(timestamp) >= datetime('now', '-' || ? || ' days')"
        params.append(lookback_days)

    residuals = defaultdict(list)
    with get_connection() as conn:
        for row in conn.execute(query, params):
            # Rows written before the correction existed have no raw column;
            # for those the published value *is* the raw model output.
            raw = row['predicted_volatility_raw']
            if raw is None:
                raw = row['predicted_volatility']

            raw_delta = raw - row['current_volatility']
            actual_delta = row['actual_volatility'] - row['current_volatility']
            residuals[_et_hour(row['timestamp'])].append(actual_delta - raw_delta)

    return {
        hour: sum(values) / len(values)
        for hour, values in residuals.items()
        if len(values) >= min_samples
    }


def get_bias_correction_for_hour(et_hour, min_samples=MIN_BIAS_SAMPLES,
                                 lookback_days=None):
    """Correction for one ET hour, or 0.0 when there isn't enough history."""
    return get_hourly_bias_corrections(min_samples, lookback_days).get(et_hour, 0.0)


def get_last_prediction_time():
    """UTC datetime of the most recent prediction, or None if the table is empty.

    This is the bot's liveness signal: if it stops advancing, the process died
    and nobody would otherwise notice until someone spotted that Discord had
    gone quiet.
    """
    with get_connection() as conn:
        row = conn.execute(
            'SELECT timestamp FROM predictions ORDER BY timestamp DESC LIMIT 1'
        ).fetchone()

    if row is None or not row['timestamp']:
        return None

    parsed = datetime.fromisoformat(row['timestamp'])
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def minutes_since_last_prediction():
    """Age of the newest prediction in minutes, or None if there are none."""
    last = get_last_prediction_time()
    if last is None:
        return None
    return (datetime.now(timezone.utc) - last).total_seconds() / 60


def health_summary(stale_after_minutes=90):
    """A small dict describing whether the bot looks alive and keeping up.

    stale_after_minutes defaults to 90: predictions land hourly, so anything
    past ~1.5 cycles means a run was missed.
    """
    age = minutes_since_last_prediction()
    with get_connection() as conn:
        total = conn.execute('SELECT COUNT(*) AS n FROM predictions').fetchone()['n']
        resolved = conn.execute(
            'SELECT COUNT(*) AS n FROM predictions WHERE actual_volatility IS NOT NULL'
        ).fetchone()['n']

    return {
        'total_predictions': total,
        'resolved_outcomes': resolved,
        'pending_outcomes': len(get_predictions_needing_outcomes(max_results=1000)),
        'abandoned_outcomes': count_abandoned_outcomes(),
        'last_prediction_utc': get_last_prediction_time(),
        'minutes_since_last': age,
        # None (empty table) counts as stale — an empty database is not healthy.
        'is_stale': age is None or age > stale_after_minutes,
    }
