"""
SQLite persistence for the NQ bot.
Records every prediction and (later) the actual outcome.
"""
import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

# Database lives in a data/ folder next to the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'predictions.db')


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
    """Create the predictions table if it doesn't exist."""
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
                outcome_recorded_at TEXT DEFAULT NULL
            )
        ''')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)'
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
                last_4h_avg_volatility, nq_price, signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            sentiment.get('score') if sentiment else None,
            sentiment.get('bullish') if sentiment else None,
            sentiment.get('bearish') if sentiment else None,
            sentiment.get('total') if sentiment else None,
            volatility.get('predicted_next_hour') if volatility else None,
            volatility.get('current') if volatility else None,
            volatility.get('last_4h_avg') if volatility else None,
            volatility.get('nq_price') if volatility else None,
            signal
        ))
        conn.commit()
        return cursor.lastrowid

def get_predictions_needing_outcomes(min_hours_old=1, max_results=50):
        """
        Find predictions made at least min_hours_old hours ago that don't yet
        have an actual_volatility recorded.
        """
        with get_connection() as conn:
            cursor = conn.execute('''
                SELECT id, timestamp, predicted_volatility 
                FROM predictions 
                WHERE actual_volatility IS NULL 
                  AND datetime(timestamp) <= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp ASC
                LIMIT ?
            ''', (min_hours_old, max_results))
            return [dict(row) for row in cursor.fetchall()]

def update_outcome(prediction_id, actual_volatility):
        """Fill in the actual realized volatility for a past prediction."""
        with get_connection() as conn:
            conn.execute('''
                UPDATE predictions 
                SET actual_volatility = ?, outcome_recorded_at = ?
                WHERE id = ?
            ''', (actual_volatility, datetime.utcnow().isoformat(), prediction_id))
            conn.commit()