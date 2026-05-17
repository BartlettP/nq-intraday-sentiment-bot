"""
Backtest the bot's predictions against actual realized volatility.
Reads from predictions.db (the production database).
"""
import sqlite3
import os
from datetime import datetime, timedelta
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'predictions.db')


def load_predictions_with_outcomes(days=None, after_timestamp=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = '''
        SELECT id, timestamp, predicted_volatility, current_volatility,
               actual_volatility, sentiment_score
        FROM predictions
        WHERE actual_volatility IS NOT NULL
    '''
    params = []

    if days is not None:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query += ' AND datetime(timestamp) >= datetime(?)'
        params.append(cutoff)

    if after_timestamp is not None:
        query += ' AND datetime(timestamp) >= datetime(?)'
        params.append(after_timestamp)

    query += ' ORDER BY id ASC'

    rows = list(conn.execute(query, params))
    conn.close()
    return rows


def compute_metrics(rows, label=""):
    if not rows:
        print(f"\n{label}: No data")
        return

    predicted = np.array([r['predicted_volatility'] for r in rows])
    actual = np.array([r['actual_volatility'] for r in rows])
    current = np.array([
        r['current_volatility'] if r['current_volatility'] is not None else np.nan
        for r in rows
    ])

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else float('nan')

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Predictions evaluated: {len(rows)}")
    print(f"  Mean predicted vol: {predicted.mean():.4f}%")
    print(f"  Mean actual vol:    {actual.mean():.4f}%")
    print(f"  MAE:                {mae:.4f}%")
    print(f"  RMSE:               {rmse:.4f}%")
    print(f"  Correlation:        {correlation:.4f}")

    valid_current = ~np.isnan(current)
    if valid_current.sum() == len(actual):
        baseline_mae = np.mean(np.abs(actual - current))
        beats_baseline_pct = (baseline_mae - mae) / baseline_mae * 100

        print(f"\n  Persistence baseline MAE: {baseline_mae:.4f}%")
        print(f"  Model MAE:                {mae:.4f}%")
        if mae < baseline_mae:
            print(f"  ✅ Model beats baseline by: {beats_baseline_pct:.1f}%")
        else:
            print(f"  ❌ Baseline beats model by: {-beats_baseline_pct:.1f}%")

        actual_delta = actual - current
        pred_delta = predicted - current
        correct_dir = np.sum(np.sign(actual_delta) == np.sign(pred_delta))
        directional = correct_dir / len(actual_delta) * 100
        print(f"\n  Directional accuracy: {directional:.1f}% (random: 50%)")


if __name__ == '__main__':
    print("\n📊 NQ Bot Backtest")
    print(f"   Database: {DB_PATH}")
    print(f"   Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows = load_predictions_with_outcomes()
    compute_metrics(all_rows, label="📈 ALL-TIME")

    V2_DEPLOY_TIME = '2026-05-13 20:00:00'
    v2_rows = load_predictions_with_outcomes(after_timestamp=V2_DEPLOY_TIME)
    compute_metrics(v2_rows, label=f"🚀 V2 ONLY (since {V2_DEPLOY_TIME})")