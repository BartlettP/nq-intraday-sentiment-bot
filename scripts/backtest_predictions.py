"""
Backtest the bot's predictions against actual realized volatility.
Reads from predictions.db (the production database).
"""
import sqlite3
import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config  # noqa: E402
import database  # noqa: E402

DB_PATH = config.DB_PATH


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
        cutoff = (datetime.now(timezone.utc).replace(tzinfo=None)
                  - timedelta(days=days)).isoformat()
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


def evaluate_bias_correction(min_samples=8):
    """
    Walk-forward test of the per-hour bias correction.

    Each prediction is corrected using only *earlier* same-hour observations,
    so there is no lookahead. This is the check that stops the correction from
    quietly becoming a fitted-to-noise liability as the data grows — re-run it
    periodically, and turn the correction off (NQ_BIAS_CORRECTION=0) if the
    corrected row stops winning.
    """
    import sqlite3
    from collections import defaultdict

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute('''
        SELECT timestamp, predicted_volatility, predicted_volatility_raw,
               current_volatility, actual_volatility
        FROM predictions
        WHERE actual_volatility IS NOT NULL
          AND current_volatility IS NOT NULL
          AND predicted_volatility IS NOT NULL
        ORDER BY timestamp ASC
    '''))
    conn.close()

    print(f"\n{'=' * 60}")
    print("🔧 PER-HOUR BIAS CORRECTION (walk-forward, no lookahead)")
    print(f"{'=' * 60}")

    if not rows:
        print("  No data")
        return

    history = defaultdict(list)
    raw_err, adj_err, base_err = [], [], []
    raw_dir, adj_dir = [], []

    for r in rows:
        raw = r['predicted_volatility_raw']
        if raw is None:
            raw = r['predicted_volatility']

        raw_delta = raw - r['current_volatility']
        actual_delta = r['actual_volatility'] - r['current_volatility']
        hour = database._et_hour(r['timestamp'])

        prior = history[hour]
        if len(prior) >= min_samples:
            correction = sum(prior) / len(prior)
            adj_delta = raw_delta + correction

            raw_err.append(abs(raw_delta - actual_delta))
            adj_err.append(abs(adj_delta - actual_delta))
            base_err.append(abs(actual_delta))
            raw_dir.append(np.sign(raw_delta) == np.sign(actual_delta))
            adj_dir.append(np.sign(adj_delta) == np.sign(actual_delta))

        history[hour].append(actual_delta - raw_delta)

    if not adj_err:
        print(f"  Not enough history yet (need {min_samples} per hour)")
        return

    base_mae = np.mean(base_err)
    raw_mae, adj_mae = np.mean(raw_err), np.mean(adj_err)
    print(f"  Evaluated on {len(adj_err)} out-of-sample predictions\n")
    print(f"  {'':22} {'MAE':>9} {'vs baseline':>12} {'directional':>12}")
    print(f"  {'persistence baseline':22} {base_mae:>9.4f} {'—':>12} {'—':>12}")
    print(f"  {'raw model':22} {raw_mae:>9.4f} "
          f"{(base_mae - raw_mae) / base_mae * 100:>+11.1f}% {np.mean(raw_dir) * 100:>11.1f}%")
    print(f"  {'+ bias correction':22} {adj_mae:>9.4f} "
          f"{(base_mae - adj_mae) / base_mae * 100:>+11.1f}% {np.mean(adj_dir) * 100:>11.1f}%")

    verdict = "helping" if adj_mae < raw_mae else "NOT helping — consider NQ_BIAS_CORRECTION=0"
    print(f"\n  Verdict: correction is {verdict}")

    live = database.get_hourly_bias_corrections()
    if live:
        print("\n  Corrections currently in force (ET hour):")
        for hour in sorted(live):
            print(f"    {hour:2d}:00  {live[hour]:+.5f}")


if __name__ == '__main__':
    print("\n📊 NQ Bot Backtest")
    print(f"   Database: {DB_PATH}")
    print(f"   Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows = load_predictions_with_outcomes()
    compute_metrics(all_rows, label="📈 ALL-TIME")

    V2_DEPLOY_TIME = '2026-05-13 20:00:00'
    v2_rows = load_predictions_with_outcomes(after_timestamp=V2_DEPLOY_TIME)
    compute_metrics(v2_rows, label=f"🚀 V2 ONLY (since {V2_DEPLOY_TIME})")

    evaluate_bias_correction()