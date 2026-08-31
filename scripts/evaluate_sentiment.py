"""
Does the sentiment half of the bot earn its keep?

The backtest measures the volatility model only. This measures the other input:
whether sentiment_score has any relationship to what the market subsequently
did, and whether the signal taxonomy built on top of it is even reachable.

    python scripts/evaluate_sentiment.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config  # noqa: E402
import database  # noqa: E402

# A forward return is only meaningful between consecutive in-session
# predictions. Anything outside this gap spans a close, a weekend, or a gap in
# the record.
MIN_GAP_SECONDS = 30 * 60
MAX_GAP_SECONDS = 2 * 60 * 60


def load():
    with database.get_connection() as conn:
        return pd.read_sql_query(
            'SELECT * FROM predictions ORDER BY timestamp ASC',
            conn, parse_dates=['timestamp']
        )


def section(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def report_reachability(df):
    section("SIGNAL REACHABILITY")
    s = df['sentiment_score'].dropna()
    if s.empty:
        print("  no sentiment scores recorded")
        return

    strong, neutral = config.SENTIMENT_STRONG, config.SENTIMENT_NEUTRAL
    print(f"  observed range     : {s.min():+.2f} .. {s.max():+.2f}")
    print(f"  mean / std         : {s.mean():+.2f} / {s.std():.2f}")
    print()
    for label, mask in [
        (f"bullish gate (s > {strong})", s > strong),
        (f"bearish gate (s < -{strong})", s < -strong),
        (f"neutral gate (|s| < {neutral})", s.abs() < neutral),
        ("score is negative", s < 0),
    ]:
        n = int(mask.sum())
        flag = "   <-- NEVER FIRES" if n == 0 else ""
        print(f"  {label:32} {n:4d} / {len(s)}  ({n / len(s) * 100:5.1f}%){flag}")

    print("\n  Signals actually emitted:")
    for sig, n in df['signal'].value_counts().items():
        ascii_sig = ''.join(c for c in str(sig) if ord(c) < 128).strip()
        print(f"    {n:5d}  {ascii_sig}")


def report_predictive_power(df):
    section("PREDICTIVE POWER")
    d = df.dropna(subset=['actual_volatility', 'sentiment_score'])
    if len(d) < 10:
        print("  not enough resolved outcomes yet")
        return

    print(f"  n = {len(d)} predictions with recorded outcomes\n")
    pairs = [
        ("sentiment vs actual volatility", d['sentiment_score'], d['actual_volatility']),
        ("|sentiment| vs actual volatility", d['sentiment_score'].abs(), d['actual_volatility']),
        ("sentiment vs volatility change",
         d['sentiment_score'], d['actual_volatility'] - d['current_volatility']),
    ]
    for label, x, y in pairs:
        valid = x.notna() & y.notna()
        r = x[valid].corr(y[valid])
        print(f"  corr({label:36}) = {r:+.3f}")

    # Forward price return, from consecutive in-session predictions.
    f = df.dropna(subset=['nq_price', 'sentiment_score']).copy()
    f['fwd_return'] = f['nq_price'].shift(-1) / f['nq_price'] - 1
    gap = (f['timestamp'].shift(-1) - f['timestamp']).dt.total_seconds()
    f = f[gap.between(MIN_GAP_SECONDS, MAX_GAP_SECONDS)].dropna(subset=['fwd_return'])

    print()
    if len(f) < 20:
        print("  not enough consecutive in-session pairs for a direction test")
        return

    r = f['sentiment_score'].corr(f['fwd_return'])
    print(f"  corr(sentiment vs next-hour NQ return)     = {r:+.3f}   (n={len(f)})")

    median = f['sentiment_score'].median()
    hi = f[f['sentiment_score'] > median]
    lo = f[f['sentiment_score'] <= median]
    base_rate = (f['fwd_return'] > 0).mean() * 100
    hit_rate = (hi['fwd_return'] > 0).mean() * 100

    print(f"  mean next-hour return, above-median sentiment = {hi['fwd_return'].mean() * 100:+.4f}%")
    print(f"  mean next-hour return, below-median sentiment = {lo['fwd_return'].mean() * 100:+.4f}%")
    print(f"  up-move rate when sentiment above median      = {hit_rate:.1f}%")
    print(f"  up-move base rate (all predictions)           = {base_rate:.1f}%")
    print(f"  edge over base rate                           = {hit_rate - base_rate:+.1f} pts")


def report_verdict(df):
    section("VERDICT")
    s = df['sentiment_score'].dropna()
    d = df.dropna(subset=['actual_volatility', 'sentiment_score'])

    problems = []
    if not s.empty and (s < -config.SENTIMENT_STRONG).sum() == 0:
        problems.append(
            f"The bearish gate (s < -{config.SENTIMENT_STRONG}) has never fired in "
            f"{len(s)} predictions, so both BEARISH signals are unreachable."
        )
    if not s.empty and (s < 0).sum() == 0:
        problems.append("The score has never been negative — it is a positive offset, not a signal.")
    if len(d) >= 30:
        r = abs(d['sentiment_score'].corr(d['actual_volatility']))
        if r < 0.15:
            problems.append(
                f"Correlation with realized volatility is {r:.3f} — indistinguishable from noise."
            )

    if not problems:
        print("  No structural problems detected.")
        return

    for p in problems:
        print(f"  - {p}")

    print("""
  Sentiment is not currently contributing information.

  Root cause: the classifier is binary (Bearish/Bullish) with no Neutral
  class, so every mundane headline — most of what an RSS feed carries — is
  forced into one side. That noise averages out to a roughly constant
  positive offset rather than a signal.

  Options, cheapest first:

  1. Score relative to a rolling baseline instead of an absolute zero.
     Compare each score against, say, the trailing 20-prediction mean, and
     gate on the deviation. Makes the taxonomy reachable immediately, no
     retraining.
  2. Use predict_proba and ignore low-confidence headlines, so only
     headlines the model is actually sure about get a vote.
  3. Retrain with a Neutral class. Most correct, most work — needs
     relabelled training data.

  Re-run this script after any change to see whether it moved.""")


def main():
    df = load()
    print(f"Database : {config.DB_PATH}")
    print(f"Rows     : {len(df)}")
    if df.empty:
        print("No predictions recorded.")
        return 1

    print(f"Range    : {df['timestamp'].min()} .. {df['timestamp'].max()}")
    report_reachability(df)
    report_predictive_power(df)
    report_verdict(df)
    return 0


if __name__ == '__main__':
    sys.exit(main())
