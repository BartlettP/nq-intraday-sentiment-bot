"""
Liveness and integrity check for the predictions database.

Answers the question the project previously had no way to answer: is the bot
still running and keeping up? Exits non-zero when it isn't, so it can be wired
into cron, a systemd timer, or a monitoring check.

    python scripts/db_health.py
    python scripts/db_health.py --stale-after 120
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config  # noqa: E402
import database  # noqa: E402


def describe_tables():
    with database.get_connection() as conn:
        tables = [r['name'] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        counts = {}
        for t in tables:
            # Table names come from sqlite_master, not user input.
            counts[t] = conn.execute(f'SELECT COUNT(*) AS n FROM "{t}"').fetchone()['n']
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stale-after', type=int, default=90, metavar='MINUTES',
                        help='age at which the newest prediction counts as stale '
                             '(default: 90, i.e. 1.5 hourly cycles)')
    args = parser.parse_args()

    print(f"Database: {config.DB_PATH}")
    if not os.path.exists(config.DB_PATH):
        print("  ERROR: file does not exist")
        return 2
    print(f"  size: {os.path.getsize(config.DB_PATH):,} bytes")

    try:
        counts = describe_tables()
    except sqlite3.DatabaseError as e:
        print(f"  ERROR: not a readable SQLite database ({e})")
        return 2

    print("\nTables:")
    for name, n in counts.items():
        print(f"  {name:20} {n:>7,} rows")

    if 'predictions' not in counts:
        print("\nERROR: no predictions table")
        return 2

    h = database.health_summary(stale_after_minutes=args.stale_after)

    print("\nPredictions:")
    print(f"  total              {h['total_predictions']:>7,}")
    print(f"  with outcomes      {h['resolved_outcomes']:>7,}")
    print(f"  awaiting outcome   {h['pending_outcomes']:>7,}")
    print(f"  abandoned          {h['abandoned_outcomes']:>7,}"
          f"   (gave up after {database.MAX_OUTCOME_ATTEMPTS} attempts)")

    print("\nLiveness:")
    if h['last_prediction_utc'] is None:
        print("  no predictions recorded")
    else:
        print(f"  last prediction    {h['last_prediction_utc']:%Y-%m-%d %H:%M} UTC")
        print(f"  age                {h['minutes_since_last']:.0f} min")

    if h['is_stale']:
        print(f"\nSTALE: no prediction in the last {args.stale_after} min. "
              f"The bot may not be running.")
        return 1

    print("\nOK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
