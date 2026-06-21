import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to SQLite
sqlite_conn = sqlite3.connect('predictions.db')
sqlite_conn.row_factory = sqlite3.Row

# Connect to PostgreSQL
pg_conn = psycopg2.connect(
    host='127.0.0.1',
    port=5433,  # Changed from 5432
    database='nq_bot',
    user='postgres',
    password='mypassword'
)
pg_cursor = pg_conn.cursor()

# Check SQLite schema first
print("\n=== SQLite schema ===")
cursor = sqlite_conn.execute("PRAGMA table_info(predictions)")
for row in cursor.fetchall():
    print(f"  {row[1]} ({row[2]})")

# Get a sample row to understand structure
print("\n=== Sample SQLite row ===")
sample = sqlite_conn.execute('SELECT * FROM predictions LIMIT 1').fetchone()
if sample:
    print(dict(sample))

# Migrate predictions
print("\n=== Migrating data ===")
sqlite_cursor = sqlite_conn.execute('SELECT * FROM predictions')
count = 0
errors = 0

for row in sqlite_cursor:
    row_dict = dict(row)

    try:
        pg_cursor.execute(
            '''INSERT INTO predictions 
               (id, timestamp, sentiment_score, predicted_volatility, 
                current_volatility, actual_volatility, nq_price, signal,
                delta_predicted, outcome_timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO NOTHING''',
            (
                row_dict.get('id'),
                row_dict.get('timestamp'),
                row_dict.get('sentiment_score'),
                row_dict.get('predicted_volatility'),
                row_dict.get('current_volatility'),
                row_dict.get('actual_volatility'),
                row_dict.get('nq_price'),
                row_dict.get('signal'),
                row_dict.get('delta_predicted'),
                row_dict.get('outcome_timestamp'),
            )
        )
        count += 1
    except Exception as e:
        errors += 1
        print(f"Error on row {row_dict.get('id')}: {e}")

pg_conn.commit()
print(f"\n✅ Migrated {count} predictions, {errors} errors")

sqlite_conn.close()
pg_conn.close()