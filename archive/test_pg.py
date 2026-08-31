import psycopg2

# Try connecting with explicit IPv4
try:
    conn = psycopg2.connect(
        host='127.0.0.1',
        port=5433,  # Changed from 5432
        database='nq_bot',
        user='postgres',
        password='mypassword'
    )
    print("✅ Connection successful with 127.0.0.1")
    conn.close()
except Exception as e:
    print(f"❌ Failed with 127.0.0.1: {e}")