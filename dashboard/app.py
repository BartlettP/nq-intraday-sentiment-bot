"""
NQ Volatility Bot - Streamlit Dashboard

Reads the predictions database and displays current signal,
recent history, and prediction vs. actual chart.
"""
import streamlit as st
import sqlite3
import pandas as pd
import os
import boto3
import tempfile
from datetime import datetime
from botocore.exceptions import ClientError

# S3 configuration (read from environment variables on Render)
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_REGION = os.environ.get('AWS_REGION', 'us-east-2')

# Local fallback path (used when running locally and S3 not configured)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DB_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'predictions.db')


@st.cache_data(ttl=60)  # Cache for 60 seconds to avoid hammering S3
def download_db_from_s3():
    """Download the predictions database from S3 to a temp file. Returns path or None."""
    if not (S3_BUCKET and AWS_ACCESS_KEY and AWS_SECRET_KEY):
        return None

    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=S3_REGION
        )

        # Download to a temp file
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        tmp_file.close()
        s3.download_file(S3_BUCKET, 'predictions.db', tmp_file.name)
        return tmp_file.name
    except ClientError as e:
        st.error(f"Could not download database from S3: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error downloading from S3: {e}")
        return None


def get_db_path():
    """Try S3 first, fall back to local file."""
    s3_path = download_db_from_s3()
    if s3_path is not None:
        return s3_path
    elif os.path.exists(LOCAL_DB_PATH):
        return LOCAL_DB_PATH
    else:
        return None


DB_PATH = get_db_path()

# Page setup
st.set_page_config(
    page_title="NQ Volatility Bot",
    page_icon="📊",
    layout="wide",
)

st.title("NQ Intraday Volatility & Sentiment Bot")
st.caption("Live ML-powered volatility forecasting for NQ futures")


def load_data():
    """Load all predictions from the database."""
    if DB_PATH is None or not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY timestamp DESC",
        conn,
        parse_dates=['timestamp']
    )
    conn.close()
    return df


df = load_data()

if df.empty:
    st.error("No data available. Make sure the database exists.")
    st.stop()

# Latest prediction panel
st.header("Latest Signal")
latest = df.iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Predicted Next Hour",
    f"{latest['predicted_volatility']:.3f}%",
    delta=f"{latest['predicted_volatility'] - latest['current_volatility']:+.3f}%"
        if latest['current_volatility'] is not None else None
)
col2.metric("Current Vol", f"{latest['current_volatility']:.3f}%"
            if latest['current_volatility'] is not None else "N/A")
col3.metric("Sentiment Score", f"{latest['sentiment_score']:+.1f}"
            if latest['sentiment_score'] is not None else "N/A")
col4.metric("NQ Price", f"${latest['nq_price']:,.0f}"
            if latest['nq_price'] is not None else "N/A")

st.caption(f"Last updated: {latest['timestamp']}")

# Recent predictions chart
st.header("Recent Predictions vs. Actuals")

# Filter to predictions that have outcomes
df_with_outcomes = df[df['actual_volatility'].notna()].copy()

if not df_with_outcomes.empty:
    chart_df = df_with_outcomes.tail(60).set_index('timestamp')[
        ['predicted_volatility', 'actual_volatility']
    ]
    st.line_chart(chart_df)
    st.caption(
        "Predictions shown by calendar timestamp. Gaps reflect overnight and weekend periods when the bot doesn't run (RTH-only, 8 AM – 4 PM ET, weekdays)."
    )
else:
    st.info("No predictions with outcomes yet. Outcomes are filled in an hour after each prediction.")

# Recent predictions table
st.header("Recent Predictions")
display_cols = [
    'timestamp', 'sentiment_score', 'predicted_volatility',
    'current_volatility', 'actual_volatility', 'signal'
]
display_df = df[display_cols].head(20).copy()
# Format numeric columns
for col in ['sentiment_score', 'predicted_volatility', 'current_volatility', 'actual_volatility']:
    display_df[col] = display_df[col].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Summary stats
st.header("Performance Summary")
if not df_with_outcomes.empty:
    n = len(df_with_outcomes)
    mae = (df_with_outcomes['predicted_volatility'] - df_with_outcomes['actual_volatility']).abs().mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions with Outcomes", n)
    col2.metric("Mean Absolute Error", f"{mae:.4f}%")

    # Persistence baseline comparison
    if df_with_outcomes['current_volatility'].notna().all():
        baseline_mae = (df_with_outcomes['actual_volatility'] - df_with_outcomes['current_volatility']).abs().mean()
        improvement = (baseline_mae - mae) / baseline_mae * 100
        col3.metric(
            "Beats Persistence Baseline",
            f"{improvement:+.1f}%",
            delta=f"vs. baseline MAE: {baseline_mae:.4f}%"
        )