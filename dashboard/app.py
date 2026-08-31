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
import sys
from pathlib import Path
from botocore.exceptions import ClientError
# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calendar_events import get_upcoming_events, get_events_within
import config

# S3 configuration (read from .env locally, environment variables on Render)
S3_BUCKET = config.S3_BUCKET
S3_KEY = config.S3_KEY
AWS_ACCESS_KEY = config.AWS_ACCESS_KEY
AWS_SECRET_KEY = config.AWS_SECRET_KEY
S3_REGION = config.S3_REGION

# Local fallback path (used when running locally and S3 not configured)
LOCAL_DB_PATH = config.DB_PATH

# Predictions land hourly, so nothing newer than this means a cycle was missed.
STALE_AFTER_MINUTES = 90


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
        s3.download_file(S3_BUCKET, S3_KEY, tmp_file.name)
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

# === Bot liveness ===
# Without this, a dead bot looks identical to a healthy one — the page just
# keeps showing the last prediction it ever made.
latest_ts = pd.to_datetime(df['timestamp'].max())
if latest_ts.tzinfo is None:
    latest_ts = latest_ts.tz_localize('UTC')
age_minutes = (pd.Timestamp.now(tz='UTC') - latest_ts).total_seconds() / 60

if age_minutes > STALE_AFTER_MINUTES:
    if age_minutes < 60 * 24:
        age_text = f"{age_minutes / 60:.1f} hours"
    else:
        age_text = f"{age_minutes / 1440:.1f} days"
    st.error(
        f"⚠️ **Stale data** — the most recent prediction is {age_text} old. "
        f"Predictions should arrive hourly during market hours (8 AM – 4 PM ET, "
        f"weekdays), so the bot may not be running."
    )

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
    chart_df = df_with_outcomes.head(60).copy().sort_values('timestamp')

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df['timestamp'],
        y=chart_df['actual_volatility'],
        name='Actual Volatility',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x|%b %d %I:%M %p}</b><br>Actual: %{y:.3f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=chart_df['timestamp'],
        y=chart_df['predicted_volatility'],
        name='Predicted Volatility',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %d %I:%M %p}</b><br>Predicted: %{y:.3f}%<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Volatility (%)',
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Predictions shown by calendar timestamp. Gaps reflect overnight and weekend periods when the bot doesn't run (RTH-only, 8 AM – 4 PM ET, weekdays)."
    )
else:
    st.info("No predictions with outcomes yet. Outcomes are filled in an hour after each prediction.")
# === NEW: Upcoming Economic Events ===
st.header("📅 Upcoming Economic Events")

try:
    from datetime import datetime
    import pytz

    ET = pytz.timezone('US/Eastern')

    # Get high-impact events for next 14 days
    high_impact_events = get_upcoming_events(impact_levels=('high',), days_ahead=14)

    # Get imminent events (next 4 hours)
    imminent = get_events_within(minutes=240, impact_levels=('high', 'medium'))

    if imminent:
        for event in imminent[:3]:
            minutes = int((event['datetime_et'] - datetime.now(ET)).total_seconds() / 60)
            if event['impact'] == 'high':
                st.warning(f"⚠️ **{event['name']}** in {minutes} min ({event['datetime_et'].strftime('%I:%M %p ET')})")
            else:
                st.info(f"📅 {event['name']} in {minutes} min ({event['datetime_et'].strftime('%I:%M %p ET')})")

    if high_impact_events:
        events_data = []
        for e in high_impact_events[:10]:
            events_data.append({
                'When': e['datetime_et'].strftime('%a %b %d, %I:%M %p ET'),
                'Event': e['name'],
            })
        st.dataframe(pd.DataFrame(events_data), hide_index=True, width='stretch')
    else:
        st.info("No high-impact events scheduled in the next 14 days.")

except Exception as e:
    st.warning(f"Could not fetch economic calendar: {e}")

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
st.dataframe(display_df, width='stretch', hide_index=True)

# Summary stats
st.header("Performance Summary")
if not df_with_outcomes.empty:
    n = len(df_with_outcomes)
    mae = (df_with_outcomes['predicted_volatility'] - df_with_outcomes['actual_volatility']).abs().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predictions with Outcomes", n)
    col2.metric("Mean Absolute Error", f"{mae:.4f}%")
    col4.metric(
        "Last Prediction",
        f"{age_minutes:.0f} min ago" if age_minutes < 120 else f"{age_minutes / 60:.1f} hrs ago",
        delta="stale" if age_minutes > STALE_AFTER_MINUTES else "live",
        delta_color="inverse" if age_minutes > STALE_AFTER_MINUTES else "normal",
    )

    # Persistence baseline comparison
    if df_with_outcomes['current_volatility'].notna().all():
        baseline_mae = (df_with_outcomes['actual_volatility'] - df_with_outcomes['current_volatility']).abs().mean()
        improvement = (baseline_mae - mae) / baseline_mae * 100
        col3.metric(
            "Beats Persistence Baseline",
            f"{improvement:+.1f}%",
            delta=f"vs. baseline MAE: {baseline_mae:.4f}%"
        )
