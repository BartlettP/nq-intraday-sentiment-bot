"""
NQ Volatility Bot - Streamlit Dashboard

Reads the predictions database and displays current signal,
recent history, and prediction vs. actual chart.
"""
import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

# Path to the database (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'predictions.db')

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
    if not os.path.exists(DB_PATH):
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