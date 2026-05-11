import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Your predictions from Discord (manually extracted)
predictions = [
    # 1/15/2026 (Wednesday)
    {'date': '2026-01-15', 'time': '11:05', 'predicted': 0.105},
    {'date': '2026-01-15', 'time': '12:05', 'predicted': 0.063},
    {'date': '2026-01-15', 'time': '13:05', 'predicted': 0.049},
    {'date': '2026-01-15', 'time': '14:05', 'predicted': 0.043},
    {'date': '2026-01-15', 'time': '15:05', 'predicted': 0.066},

    # 1/16/2026 (Thursday)
    {'date': '2026-01-16', 'time': '08:00', 'predicted': 0.034},
    {'date': '2026-01-16', 'time': '09:05', 'predicted': 0.046},
    {'date': '2026-01-16', 'time': '10:05', 'predicted': 0.085},
    {'date': '2026-01-16', 'time': '11:05', 'predicted': 0.108},
    {'date': '2026-01-16', 'time': '12:05', 'predicted': 0.070},
    {'date': '2026-01-16', 'time': '13:05', 'predicted': 0.041},
    {'date': '2026-01-16', 'time': '14:05', 'predicted': 0.039},
    {'date': '2026-01-16', 'time': '15:05', 'predicted': 0.054},

    # 1/19/2026 (Sunday - Holiday?) - All same prediction 0.083
    # Skipping these as they look like a data issue

    # 1/20/2026 (Monday)
    {'date': '2026-01-20', 'time': '08:00', 'predicted': 0.058},
    {'date': '2026-01-20', 'time': '09:05', 'predicted': 0.096},
    {'date': '2026-01-20', 'time': '10:05', 'predicted': 0.144},
    {'date': '2026-01-20', 'time': '11:05', 'predicted': 0.080},
    {'date': '2026-01-20', 'time': '12:05', 'predicted': 0.115},
    {'date': '2026-01-20', 'time': '13:05', 'predicted': 0.091},
    {'date': '2026-01-20', 'time': '14:05', 'predicted': 0.071},
    {'date': '2026-01-20', 'time': '15:05', 'predicted': 0.075},

    # 1/21/2026 (Tuesday)
    {'date': '2026-01-21', 'time': '08:00', 'predicted': 0.058},
    {'date': '2026-01-21', 'time': '09:05', 'predicted': 0.087},
    {'date': '2026-01-21', 'time': '11:05', 'predicted': 0.064},
    {'date': '2026-01-21', 'time': '12:05', 'predicted': 0.148},
    {'date': '2026-01-21', 'time': '13:05', 'predicted': 0.155},
    {'date': '2026-01-21', 'time': '14:05', 'predicted': 0.112},
    {'date': '2026-01-21', 'time': '15:05', 'predicted': 0.133},

    # 1/22/2026 (Today/Wednesday)
    {'date': '2026-01-22', 'time': '08:00', 'predicted': 0.027},
    {'date': '2026-01-22', 'time': '09:05', 'predicted': 0.066},
    {'date': '2026-01-22', 'time': '10:05', 'predicted': 0.136},
    {'date': '2026-01-22', 'time': '11:05', 'predicted': 0.101},
    {'date': '2026-01-22', 'time': '12:05', 'predicted': 0.053},
    {'date': '2026-01-22', 'time': '13:05', 'predicted': 0.042},
    {'date': '2026-01-22', 'time': '14:05', 'predicted': 0.035},
    {'date': '2026-01-22', 'time': '15:05', 'predicted': 0.043},
]

# Download NQ data
print("Downloading NQ data...")
nq = yf.download('NQ=F', start='2026-01-15', end='2026-01-23', interval='5m', progress=False)

if isinstance(nq.columns, pd.MultiIndex):
    nq.columns = nq.columns.get_level_values(0)

# Calculate volatility
nq['Returns'] = nq['Close'].pct_change()
nq['Volatility'] = nq['Returns'].rolling(window=12).std() * 100
nq = nq.dropna()

# Convert to ET
ET = pytz.timezone('US/Eastern')
nq.index = nq.index.tz_convert('US/Eastern')

# Calculate actuals for each prediction
results = []
for pred in predictions:
    pred_datetime = pd.Timestamp(f"{pred['date']} {pred['time']}", tz='US/Eastern')

    # Find actual data 1 hour later
    one_hour_later = pred_datetime + pd.Timedelta(hours=1)

    # Get volatility data for next hour
    mask = (nq.index > pred_datetime) & (nq.index <= one_hour_later)
    next_hour_data = nq.loc[mask, 'Volatility']

    if len(next_hour_data) > 0:
        actual = next_hour_data.mean()
        error = abs(pred['predicted'] - actual)

        results.append({
            'datetime': pred_datetime,
            'predicted': pred['predicted'],
            'actual': actual,
            'error': error,
            'error_pct': (error / actual * 100) if actual > 0 else 0
        })

# Analysis
if results:
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("FULL WEEK ACCURACY ANALYSIS")
    print("=" * 70)

    print(f"\n📊 Overall Performance ({len(results)} predictions):")
    print(f"   Average Error (MAE): {df['error'].mean():.4f}%")
    print(f"   Median Error: {df['error'].median():.4f}%")
    print(f"   Best Prediction: {df['error'].min():.4f}%")
    print(f"   Worst Prediction: {df['error'].max():.4f}%")
    print(f"   Std Dev of Errors: {df['error'].std():.4f}%")

    # Compare to test set
    print(f"\n🎯 Comparison to Test Set:")
    print(f"   Test MAE: 0.0102%")
    print(f"   Production MAE: {df['error'].mean():.4f}%")
    if df['error'].mean() < 0.0102:
        print(f"   ✅ Better than test! (+{((0.0102 - df['error'].mean()) / 0.0102 * 100):.1f}%)")
    else:
        print(f"   ⚠️ Slightly worse ({((df['error'].mean() - 0.0102) / 0.0102 * 100):.1f}% higher error)")

    # Directional accuracy
    print(f"\n📈 Directional Analysis:")
    df['actual_change'] = df['actual'].diff()
    df['pred_change'] = df['predicted'].diff()
    df_direction = df[1:]  # Skip first row (no previous to compare)

    correct_direction = ((df_direction['actual_change'] > 0) == (df_direction['pred_change'] > 0)).sum()
    directional_acc = correct_direction / len(df_direction) * 100

    print(f"   Directional Accuracy: {directional_acc:.1f}%")
    print(f"   Test Directional: 66.7%")
    print(f"   Random Guess: 50%")

    # By time of day
    print(f"\n⏰ Performance by Time of Day:")
    df['hour'] = df['datetime'].dt.hour
    for hour in sorted(df['hour'].unique()):
        hour_data = df[df['hour'] == hour]
        print(f"   {hour:2d}:00 - MAE: {hour_data['error'].mean():.4f}% ({len(hour_data)} predictions)")

    # Print worst predictions
    print(f"\n❌ Top 5 Worst Predictions:")
    worst = df.nlargest(5, 'error')[['datetime', 'predicted', 'actual', 'error']]
    for idx, row in worst.iterrows():
        print(
            f"   {row['datetime'].strftime('%m/%d %I:%M %p')}: Pred {row['predicted']:.3f}%, Actual {row['actual']:.3f}%, Error {row['error']:.4f}%")

    # Print best predictions
    print(f"\n✅ Top 5 Best Predictions:")
    best = df.nsmallest(5, 'error')[['datetime', 'predicted', 'actual', 'error']]
    for idx, row in best.iterrows():
        print(
            f"   {row['datetime'].strftime('%m/%d %I:%M %p')}: Pred {row['predicted']:.3f}%, Actual {row['actual']:.3f}%, Error {row['error']:.4f}%")

    import pandas as pd

    # From your results
    # Classify by date
    df['week'] = df['datetime'].dt.date.apply(
        lambda x: 'Week 1 (Normal)' if x <= pd.Timestamp('2026-01-16').date()
        else 'Week 2 (News Event)'
    )

    print("\n" + "=" * 60)
    print("PERFORMANCE BY MARKET REGIME")
    print("=" * 60)

    for week in ['Week 1 (Normal)', 'Week 2 (News Event)']:
        week_data = df[df['week'] == week]
        print(f"\n{week}:")
        print(f"  Predictions: {len(week_data)}")
        print(f"  MAE: {week_data['error'].mean():.4f}%")
        print(f"  Median Error: {week_data['error'].median():.4f}%")
        print(f"  Max Error: {week_data['error'].max():.4f}%")

    # Check if NQ actually went up during bullish sentiment
    # Download NQ daily closes
    nq_daily = yf.download('NQ=F', start='2026-01-15', end='2026-01-23', interval='1d')

    print("\n" + "=" * 60)
    print("NQ DAILY PERFORMANCE")
    print("=" * 60)

    # Handle MultiIndex columns if present
    if isinstance(nq_daily.columns, pd.MultiIndex):
        nq_daily.columns = nq_daily.columns.get_level_values(0)

    # Calculate daily changes
    nq_daily['Prev_Close'] = nq_daily['Close'].shift(1)
    nq_daily['Change_Pct'] = ((nq_daily['Close'] - nq_daily['Prev_Close']) / nq_daily['Prev_Close'] * 100)

    # Print results
    for date in nq_daily.index:
        change = nq_daily.loc[date, 'Change_Pct']

        if not pd.isna(change):
            direction = "📈 UP" if change > 0 else "📉 DOWN"
            print(f"{date.date()}: {direction} {change:+.2f}%")

    # Compare to your sentiment scores
    print("\nYour average sentiment: +28 (Bullish)")
    print("Did NQ trend match sentiment?")