import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

ET = pytz.timezone('US/Eastern')
UTC = pytz.UTC

# Get last 2 days
nq = yf.download('NQ=F', period='2d', interval='5m', progress=False)

print(f"Downloaded {len(nq)} bars")
print(f"UTC Date range: {nq.index[0]} to {nq.index[-1]}")

if len(nq) == 0:
    print("No data downloaded!")
    exit()

# Flatten MultiIndex columns
if isinstance(nq.columns, pd.MultiIndex):
    nq.columns = nq.columns.get_level_values(0)

# Calculate volatility
nq['Returns'] = nq['Close'].pct_change()
nq['Volatility'] = nq['Returns'].rolling(window=12).std() * 100
nq = nq.dropna()

# Convert to ET timezone
nq.index = nq.index.tz_convert('US/Eastern')

print(f"ET Date range: {nq.index[0]} to {nq.index[-1]}")

# Get all trading hours data (8 AM - 4 PM ET)
trading_data = nq[(nq.index.hour >= 8) & (nq.index.hour <= 16)]

if len(trading_data) == 0:
    print("\nNo trading hours data found!")
    exit()

# Get unique trading days
trading_days = trading_data.index.date
unique_days = sorted(set(trading_days))

print(f"\nTrading days in data: {unique_days}")

# Use the most recent trading day
latest_day = unique_days[-1]
today_data = trading_data[trading_data.index.date == latest_day]

print(f"\nShowing data for: {latest_day}")
print(f"Data points: {len(today_data)}")

print("\n" + "=" * 60)
print(f"VOLATILITY THROUGHOUT THE DAY - {latest_day}")
print("=" * 60)

# Show hourly summary
for hour in range(8, 17):
    hour_data = today_data[today_data.index.hour == hour]
    if len(hour_data) > 0:
        avg_vol = hour_data['Volatility'].mean()
        min_vol = hour_data['Volatility'].min()
        max_vol = hour_data['Volatility'].max()
        print(f"{hour:2d}:00-{hour + 1:2d}:00 | Avg: {avg_vol:.4f}% | Range: {min_vol:.4f}%-{max_vol:.4f}%")

# Show predictions vs actuals at update times
print("\n" + "=" * 60)
print("BOT PREDICTION TIMES (Compare to Discord)")
print("=" * 60)
print(f"{'Time':<12} {'Current Vol':<12} {'Next Hr Actual':<15}")
print("-" * 60)

for hour in range(8, 17):
    # Find :05 data (or closest)
    hour_window = today_data[(today_data.index.hour == hour) &
                             (today_data.index.minute >= 3) &
                             (today_data.index.minute <= 10)]

    if len(hour_window) > 0:
        closest = hour_window.iloc[0]
        timestamp = closest.name.strftime('%I:%M %p')
        current_vol = closest['Volatility']

        # Calculate actual next hour average
        next_hour_start = closest.name
        next_hour_end = next_hour_start + pd.Timedelta(hours=1)
        next_hour_data = today_data[(today_data.index > next_hour_start) &
                                    (today_data.index <= next_hour_end)]

        if len(next_hour_data) > 0:
            next_hour_avg = next_hour_data['Volatility'].mean()
            print(f"{timestamp:<12} {current_vol:.4f}%      {next_hour_avg:.4f}%")
        else:
            print(f"{timestamp:<12} {current_vol:.4f}%      (no data)")

print("\n" + "=" * 60)
print("Now check your Discord for predicted values at these times!")
print("Compare 'Predicted Next Hour' to 'Next Hr Actual' above")
print("=" * 60)