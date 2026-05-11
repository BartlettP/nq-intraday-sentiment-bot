# NQ Intraday Volatility & Sentiment Bot

An automated market analysis system that combines news-sentiment classification with LSTM volatility forecasting for NQ futures, posting hourly signals to Discord during US market hours.

## Architecture

- **Sentiment classifier**: TF-IDF + logistic regression trained on labeled financial headlines, scoring news from 16 financial RSS feeds
- **Volatility forecaster**: Multi-feature LSTM trained on 5-minute NQ futures bars predicting next-hour realized volatility
- **Persistence**: SQLite database recording every prediction and the actual realized outcome for ongoing evaluation
- **Deployment**: AWS EC2, managed by systemd with auto-restart on failure
- **Notification**: Discord webhook posts signal updates hourly during 8 AM - 4 PM ET

## Project Structure
```
financial-sentiment/
├── src/
│   ├── nq_sentiment.py     # Main bot
│   └── database.py         # SQLite persistence layer
├── scripts/
│   ├── inspect_nq_data.py        # Quick NQ data inspector
│   └── backtest_predictions.py   # Historical accuracy analysis
├── models/                  # Not committed - see Setup
├── data/                    # SQLite database - not committed
├── logs/                    # Runtime logs - not committed
└── requirements.txt
```
## Setup

### Local Development

1. Clone the repo
2. Create a virtual environment: `python -m venv .venv && .venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Obtain trained model files (see below) and place in `models/`
5. Create a `.env` file with `DISCORD_WEBHOOK_URL=<your-webhook-url>`
6. Run: `python src/nq_sentiment.py`

### Model Files

The trained models are not committed to keep the repo lightweight. The five required files are:
- `nq_intraday_volatility_multifeature_lstm.keras`
- `intraday_multifeature_scaler_X.pkl`
- `intraday_multifeature_scaler_y.pkl`
- `sentiment_model.pkl`
- `vectorizer.pkl`

To regenerate: see the training notebook in `notebooks/`.

## Performance

The system has been continuously running in production since April 2026, collecting hourly predictions paired with actual outcomes. Predictions and accuracy metrics are persisted to SQLite for ongoing evaluation.

Recent performance metrics are available via `scripts/backtest_predictions.py`.

## Tech Stack

Python 3.12, TensorFlow/Keras, scikit-learn, pandas, NumPy, yfinance, AWS EC2, systemd, SQLite