# NQ Intraday Volatility & Sentiment Bot

An automated market analysis system that combines news-sentiment classification with LSTM volatility forecasting for NQ futures, posting hourly signals to Discord during US market hours.

## Architecture

- **Sentiment classifier**: TF-IDF + logistic regression trained on labeled financial headlines, scoring news from 16 financial RSS feeds
- **Volatility forecaster**: Multi-feature LSTM trained on 5-minute NQ futures bars predicting next-hour realized volatility, with a self-learning per-hour bias correction applied on top
- **Persistence**: SQLite database recording every prediction and the actual realized outcome for ongoing evaluation
- **Deployment**: AWS EC2, managed by systemd with auto-restart on failure
- **Notification**: Discord webhook posts signal updates hourly during 8 AM - 4 PM ET

## Project Structure
```
financial-sentiment/
├── src/
│   ├── config.py            # Paths, credentials, shared thresholds
│   ├── nq_sentiment.py      # Main bot
│   ├── database.py          # SQLite persistence layer
│   └── calendar_events.py   # Finnhub economic calendar
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── scripts/
│   ├── inspect_nq_data.py         # Quick NQ data inspector
│   ├── backtest_predictions.py    # Historical accuracy analysis
│   ├── db_health.py               # Liveness check (exits non-zero if stale)
│   └── evaluate_sentiment.py      # Is the sentiment input earning its keep?
├── tests/                   # pytest suite
├── archive/                 # Abandoned experiments — see archive/README.md
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
5. Create a `.env` file (see Configuration)
6. Run: `python src/nq_sentiment.py`

### Configuration

All configuration resolves through `src/config.py`, which reads `.env` first and
then the real environment. Recognised keys:

| Key | Required | Purpose |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | yes | Where signals are posted |
| `FINNHUB_API_KEY` | no | Economic calendar; the calendar block is skipped without it |
| `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | no | Database sync for the dashboard |
| `AWS_REGION` | no | Defaults to `us-east-2` |
| `S3_KEY` | no | Object key for the synced database (default `predictions.db`) |
| `NQ_ENV` | on the server | Set to `production` to enable S3 upload |
| `NQ_DB_PATH` | no | Override the database location (default `data/predictions.db`) |

**`NQ_ENV=production` must be set on the server.** The bot only uploads to S3 in
production. It falls back to the old `hostname.startswith('ip-')` check when
`NQ_ENV` is unset, but that breaks on any non-default host name.

### Model Files

The trained models are not committed to keep the repo lightweight. The current production model (v2) consists of:

- `nq_intraday_volatility_v2_delta.keras` — LSTM predicting volatility deltas
- `intraday_v2_scaler_X.pkl` — feature scaler
- `intraday_v2_scaler_y.pkl` — target scaler
- `sentiment_model.pkl` — TF-IDF + logistic regression sentiment classifier
- `vectorizer.pkl` — TF-IDF vectorizer

The v1 model files are retained for rollback purposes:
- `nq_intraday_volatility_multifeature_lstm.keras`
- `intraday_multifeature_scaler_X.pkl`
- `intraday_multifeature_scaler_y.pkl`

> **scikit-learn is pinned to 1.6.1** because that is the version the `.pkl`
> files were written with. Loading them under a different version raises
> `InconsistentVersionWarning` and may produce invalid results. Re-fit and
> re-pickle before bumping the pin.

### Feature engineering note

`Time_Normalized` is derived from the **UTC** hour of each bar, which is what
the model was trained on — the fitted scaler's mean of 0.6678 corresponds to
16:00 UTC, the midpoint of the RTH session. Do not convert this feature to ET
without retraining; it would shift the input roughly 1.7σ out of distribution.

Because the feature is absolute UTC clock time, the DST transition shifts the
session's representation by one hour (~0.43σ). Expect a possible accuracy change
after DST boundaries until the feature is redefined as minutes-since-open.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

130 tests covering the persistence layer, signal thresholds, trading-hours
math, feed scraping, the outcome-backfill queue, and the economic calendar.
No test touches the network.

## Operations

```bash
python scripts/db_health.py          # exits 1 if no prediction in 90 min
python scripts/backtest_predictions.py
python scripts/evaluate_sentiment.py
python download_db.py                # pull the production DB from S3
```

`db_health.py` is designed for cron or a systemd timer — it exits non-zero when
the newest prediction is stale, which is the only automated signal that the bot
has stopped.

### Performance

Running in production since April 2026, recording every hourly prediction
alongside the realized outcome. All figures below are measured against a
persistence baseline that simply predicts "no change" — the bar a volatility
model has to clear to be worth running at all.

#### Live production (584 predictions, the number that matters)

| | MAE | vs persistence | Directional |
|---|---|---|---|
| Raw v2 model | 0.0275 | +6.3% | 54.4% |
| **Deployed (+ per-hour bias correction)** | **0.0231** | **+21.3%** | **75.7%** |

Measured walk-forward with no lookahead over 518 out-of-sample predictions.
Reproduce with `python scripts/backtest_predictions.py`.

The raw model does not perform in production the way it did in training: its
directional accuracy is a coin flip, because it predicts a *decrease* at nearly
every hour and so is systematically wrong across the 9:30 open. The per-hour
bias correction described below recovers that, and the deployed system now
roughly matches the original training-holdout numbers.

#### Training holdout (model v2)

These are the numbers from model development, on held-out data — **not**
production performance. They are recorded here for provenance; the live table
above supersedes them.

- MAE 0.0127% on volatility levels
- Beats persistence baseline by 22.9%
- Directional accuracy 70.9%
- Correlation with actuals 0.867

The gap between these and the raw live numbers is the reason the bias
correction exists. Most of it is attributable to the model never having seen
the pre-open → open transition (`rth_only: true` in training).

#### Model v1 (previous — level prediction)

The original model predicted absolute volatility levels and underperformed a persistence baseline on point-estimate MAE while still achieving 0.84 correlation and 53.7% directional accuracy. This is a known failure mode for highly autocorrelated time series — the model regressed toward the mean rather than predicting changes.

Pivoting to delta prediction (Phase 2) and restricting training data to regular trading hours produced the v2 training results above. Note that the same RTH restriction is what later cost the model the open transition in live trading.

### Per-hour bias correction

The v2 model predicts a *decrease* in volatility at nearly every hour — its
predicted deltas carry ~38% of the standard deviation of actual ones and are
negative ~71% of the time. That matches the afternoon, when volatility really
does decay, but it is badly wrong across the 9:30 open, where realized
volatility rises on the overwhelming majority of days while the model
predicts a rise only ~14% of the time. Training used `rth_only: true`, so the model never saw the
pre-open → open ramp and cannot represent it.

`database.get_hourly_bias_corrections()` measures the systematic miss per ET
hour and adds it back. The aggregate effect is in the Performance table above
(+6.3% → +21.3% over the baseline, 54.4% → 75.7% directional). Per hour, it
repairs exactly where the model was inverted:

| Hour (ET) | n | Raw | Corrected |
|---|---|---|---|
| 8:00 | 73 | 47.9% | 79.5% |
| 9:00 | 73 | **16.4%** | **93.2%** |
| 10:00 | 73 | 31.5% | 80.8% |

Two properties worth preserving if you touch this:

- The correction is learned from `predicted_volatility_raw`, **never** from an
  already-corrected prediction. Learning from corrected values would fold each
  correction into the next and compound without limit. There are tests for this.
- An hour with fewer than `MIN_BIAS_SAMPLES` (8) observations gets a correction
  of 0.0 rather than one fitted to noise, so new hours degrade to raw output.

`predicted_volatility` always stores what the bot published; the uncorrected
output goes to `predicted_volatility_raw`. Set `NQ_BIAS_CORRECTION=0` to publish
raw. Re-check it periodically with `python scripts/backtest_predictions.py`,
which prints the walk-forward comparison and says whether the correction is
still earning its place.

The deeper fix is retraining without `rth_only` so the model learns the ramp
directly; the correction is a cheaper substitute that needs no retraining.

### Known issue: the sentiment input carries no signal

Measured over 242 live predictions (`scripts/evaluate_sentiment.py`):

- The sentiment score has **never been negative** (range +3.30 to +44.26), so
  the `s < -20` gate has never fired and both BEARISH signals are unreachable.
- Correlation with realized volatility is **+0.06**; with next-hour NQ return,
  **-0.02**. Directional hit rate is 48.6% against a 49.0% base rate.

The classifier itself is sound on clear-cut headlines, but it is binary with no
Neutral class, so mundane headlines are forced to one side and average out to a
constant positive offset. Fixing it means scoring against a rolling baseline,
gating on `predict_proba` confidence, or retraining with a Neutral class.

## Tech Stack

Python 3.13, TensorFlow/Keras, scikit-learn, pandas, NumPy, yfinance, AWS EC2, systemd, SQLite
