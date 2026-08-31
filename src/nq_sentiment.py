import feedparser
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time, timedelta
import requests
import time as time_module
import os
import pytz
from tensorflow.keras.models import load_model
import logging
from logging.handlers import RotatingFileHandler
from database import (
    init_db,
    insert_prediction,
    get_predictions_needing_outcomes,
    update_outcome,
    record_outcome_attempt,
    get_bias_correction_for_hour,
    DB_PATH,
)
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from calendar_events import get_upcoming_events, get_events_within
import config



# === LOGGING SETUP ===

# Log file lives next to the script, in a logs/ subdirectory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'nq_bot.log')

# Rotating file handler: 10 MB per file, keep 5 old files
log_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
console_handler = logging.StreamHandler()

formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger('nq_bot')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console_handler)

# Don't propagate to root logger (avoids duplicate messages)
logger.propagate = False
# === CONFIGURATION ===

# Credentials and paths resolve through config (one .env parser for the
# whole project, instead of the two hand-rolled ones that used to exist).
DISCORD_WEBHOOK_URL = config.DISCORD_WEBHOOK_URL

S3_BUCKET = config.S3_BUCKET
S3_KEY = config.S3_KEY
AWS_ACCESS_KEY = config.AWS_ACCESS_KEY
AWS_SECRET_KEY = config.AWS_SECRET_KEY
S3_REGION = config.S3_REGION

s3_client = None
if S3_BUCKET and AWS_ACCESS_KEY and AWS_SECRET_KEY:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=S3_REGION
        )
        logger.info(f"✅ S3 sync enabled (bucket: {S3_BUCKET})")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize S3 client: {e}")
        s3_client = None
else:
    logger.info("ℹ️ S3 sync disabled (credentials not configured)")


ET = pytz.timezone('US/Eastern')
PRE_MARKET_START = time(8, 0)
MARKET_CLOSE = time(16, 0)

# How often the main loop wakes up to check the clock.
LOOP_TICK_SECONDS = 30
# Minimum gap between the off-schedule "did sentiment lurch?" checks. Each one
# hits all 16 RSS feeds, so running it on every tick was ~15k requests/day and
# a fast route to being rate-limited.
SHIFT_CHECK_INTERVAL = 300  # 5 minutes
# Per-feed HTTP timeouts (connect, read). Without these a single hung feed
# blocks the whole single-threaded loop indefinitely.
FEED_TIMEOUT = (5, 10)
DISCORD_TIMEOUT = 10

# Per-hour bias correction (see database.get_hourly_bias_corrections).
# Set NQ_BIAS_CORRECTION=0 to publish the raw model output instead.
APPLY_BIAS_CORRECTION = config.env('NQ_BIAS_CORRECTION', '1') not in ('0', 'false', 'False')

NEWS_FEEDS = [
    'https://feeds.bloomberg.com/markets/news.rss',
    'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'https://www.reuters.com/rssFeed/businessNews',
    'https://www.wsj.com/xml/rss/3_7085.xml',
    'https://www.ft.com/rss/home',
    'https://www.investing.com/rss/news.rss',
    'https://finance.yahoo.com/news/rssindex',
    'https://www.marketwatch.com/rss/topstories',
    'https://www.barrons.com/rss',
    'https://seekingalpha.com/market_currents.xml',
    'https://www.forbes.com/real-time/feed2/',
    'https://www.investors.com/feed/',
    'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'https://feeds.content.dowjones.io/public/rss/mw_marketpulse',
    'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
    'https://www.zerohedge.com/fullrss2.xml',
]
# Build an absolute path to the models folder regardless of where the script is run from
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# === MODELS ===
# Populated by load_models(), which main() calls at startup. Keeping these out
# of module scope means the module can be imported (by tests, or the dashboard)
# without loading TensorFlow or starting the bot.
sentiment_classifier = None
vectorizer = None
volatility_model = None
scaler_X = None
scaler_y = None

# === HELPER FUNCTIONS ===

def is_trading_time():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return PRE_MARKET_START <= now.time() <= MARKET_CLOSE


def time_until_next_session():
    now_et = datetime.now(ET)
    current_weekday = now_et.weekday()

    if current_weekday == 5:
        days = 2
    elif current_weekday == 6:
        days = 1
    else:
        days = 0

    if now_et.time() > MARKET_CLOSE and days == 0:
        days = 1
        if current_weekday == 4:
            days = 3

    next_start = now_et.replace(hour=8, minute=0, second=0, microsecond=0)
    if days > 0:
        next_start = next_start + timedelta(days=days)
    elif now_et.time() >= PRE_MARKET_START:
        next_start = next_start + timedelta(days=1)

    return (next_start - now_et).total_seconds()


def scrape_news():
    headlines = []
    seen = set()
    failed = []

    for feed_url in NEWS_FEEDS:
        try:
            # Fetch ourselves rather than letting feedparser open the URL, so
            # we control the timeout. feedparser.parse() has none and will
            # block the loop forever on an unresponsive host.
            response = requests.get(
                feed_url,
                timeout=FEED_TIMEOUT,
                headers={'User-Agent': 'nq-bot/1.0 (+RSS reader)'},
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries[:25]:
                if hasattr(entry, 'title') and len(entry.title) > 10:
                    # Deduplicate headlines
                    title_lower = entry.title.lower()
                    if title_lower not in seen:
                        headlines.append(entry.title)
                        seen.add(title_lower)
        except Exception as e:
            # Never swallow silently — a permanently dead feed should be visible.
            failed.append(f"{feed_url.split('/')[2]}: {type(e).__name__}")

    if failed:
        logger.warning(f"   ⚠️ {len(failed)}/{len(NEWS_FEEDS)} feeds failed: {', '.join(failed)}")

    return headlines

def get_event_context_for_discord():
    """Returns a formatted block of upcoming high-impact events for Discord."""
    try:
        # Events in next 2 hours (highest priority - imminent)
        imminent = get_events_within(minutes=120, impact_levels=('high',))

        # All upcoming high-impact events (next 14 days)
        upcoming = get_upcoming_events(impact_levels=('high',))

        lines = []

        # Show imminent events with the warning emoji
        for e in imminent[:1]:  # max 1 imminent event
            minutes = int((e['datetime_et'] - datetime.now(ET)).total_seconds() / 60)
            lines.append(f"⚠️ {e['name']} in {minutes}min")

        # Show next 2-3 upcoming (non-imminent) high-impact events
        non_imminent = [e for e in upcoming if e not in imminent]
        if not lines:  # no imminent events, show upcoming
            for e in non_imminent[:2]:
                lines.append(f"📅 {e['name']} — {e['datetime_et'].strftime('%a %b %d %I:%M %p ET')}")
        elif non_imminent:  # we have imminent, also show one more upcoming
            e = non_imminent[0]
            lines.append(f"📅 Next: {e['name']} — {e['datetime_et'].strftime('%a %b %d %I:%M %p ET')}")

        if not lines:
            return ""

        return "📅 **Upcoming:**\n" + "\n".join(lines)

    except Exception as e:
        logger.warning(f"   ⚠️ Calendar fetch failed (continuing): {e}")
        return ""

def analyze_sentiment(headlines):
    if not headlines:
        return None
    try:
        predictions = sentiment_classifier.predict(vectorizer.transform(headlines))
        bullish = list(predictions).count('Bullish')
        bearish = list(predictions).count('Bearish')
        total = len(predictions)
        score = ((bullish - bearish) / total) * 100
        return {
            'score': score,
            'bullish': bullish,
            'bearish': bearish,
            'total': total,
            'bullish_pct': (bullish / total * 100),
            'bearish_pct': (bearish / total * 100)
        }
    except Exception as e:
        logger.warning(f"   ⚠️ Sentiment error: {e}")
        return None


def predict_intraday_volatility():
    """
    Predict volatility for next 1 hour using intraday multi-feature model
    """
    try:
        # Download recent 5-min data
        nq = yf.download('NQ=F', period='2d', interval='5m', progress=False)

        if len(nq) == 0:
            logger.warning("   ⚠️ No NQ data downloaded")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(nq.columns, pd.MultiIndex):
            nq.columns = nq.columns.get_level_values(0)

        # Calculate returns
        nq['Returns'] = nq['Close'].pct_change()
        nq = nq.dropna().copy()  # .copy() — dropna returns a view; assigning to it warns

        # Calculate features
        # 1. Volatility
        nq['Volatility'] = nq['Returns'].rolling(window=12).std() * 100

        # 2. Volume features
        nq['Volume_MA'] = nq['Volume'].rolling(window=12).mean()
        nq['Volume_Ratio'] = np.where(
            nq['Volume_MA'] > 0,
            nq['Volume'] / nq['Volume_MA'],
            1.0
        )

        # 3. Price momentum
        nq['Price_Change'] = nq['Close'].pct_change(periods=12) * 100

        # 4. Range
        nq['Range'] = (nq['High'] - nq['Low']) / nq['Close'] * 100

        # 5. Time of day
        nq['Hour'] = nq.index.hour
        nq['Minute'] = nq.index.minute
        nq['Time_Normalized'] = (nq['Hour'] * 60 + nq['Minute']) / (24 * 60)

        # Remove NaN
        nq = nq.dropna()

        if len(nq) < 48:
            logger.warning(f"   ⚠️ Not enough data: {len(nq)} bars (need 48)")
            return None

        # Get last 48 bars of features
        feature_cols = ['Volatility', 'Volume_Ratio', 'Price_Change', 'Range', 'Time_Normalized']
        recent_features = nq[feature_cols].values[-48:]  # Shape: (48, 5)

        # Reshape to (48, 5) for scaler, then (1, 48, 5) for model
        X_2d = recent_features.reshape(48, 5)
        X_scaled_2d = scaler_X.transform(X_2d)
        X_scaled_3d = X_scaled_2d.reshape(1, 48, 5)

        # Predict (v2 model outputs DELTA from current volatility, not level)
        y_pred_scaled = volatility_model.predict(X_scaled_3d, verbose=0)
        predicted_delta = scaler_y.inverse_transform(y_pred_scaled)[0, 0]

        # Get current volatility, 4h average, and current price
        current_vol = nq['Volatility'].values[-1]
        last_4h_avg = np.mean(nq['Volatility'].values[-48:])
        current_nq_price = nq['Close'].iloc[-1]

        raw_level = max(current_vol + predicted_delta, 0.0)

        # Apply the learned per-hour bias correction. The model systematically
        # predicts decay and cannot represent the volatility ramp across the
        # 9:30 open; this supplies the diurnal component it's missing.
        correction = 0.0
        if APPLY_BIAS_CORRECTION:
            try:
                correction = get_bias_correction_for_hour(datetime.now(ET).hour)
            except Exception as e:
                # Never let a history query stop a prediction going out.
                logger.warning(f"   ⚠️ Bias correction unavailable, using raw: {e}")
                correction = 0.0

        corrected_delta = predicted_delta + correction
        predicted_level = max(current_vol + corrected_delta, 0.0)

        if correction:
            logger.info(
                f"   🔧 Bias correction {correction:+.4f} applied "
                f"({raw_level:.4f} -> {predicted_level:.4f})"
            )

        return {
            # What the bot publishes and what gets evaluated.
            'predicted_next_hour': float(predicted_level),
            # The model's uncorrected output, kept so the correction can keep
            # being learned from an unpolluted signal.
            'predicted_next_hour_raw': float(raw_level),
            'bias_correction': float(correction),
            'current': float(current_vol),
            'last_4h_avg': float(last_4h_avg),
            'change': float(corrected_delta),
            'change_raw': float(predicted_delta),
            'nq_price': float(current_nq_price)
        }

    except Exception as e:
        logger.error(f"   ❌ Volatility prediction error: {e}")
        return None

def get_volatility_context(vol_pct):
    """
    Translate volatility % into actionable context.
    Delegates to the single shared scale in config so this can't drift out of
    agreement with the Discord embed and the signal wording again.
    """
    return config.volatility_level(vol_pct)




def generate_signal(sentiment, volatility):
    if not sentiment or not volatility:
        return "⚠️ Insufficient data"

    s = sentiment['score']
    v = volatility['predicted_next_hour']

    # Thresholds come from config so all three call sites agree on what
    # "quiet" and "active" mean.
    strong = config.SENTIMENT_STRONG
    neutral = config.SENTIMENT_NEUTRAL
    quiet = config.VOL_QUIET_MAX
    active = config.VOL_ACTIVE_MIN

    if s > strong and v > active:
        return "🚀 STRONG BULLISH - High conviction + big moves expected"
    elif s > strong and v < quiet:
        return "📈 WEAK BULLISH - Positive but small moves (patience needed)"
    elif s < -strong and v > active:
        return "💥 STRONG BEARISH - High conviction + big moves expected"
    elif s < -strong and v < quiet:
        return "📉 WEAK BEARISH - Negative but small moves"
    elif abs(s) < neutral and v > active:
        return "⚡ HIGH VOLATILITY - Uncertain direction, caution"
    else:
        return "😐 NEUTRAL - Low conviction, range-bound"


def update_past_outcomes():
    """
    For predictions made ~1 hour ago, fetch what NQ actually did and record it.
    """
    pending = get_predictions_needing_outcomes(min_hours_old=1)
    if not pending:
        return

    logger.info(f"📊 Checking outcomes for {len(pending)} past predictions...")

    try:
        # 5 days, not 1 — a prediction made Friday afternoon needs data that
        # only becomes reachable after the weekend. With period='1d' anything
        # that missed its window on the first pass could never be filled.
        nq = yf.download('NQ=F', period='5d', interval='5m', progress=False)
        if isinstance(nq.columns, pd.MultiIndex):
            nq.columns = nq.columns.get_level_values(0)
        nq['Returns'] = nq['Close'].pct_change()
        nq = nq.dropna().copy()  # .copy() — dropna returns a view; assigning to it warns
        nq['Volatility'] = nq['Returns'].rolling(window=12).std() * 100
        nq = nq.dropna()

        # NQ index is in UTC by default from yfinance
        if nq.index.tz is None:
            nq.index = nq.index.tz_localize('UTC')

        unresolved = []
        for pred in pending:
            pred_time = pd.Timestamp(pred['timestamp']).tz_localize('UTC')
            # Take the 12 5-min bars (= 1 hour) right after the prediction
            future_window = nq[nq.index >= pred_time].head(12)

            if len(future_window) >= 12:
                actual_vol = float(future_window['Volatility'].mean())
                update_outcome(pred['id'], actual_vol)
                logger.info(
                    f"   ✅ Updated #{pred['id']}: predicted {pred['predicted_volatility']:.4f}, actual {actual_vol:.4f}"
                )
            else:
                unresolved.append(pred['id'])

        # Count the miss so rows that can never be resolved eventually drop out
        # of the queue instead of being re-fetched on every cycle forever.
        if unresolved:
            record_outcome_attempt(unresolved)
            logger.info(
                f"   ⏳ {len(unresolved)} prediction(s) still awaiting data "
                f"(attempt recorded)"
            )

    except Exception as e:
        logger.warning(f"   ⚠️ Outcome update error: {e}")

def send_discord(sentiment, volatility, shift_info=None):
    if not sentiment or not volatility:
        return

    # Calculate values
    vol_pct = volatility['predicted_next_hour']
    if 'nq_price' in volatility:
        nq_range = (vol_pct / 100) * volatility['nq_price']
    else:
        # Fallback if price not available
        nq_range = vol_pct * 250

    # Volatility level — same scale the signal wording and dashboard use.
    vol_label, _ = config.volatility_level(vol_pct)

    # Sentiment direction
    if sentiment['score'] > config.SENTIMENT_STRONG:
        sent_label = "📈 Bullish"
        color = 0x00ff00
    elif sentiment['score'] < -config.SENTIMENT_STRONG:
        sent_label = "📉 Bearish"
        color = 0xff0000
    else:
        sent_label = "➡️ Neutral"
        color = 0xffff00

    # Trading signal
    signal = generate_signal(sentiment, volatility)

    # Build message
    description = f"""**Next Hour: {vol_pct:.3f}%** {vol_label}
    Expected Range: **±{nq_range:.0f} points**

    **Sentiment: {sentiment['score']:+.1f}** {sent_label}
    News: {sentiment['bullish']}🟢 / {sentiment['bearish']}🔴

    **Signal:** {signal}"""

    # Add upcoming event context if available
    event_context = get_event_context_for_discord()
    if event_context:
        description = f"{description}\n\n{event_context}"

    if shift_info:
        description = f"🚨 **Sentiment Shift:** {shift_info}\n\n{description}"

    embed = {
        "embeds": [{
            "title": f"📊 NQ Update - {datetime.now(ET).strftime('%I:%M %p')}",
            "description": description,
            "color": color,
            "footer": {"text": "Updates every hour"}
        }]
    }
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=embed, timeout=DISCORD_TIMEOUT)
        if response.status_code == 204:
            logger.info("   ✅ Discord sent!")
        else:
            # 429 (rate limited) and 400 (bad embed) used to pass silently.
            logger.warning(
                f"   ⚠️ Discord returned {response.status_code}: {response.text[:200]}"
            )
    except Exception as e:
        logger.warning(f"   ⚠️ Discord error: {e}")


def upload_db_to_s3():
    """Upload the predictions database to S3 for the dashboard to read.
    Only uploads in production (skips local test runs)."""
    import sqlite3
    import tempfile

    if s3_client is None:
        return False

    # Gate on NQ_ENV, falling back to the old hostname sniff. See
    # config.is_production() for why the hostname check alone was fragile.
    if not config.is_production():
        logger.info("   ℹ️ Skipping S3 upload (not running in production)")
        return False

    snapshot = None
    try:
        if not os.path.exists(DB_PATH):
            logger.warning("   ⚠️ Database file not found for S3 upload")
            return False

        # Copy through SQLite's backup API rather than uploading the live file.
        # Uploading a database that has a write in flight can ship a torn page
        # and hand the dashboard a corrupt file.
        fd, snapshot = tempfile.mkstemp(suffix='.db', prefix='nq_upload_')
        os.close(fd)
        with sqlite3.connect(DB_PATH) as source, sqlite3.connect(snapshot) as dest:
            source.backup(dest)

        s3_client.upload_file(snapshot, S3_BUCKET, S3_KEY)
        logger.info(f"   ☁️ Uploaded database to S3 (s3://{S3_BUCKET}/{S3_KEY})")
        return True
    except ClientError as e:
        logger.error(f"   ❌ S3 upload failed: {e}")
        return False
    except Exception as e:
        logger.error(f"   ❌ Unexpected error during S3 upload: {e}")
        return False
    finally:
        if snapshot and os.path.exists(snapshot):
            try:
                os.unlink(snapshot)
            except OSError:
                pass


# === MAIN LOOP ===

def run_loop():
    """The bot's main scheduling loop. Runs until interrupted."""
    previous_score = None
    update_count = 0
    last_run_hour = -1  # Track last run to avoid duplicates
    last_shift_check = 0.0  # Monotonic timestamp of the last off-schedule check

    while True:
        try:
            now_et = datetime.now(ET)
            logger.debug(f"Loop tick at {now_et.strftime('%Y-%m-%d %H:%M:%S')} ET")

            if not is_trading_time():
                logger.info(f"💤 CLOSED ({now_et.strftime('%A %I:%M %p')} ET)")
                wait = time_until_next_session()
                logger.info(f"⏰ Sleeping {wait / 3600:.1f} hours...")
                time_module.sleep(wait)
                last_run_hour = -1  # Reset for next session
                continue

            current_hour = now_et.hour
            current_minute = now_et.minute

            # Run at each hour
            should_run = False


            if current_minute == 0 and current_hour != last_run_hour:
                # Regular hourly update
                should_run = True
                last_run_hour = current_hour

            if should_run:
                update_count += 1
                logger.info(f"⏰ Update #{update_count} - {now_et.strftime('%I:%M %p')} ET")

                # Get sentiment
                headlines = scrape_news()
                logger.info(f"📰 {len(headlines)} headlines")
                sentiment = analyze_sentiment(headlines)

                # The scheduled run just refreshed the feeds; don't let the
                # off-schedule check scrape again immediately after.
                last_shift_check = time_module.monotonic()

                # Get intraday volatility
                volatility = predict_intraday_volatility()

                if sentiment:
                    logger.info(f"💭 Sentiment: {sentiment['score']:+.1f}")
                if volatility:
                    logger.info(f"⚡ Next hour vol: {volatility['predicted_next_hour']:.3f}%")
                    logger.info(f"📊 Current vol: {volatility['current']:.3f}%")

                # Check for BIG sentiment shifts (threshold: 20 points)
                shift_info = None
                if sentiment and previous_score is not None:
                    shift = sentiment['score'] - previous_score
                    if abs(shift) > 20:  # Big shift (increased threshold)
                        shift_info = f"{previous_score:+.1f} → {sentiment['score']:+.1f}"
                        logger.info(f"🚨 MAJOR sentiment shift detected!")

                # Send to Discord (always on scheduled run, or if major shift)
                if sentiment and volatility:
                    logger.info("📤 Sending to Discord...")
                    send_discord(sentiment, volatility, shift_info)

                    # Record this prediction in the database
                    signal_text = generate_signal(sentiment, volatility)
                    pred_id = insert_prediction(sentiment, volatility, signal_text)
                    logger.info(f"📊 Recorded prediction #{pred_id}")
                    update_past_outcomes()

                    # Upload database to S3 so dashboard can read latest data
                    upload_db_to_s3()

                if sentiment:
                    previous_score = sentiment['score']

            # If major shift detected outside of scheduled time, send immediate
            # update. Throttled: this hits all 16 feeds, so it must not run on
            # every tick.
            elif (previous_score is not None
                  and time_module.monotonic() - last_shift_check >= SHIFT_CHECK_INTERVAL):
                last_shift_check = time_module.monotonic()

                # Quick check for major shifts between scheduled updates
                headlines = scrape_news()
                sentiment = analyze_sentiment(headlines)

                if sentiment:
                    shift = sentiment['score'] - previous_score
                    if abs(shift) > 25:  # VERY major shift (even higher threshold)
                        logger.info(f"🚨 EMERGENCY UPDATE - {now_et.strftime('%I:%M %p')} ET")
                        logger.info(f"💭 Sentiment shift: {previous_score:+.1f} → {sentiment['score']:+.1f}")

                        volatility = predict_intraday_volatility()
                        if volatility:
                            shift_info = f"{previous_score:+.1f} → {sentiment['score']:+.1f}"
                            logger.info("📤 Sending emergency Discord update...")
                            send_discord(sentiment, volatility, shift_info)
                            previous_score = sentiment['score']

            time_module.sleep(LOOP_TICK_SECONDS)

        except KeyboardInterrupt:
            logger.info("👋 Stopped")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            logger.info("   Retrying in 1 minute...")
            time_module.sleep(60)


def load_models():
    """Load the sentiment and volatility models into module globals."""
    global sentiment_classifier, vectorizer, volatility_model, scaler_X, scaler_y

    logger.info("📂 Loading models...")
    try:
        sentiment_classifier = joblib.load(os.path.join(MODEL_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))
        volatility_model = load_model(os.path.join(MODEL_DIR, 'nq_intraday_volatility_v2_delta.keras'))
        scaler_X = joblib.load(os.path.join(MODEL_DIR, 'intraday_v2_scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(MODEL_DIR, 'intraday_v2_scaler_y.pkl'))
        logger.info("✅ All models loaded!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise SystemExit(1)


def main():
    if not DISCORD_WEBHOOK_URL:
        logger.error("DISCORD_WEBHOOK_URL not found in .env or environment variables")
        raise SystemExit(1)

    load_models()

    init_db()
    logger.info("✅ Database initialized")

    logger.info("🚀 NQ INTRADAY BOT STARTED")
    logger.info("Sentiment + Intraday Volatility | Updates at each hour")

    run_loop()


if __name__ == '__main__':
    main()