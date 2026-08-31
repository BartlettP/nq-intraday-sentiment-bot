"""
Single source of truth for paths, credentials, and shared thresholds.

Before this existed the database lived at three different paths depending on
which script you ran (src/database.py -> data/, download_db.py -> repo root,
check_db.py -> a hardcoded absolute path), and the copies had silently
diverged. Everything now resolves through here.
"""
import os
from pathlib import Path

# === PATHS ===

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
LOG_DIR = PROJECT_ROOT / 'logs'


def _load_env_file():
    """Parse .env into a dict. Kept dependency-free so the bot can start on a
    bare EC2 box; python-dotenv is used by the standalone scripts."""
    env = {}
    env_path = PROJECT_ROOT / '.env'
    if env_path.exists():
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key.strip()] = value.strip()
    return env


_ENV = _load_env_file()


def env(name, default=None):
    """Look up config from .env first, then the real environment."""
    return _ENV.get(name) or os.environ.get(name) or default


# The database path. Override with NQ_DB_PATH to point a script at a downloaded
# production snapshot instead of the local one.
DB_PATH = str(Path(env('NQ_DB_PATH', str(DATA_DIR / 'predictions.db'))))

# === CREDENTIALS ===

DISCORD_WEBHOOK_URL = env('DISCORD_WEBHOOK_URL')
FINNHUB_API_KEY = env('FINNHUB_API_KEY')

S3_BUCKET = env('S3_BUCKET')
S3_KEY = env('S3_KEY', 'predictions.db')
AWS_ACCESS_KEY = env('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = env('AWS_SECRET_ACCESS_KEY')
S3_REGION = env('AWS_REGION', 'us-east-2')


def is_production():
    """Whether this process should sync to S3.

    Set NQ_ENV=production on the server. The previous check sniffed for a
    hostname starting with 'ip-', which silently stops working the moment the
    bot moves off a default-named EC2 instance (ECS, Fargate, a renamed host).
    The hostname check is kept as a fallback so an un-updated deployment keeps
    working.
    """
    declared = env('NQ_ENV', '').lower()
    if declared:
        return declared == 'production'

    import socket
    return socket.gethostname().startswith('ip-')


# === SHARED THRESHOLDS ===

# One volatility scale for the whole project. There used to be three
# disagreeing tables (generate_signal, send_discord, and the never-called
# get_volatility_context), so the same reading got different labels depending
# on which code path produced it.
#
# Boundaries are the production-facing ones from send_discord. Against the 232
# recorded outcomes they bucket roughly 38/27/22/8/4 percent, which is a
# reasonable spread over the observed range (p10 0.032, median 0.075, p90 0.168).
VOLATILITY_LEVELS = (
    # (upper_bound_exclusive, short_label, description)
    (0.06, "🟢 Very Low", "Extremely tight range, minimal moves"),
    (0.10, "🟡 Low", "Small moves expected, be patient"),
    (0.15, "🟠 Normal", "Standard trading conditions"),
    (0.25, "🔴 High", "Increased volatility, bigger moves"),
    (float('inf'), "🔥 Very High", "Very volatile, caution advised"),
)

# Where "quiet" ends and "big moves" begin, for signal wording. These index
# into VOLATILITY_LEVELS above rather than being independent magic numbers.
VOL_QUIET_MAX = VOLATILITY_LEVELS[0][0]   # 0.06
VOL_ACTIVE_MIN = VOLATILITY_LEVELS[1][0]  # 0.10

# Sentiment score gates for the signal taxonomy.
SENTIMENT_STRONG = 20
SENTIMENT_NEUTRAL = 10


def volatility_level(vol_pct):
    """Return (label, description) for a predicted volatility percentage."""
    for upper, label, description in VOLATILITY_LEVELS:
        if vol_pct < upper:
            return label, description
    return VOLATILITY_LEVELS[-1][1], VOLATILITY_LEVELS[-1][2]
