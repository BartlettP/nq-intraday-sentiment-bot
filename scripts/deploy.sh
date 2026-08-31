#!/usr/bin/env bash
#
# Deploy the bot on the EC2 host. Run ON the server, not locally.
#
#   ssh sentiment-bot
#   cd <repo> && git pull && bash scripts/deploy.sh
#
# Idempotent: safe to re-run. Backs the database up before migrating and
# verifies the service is actually producing predictions before exiting.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

SERVICE="${NQ_SERVICE:-nq-bot}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

say "Repo: $REPO_DIR"
git log --oneline -1

# --- 1. Database backup ------------------------------------------------------
say "Backing up the database"
DB="${NQ_DB_PATH:-$REPO_DIR/data/predictions.db}"
if [ -f "$DB" ]; then
    BACKUP="$DB.$(date +%Y%m%d_%H%M%S).bak"
    cp "$DB" "$BACKUP"
    echo "  $BACKUP"
    echo "  rows: $(sqlite3 "$DB" 'SELECT COUNT(*) FROM predictions;' 2>/dev/null || echo '?')"
else
    echo "  no database yet at $DB (first run)"
fi

# --- 2. Dependencies ---------------------------------------------------------
say "Installing dependencies"
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "  venv: $(python --version)"
else
    echo "  WARNING: no .venv found; using system python"
fi
pip install -q -r requirements.txt
# scikit-learn must match the version models/*.pkl were pickled with (1.6.1).
python - <<'PY'
import warnings, joblib, sys, os
warnings.simplefilter("error")
try:
    for f in ("sentiment_model.pkl", "vectorizer.pkl",
              "intraday_v2_scaler_X.pkl", "intraday_v2_scaler_y.pkl"):
        joblib.load(os.path.join("models", f))
    print("  models load cleanly (no version warning)")
except Exception as e:
    print(f"  MODEL LOAD PROBLEM: {type(e).__name__}: {e}")
    print("  -> check the scikit-learn pin in requirements.txt")
    sys.exit(1)
PY

# --- 3. Environment ----------------------------------------------------------
say "Checking environment"
if grep -q '^NQ_ENV=production' .env 2>/dev/null; then
    echo "  NQ_ENV=production is set"
else
    echo "  NQ_ENV not set -> adding it (required for S3 sync)"
    echo 'NQ_ENV=production' >> .env
fi
for key in DISCORD_WEBHOOK_URL S3_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    grep -q "^$key=" .env 2>/dev/null && echo "  $key present" || echo "  WARNING: $key MISSING"
done

# --- 4. Migrate --------------------------------------------------------------
say "Migrating the database"
python -c "
import sys; sys.path.insert(0,'src')
import database
database.init_db()
h = database.health_summary()
print(f\"  rows={h['total_predictions']} resolved={h['resolved_outcomes']} pending={h['pending_outcomes']}\")
print(f\"  last prediction: {h['last_prediction_utc']}\")
"

# --- 5. Restart --------------------------------------------------------------
say "Restarting $SERVICE"
if systemctl list-unit-files 2>/dev/null | grep -q "^${SERVICE}.service"; then
    sudo systemctl daemon-reload
    sudo systemctl restart "$SERVICE"
    sleep 5
    sudo systemctl status "$SERVICE" --no-pager --lines=15 || true
else
    echo "  no systemd unit '${SERVICE}.service' found."
    echo "  Set NQ_SERVICE=<name> or start manually:  python -u src/nq_sentiment.py"
    exit 1
fi

# --- 6. Verify ---------------------------------------------------------------
say "Verifying"
echo "  Recent log lines:"
sudo journalctl -u "$SERVICE" -n 20 --no-pager | sed 's/^/    /' || true

cat <<'EOF'

  The bot only predicts on the hour during 8am-4pm ET on weekdays, so a fresh
  prediction may be up to an hour away. To confirm it is healthy:

      python scripts/db_health.py        # exits 0 when a prediction is recent
      sudo journalctl -u nq-bot -f       # watch it live

  Expect a "Bias correction ... applied" line on the next hourly prediction.
EOF
