#!/usr/bin/env bash
#
# Run the liveness check and alert to Discord if the bot has gone quiet.
#
# Intended for cron during market hours only -- outside 8am-4pm ET on a
# weekday the bot is *supposed* to be idle, so checking then would be noise:
#
#   30 10,12,14,16 * * 1-5  /home/ubuntu/nq-intraday-sentiment-bot/scripts/health_alert.sh
#
# Alerts are throttled to one per ALERT_INTERVAL seconds so a multi-day
# outage doesn't spam the channel, and a single recovery message is sent when
# the bot comes back.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

STATE_FILE="${NQ_ALERT_STATE:-$REPO_DIR/logs/.health_alert_state}"
ALERT_INTERVAL="${NQ_ALERT_INTERVAL:-21600}"   # 6 hours
STALE_AFTER="${NQ_STALE_AFTER:-90}"            # minutes
LOG="$REPO_DIR/logs/health_alert.log"

mkdir -p "$(dirname "$LOG")"
now=$(date +%s)

# Read the webhook from .env without echoing it.
WEBHOOK=$(grep -m1 '^DISCORD_WEBHOOK_URL=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')

notify() {
    [ -n "$WEBHOOK" ] || return 0
    # jq isn't guaranteed to be installed; escape the few characters that matter.
    local msg="${1//\\/\\\\}"; msg="${msg//\"/\\\"}"; msg="${msg//$'\n'/\\n}"
    curl -sS -m 15 -H 'Content-Type: application/json' \
         -d "{\"content\":\"$msg\"}" "$WEBHOOK" >/dev/null 2>&1
}

output=$(python3 scripts/db_health.py --stale-after "$STALE_AFTER" 2>&1)
status=$?

last_alert=0
last_state="ok"
if [ -f "$STATE_FILE" ]; then
    read -r last_alert last_state < "$STATE_FILE" 2>/dev/null || true
    : "${last_alert:=0}"; : "${last_state:=ok}"
fi

if [ "$status" -ne 0 ]; then
    age_line=$(printf '%s\n' "$output" | grep -E '^\s+age' | tr -s ' ')
    if [ $((now - last_alert)) -ge "$ALERT_INTERVAL" ]; then
        notify "🚨 **NQ bot is stale** — no prediction in over ${STALE_AFTER} min.${age_line:+\\n\`${age_line}\`}\\nHost: $(hostname). Check: \`systemctl status sentiment\`"
        echo "$now stale" > "$STATE_FILE"
        echo "$(date '+%F %T') ALERT SENT (exit $status)" >> "$LOG"
    else
        echo "$now stale" > "$STATE_FILE"
        echo "$(date '+%F %T') stale, alert throttled" >> "$LOG"
    fi
else
    if [ "$last_state" = "stale" ]; then
        notify "✅ **NQ bot recovered** — predictions are flowing again on $(hostname)."
        echo "$(date '+%F %T') RECOVERY SENT" >> "$LOG"
    fi
    echo "0 ok" > "$STATE_FILE"
fi

exit "$status"
