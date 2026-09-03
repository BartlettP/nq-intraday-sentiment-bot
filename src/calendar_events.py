"""
Economic calendar integration via the ForexFactory weekly feed.

Fetches upcoming high-impact US economic events to provide context
for the volatility prediction model.

The feed covers the current week only, so the practical lookahead is 0-7 days
rather than the 14 the callers ask for. That is enough for the imminent-event
warning, which is the part that feeds the signal; the longer-horizon "next up"
line in Discord just shows fewer events later in the week.

Events are normalised to the shape the rest of this module expects:
country 'US', a lowercase impact, and a UTC 'YYYY-MM-DD HH:MM:SS' timestamp.
"""
import requests
import logging
import time
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

ET = pytz.timezone('US/Eastern')
UTC = pytz.UTC

# Cache: refresh every 6 hours
_cache = {'data': None, 'fetched_at': 0}
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

# The feed is unofficial and unauthenticated, so a silent switch to returning
# nothing is a real failure mode. Count consecutive empty fetches and complain
# once it stops looking like a quiet week.
_empty_fetches = 0
_EMPTY_FETCH_WARN_THRESHOLD = 4  # 4 * 6h cache TTL ~= a full day of nothing


def _fetch_calendar(days_ahead=14):
    """Fetch this week's US events from ForexFactory. Returns list of event dicts.

    days_ahead is an upper bound only - the feed never reaches beyond the
    current week regardless of what is asked for.
    """
    global _empty_fetches

    cutoff = datetime.now(UTC) + timedelta(days=days_ahead)

    try:
        response = requests.get(
            CALENDAR_URL,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as e:
        logger.error(f"Failed to fetch calendar from ForexFactory: {e}")
        return []

    events = []
    for e in raw:
        if e.get('country') != 'USD':
            continue
        try:
            # e.g. '2026-09-04T08:30:00-04:00' - already offset-aware.
            event_utc = datetime.fromisoformat(e['date']).astimezone(UTC)
        except (KeyError, TypeError, ValueError):
            continue
        if event_utc > cutoff:
            continue

        events.append({
            'country': 'US',
            'impact': (e.get('impact') or '').lower(),
            'time': event_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'event': e.get('title', 'Unknown'),
        })

    events.sort(key=lambda x: x['time'])

    if events:
        _empty_fetches = 0
    else:
        _empty_fetches += 1
        if _empty_fetches >= _EMPTY_FETCH_WARN_THRESHOLD:
            logger.warning(
                f"Economic calendar has returned no US events "
                f"{_empty_fetches} fetches running - the feed may have moved or "
                f"started blocking us ({CALENDAR_URL})"
            )

    return events


def get_upcoming_events(impact_levels=('high',), days_ahead=14, force_refresh=False):
    """
    Returns a list of upcoming US economic events.
    Each event has: name, datetime_et (datetime in ET), impact, original_time_utc.
    """
    now = time.time()

    # Use cache if recent
    cache_expired = now - _cache['fetched_at'] > _CACHE_TTL_SECONDS
    if force_refresh or _cache['data'] is None or cache_expired:
        raw_events = _fetch_calendar(days_ahead)
        _cache['data'] = raw_events
        _cache['fetched_at'] = now
        logger.info(f"Refreshed economic calendar: {len(raw_events)} events fetched")
    else:
        raw_events = _cache['data']

    # Filter to US, requested impact levels
    filtered = []
    for e in raw_events:
        if e.get('country') != 'US':
            continue
        if e.get('impact') not in impact_levels:
            continue

        # Parse time and convert UTC -> ET
        time_str = e.get('time', '')
        if not time_str:
            continue
        try:
            event_utc = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            event_utc = UTC.localize(event_utc)
            event_et = event_utc.astimezone(ET)
        except ValueError:
            continue

        # Skip events in the past
        if event_et < datetime.now(ET):
            continue

        filtered.append({
            'name': e.get('event', 'Unknown'),
            'datetime_et': event_et,
            'impact': e.get('impact', 'low'),
            'original_time_utc': time_str,
        })

    return filtered


def get_events_within(minutes, impact_levels=('high',)):
    """Get events occurring within the next N minutes."""
    events = get_upcoming_events(impact_levels)
    cutoff = datetime.now(ET) + timedelta(minutes=minutes)
    return [e for e in events if e['datetime_et'] <= cutoff]


def get_next_event_info(impact_levels=('high',)):
    """
    Returns info about the next upcoming event:
    {'event': dict, 'minutes_until': int} or None if no upcoming events.
    """
    events = get_upcoming_events(impact_levels)
    if not events:
        return None

    next_event = events[0]  # _fetch_calendar returns them sorted by time
    delta = next_event['datetime_et'] - datetime.now(ET)
    minutes_until = int(delta.total_seconds() / 60)

    return {
        'event': next_event,
        'minutes_until': minutes_until,
    }


def format_event_for_message(event_info, ref_time=None):
    """
    Format a single event for inclusion in a Discord message.
    Returns a string like "FOMC Minutes in 47 min" or "FOMC Minutes (May 20, 2:00 PM)"
    """
    if ref_time is None:
        ref_time = datetime.now(ET)

    event = event_info['event']
    minutes = event_info['minutes_until']

    if minutes < 60:
        return f"{event['name']} in {minutes}min"
    elif minutes < 24 * 60:
        return f"{event['name']} at {event['datetime_et'].strftime('%I:%M %p ET')}"
    else:
        return f"{event['name']} {event['datetime_et'].strftime('%a %b %d %I:%M %p ET')}"


# Test/debug interface
if __name__ == '__main__':
    print("Testing calendar_events module...")
    print(f"Source: {CALENDAR_URL}")

    print("\n=== High-impact events (this week) ===")
    events = get_upcoming_events(impact_levels=('high',))
    for e in events:
        print(f"  {e['datetime_et'].strftime('%a %m/%d %I:%M %p ET')} - {e['name']}")

    print("\n=== Events within next 4 hours (high+medium) ===")
    soon = get_events_within(minutes=240, impact_levels=('high', 'medium'))
    for e in soon:
        print(f"  {e['datetime_et'].strftime('%I:%M %p ET')} - {e['name']} [{e['impact']}]")

    print("\n=== Next high-impact event ===")
    next_info = get_next_event_info()
    if next_info:
        print(f"  {next_info['event']['name']} in {next_info['minutes_until']} minutes")
        print(f"  Formatted: {format_event_for_message(next_info)}")
    else:
        print("  No upcoming high-impact events")