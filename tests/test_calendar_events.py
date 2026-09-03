"""
Tests for the ForexFactory economic-calendar wrapper: feed parsing, filtering,
timezone conversion, and the 6-hour cache.
"""
from datetime import datetime, timedelta

import pytest
import pytz

UTC = pytz.UTC
ET = pytz.timezone('US/Eastern')


@pytest.fixture
def cal(monkeypatch):
    """calendar_events with a cleared cache and no network."""
    import calendar_events

    monkeypatch.setattr(calendar_events, '_cache', {'data': None, 'fetched_at': 0})
    monkeypatch.setattr(calendar_events, '_empty_fetches', 0)
    return calendar_events


def raw_event(name, when_utc, country='US', impact='high'):
    """An event as _fetch_calendar normalises it."""
    return {
        'event': name,
        'time': when_utc.strftime('%Y-%m-%d %H:%M:%S'),
        'country': country,
        'impact': impact,
    }


def _utc_in(**delta):
    return datetime.now(UTC) + timedelta(**delta)


def stub_fetch(cal, monkeypatch, events):
    monkeypatch.setattr(cal, '_fetch_calendar', lambda days_ahead=14: events)


class TestFiltering:
    def test_keeps_upcoming_us_high_impact_events(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [raw_event('CPI', _utc_in(days=2))])
        events = cal.get_upcoming_events(impact_levels=('high',))
        assert [e['name'] for e in events] == ['CPI']

    def test_drops_non_us_events(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [
            raw_event('ECB Rate Decision', _utc_in(days=1), country='EU'),
            raw_event('Nonfarm Payrolls', _utc_in(days=1), country='US'),
        ])
        events = cal.get_upcoming_events()
        assert [e['name'] for e in events] == ['Nonfarm Payrolls']

    def test_drops_unrequested_impact_levels(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [
            raw_event('CPI', _utc_in(days=1), impact='high'),
            raw_event('Truck Tonnage', _utc_in(days=1), impact='low'),
        ])
        assert [e['name'] for e in cal.get_upcoming_events(impact_levels=('high',))] == ['CPI']

        cal._cache['data'] = None  # force a refetch for the second filter
        both = cal.get_upcoming_events(impact_levels=('high', 'low'))
        assert len(both) == 2

    def test_drops_past_events(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [
            raw_event('Yesterday CPI', _utc_in(days=-1)),
            raw_event('Tomorrow CPI', _utc_in(days=1)),
        ])
        assert [e['name'] for e in cal.get_upcoming_events()] == ['Tomorrow CPI']

    def test_skips_malformed_and_missing_times(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [
            {'event': 'No time', 'country': 'US', 'impact': 'high'},
            {'event': 'Bad time', 'time': 'not-a-date', 'country': 'US', 'impact': 'high'},
            raw_event('Good', _utc_in(days=1)),
        ])
        assert [e['name'] for e in cal.get_upcoming_events()] == ['Good']

    def test_network_failure_returns_empty_not_error(self, cal, monkeypatch):
        def boom(*a, **kw):
            raise ConnectionError("feed unreachable")

        monkeypatch.setattr(cal.requests, 'get', boom)
        assert cal._fetch_calendar() == []


class TestTimezoneConversion:
    def test_converts_utc_to_eastern(self, cal, monkeypatch):
        # 2026-09-15 12:30 UTC is 08:30 ET (EDT, UTC-4)
        when = UTC.localize(datetime(2026, 9, 15, 12, 30))
        stub_fetch(cal, monkeypatch, [raw_event('CPI', when)])

        event = cal.get_upcoming_events()[0]
        assert event['datetime_et'].hour == 8
        assert event['datetime_et'].minute == 30
        assert event['datetime_et'].tzinfo is not None

    def test_original_utc_string_is_preserved(self, cal, monkeypatch):
        when = UTC.localize(datetime(2026, 9, 15, 12, 30))
        stub_fetch(cal, monkeypatch, [raw_event('CPI', when)])
        assert cal.get_upcoming_events()[0]['original_time_utc'] == '2026-09-15 12:30:00'


class TestGetEventsWithin:
    def test_includes_only_events_inside_the_window(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [
            raw_event('Soon', _utc_in(minutes=30)),
            raw_event('Later', _utc_in(hours=8)),
        ])
        names = [e['name'] for e in cal.get_events_within(minutes=120)]
        assert names == ['Soon']

    def test_empty_when_nothing_is_close(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [raw_event('Later', _utc_in(days=5))])
        assert cal.get_events_within(minutes=60) == []


class TestGetNextEventInfo:
    def test_returns_next_event_with_minutes_until(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [raw_event('CPI', _utc_in(minutes=90))])
        info = cal.get_next_event_info()
        assert info['event']['name'] == 'CPI'
        assert info['minutes_until'] == pytest.approx(90, abs=2)

    def test_none_when_no_events(self, cal, monkeypatch):
        stub_fetch(cal, monkeypatch, [])
        assert cal.get_next_event_info() is None


class TestFormatEventForMessage:
    def test_under_an_hour_shows_minutes(self, cal):
        info = {
            'event': {'name': 'CPI', 'datetime_et': datetime.now(ET) + timedelta(minutes=45)},
            'minutes_until': 45,
        }
        assert cal.format_event_for_message(info) == 'CPI in 45min'

    def test_same_day_shows_a_clock_time(self, cal):
        when = datetime.now(ET) + timedelta(hours=5)
        info = {'event': {'name': 'CPI', 'datetime_et': when}, 'minutes_until': 300}
        result = cal.format_event_for_message(info)
        assert 'CPI at' in result and 'ET' in result

    def test_beyond_a_day_shows_the_weekday(self, cal):
        when = datetime.now(ET) + timedelta(days=3)
        info = {'event': {'name': 'FOMC', 'datetime_et': when}, 'minutes_until': 3 * 24 * 60}
        result = cal.format_event_for_message(info)
        assert 'FOMC' in result
        assert when.strftime('%a') in result


class TestCaching:
    def test_second_call_within_ttl_does_not_refetch(self, cal, monkeypatch):
        calls = []

        def counting_fetch(days_ahead=14):
            calls.append(days_ahead)
            return [raw_event('CPI', _utc_in(days=1))]

        monkeypatch.setattr(cal, '_fetch_calendar', counting_fetch)

        cal.get_upcoming_events()
        cal.get_upcoming_events()
        cal.get_upcoming_events()

        assert len(calls) == 1, "the 6-hour cache is not being used"

    def test_force_refresh_refetches(self, cal, monkeypatch):
        calls = []
        monkeypatch.setattr(cal, '_fetch_calendar',
                            lambda days_ahead=14: calls.append(1) or [])

        cal.get_upcoming_events()
        cal.get_upcoming_events(force_refresh=True)
        assert len(calls) == 2

    def test_expired_cache_refetches(self, cal, monkeypatch):
        calls = []
        monkeypatch.setattr(cal, '_fetch_calendar',
                            lambda days_ahead=14: calls.append(1) or [])

        cal.get_upcoming_events()
        # Age the cache past its TTL.
        cal._cache['fetched_at'] -= cal._CACHE_TTL_SECONDS + 1
        cal.get_upcoming_events()

        assert len(calls) == 2


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def feed_row(title, iso_date, country='USD', impact='High'):
    """A row in ForexFactory's wire format, before normalisation."""
    return {'title': title, 'country': country, 'date': iso_date, 'impact': impact}


def stub_feed(cal, monkeypatch, rows):
    monkeypatch.setattr(cal.requests, 'get', lambda *a, **kw: FakeResponse(rows))


class TestFeedParsing:
    """_fetch_calendar is the one piece the other tests mock away, and it is
    the piece that breaks if the feed changes shape."""

    def test_keeps_only_usd_rows_and_relabels_them_us(self, cal, monkeypatch):
        soon = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [
            feed_row('Non-Farm Employment Change', soon.isoformat()),
            feed_row('ECB Rate Decision', soon.isoformat(), country='EUR'),
            feed_row('BOJ Outlook', soon.isoformat(), country='JPY'),
        ])
        events = cal._fetch_calendar()
        assert [e['event'] for e in events] == ['Non-Farm Employment Change']
        assert events[0]['country'] == 'US'

    def test_impact_is_lowercased_to_match_the_filters(self, cal, monkeypatch):
        soon = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [feed_row('CPI m/m', soon.isoformat())])
        assert cal._fetch_calendar()[0]['impact'] == 'high'

    def test_offset_aware_date_becomes_a_utc_timestamp(self, cal, monkeypatch):
        # The feed publishes ET offsets; downstream expects a UTC string.
        stub_feed(cal, monkeypatch, [
            feed_row('Non-Farm Employment Change', '2026-09-04T08:30:00-04:00'),
        ])
        events = cal._fetch_calendar(days_ahead=36500)
        assert events[0]['time'] == '2026-09-04 12:30:00'

    def test_round_trips_into_the_expected_eastern_time(self, cal, monkeypatch):
        when_et = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [feed_row('CPI m/m', when_et.isoformat())])

        event = cal.get_upcoming_events()[0]
        assert event['datetime_et'] == when_et

    def test_skips_rows_with_missing_or_malformed_dates(self, cal, monkeypatch):
        soon = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [
            {'title': 'No date', 'country': 'USD', 'impact': 'High'},
            feed_row('Bad date', 'not-a-date'),
            feed_row('Good', soon.isoformat()),
        ])
        assert [e['event'] for e in cal._fetch_calendar()] == ['Good']

    def test_drops_events_beyond_the_requested_horizon(self, cal, monkeypatch):
        near = (datetime.now(ET) + timedelta(days=2)).replace(microsecond=0)
        far = (datetime.now(ET) + timedelta(days=30)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [
            feed_row('Near', near.isoformat()),
            feed_row('Far', far.isoformat()),
        ])
        assert [e['event'] for e in cal._fetch_calendar(days_ahead=7)] == ['Near']

    def test_returns_events_sorted_by_time(self, cal, monkeypatch):
        base = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [
            feed_row('Third', (base + timedelta(hours=3)).isoformat()),
            feed_row('First', base.isoformat()),
            feed_row('Second', (base + timedelta(hours=1)).isoformat()),
        ])
        assert [e['event'] for e in cal._fetch_calendar()] == ['First', 'Second', 'Third']

    def test_a_holiday_row_survives_parsing_and_is_filtered_downstream(self, cal, monkeypatch):
        soon = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [feed_row('Labor Day', soon.isoformat(), impact='Holiday')])
        assert cal._fetch_calendar()[0]['impact'] == 'holiday'
        assert cal.get_upcoming_events(impact_levels=('high',)) == []


class TestEmptyFeedWarning:
    """A feed that quietly starts returning nothing is why the old provider
    sat broken unnoticed."""

    def test_stays_quiet_for_a_single_empty_fetch(self, cal, monkeypatch, caplog):
        stub_feed(cal, monkeypatch, [])
        with caplog.at_level('WARNING'):
            cal._fetch_calendar()
        assert 'no US events' not in caplog.text

    def test_warns_once_empty_fetches_stop_looking_like_a_quiet_week(self, cal, monkeypatch, caplog):
        stub_feed(cal, monkeypatch, [])
        with caplog.at_level('WARNING'):
            for _ in range(cal._EMPTY_FETCH_WARN_THRESHOLD):
                cal._fetch_calendar()
        assert 'no US events' in caplog.text

    def test_a_successful_fetch_resets_the_counter(self, cal, monkeypatch):
        stub_feed(cal, monkeypatch, [])
        for _ in range(cal._EMPTY_FETCH_WARN_THRESHOLD - 1):
            cal._fetch_calendar()

        soon = (datetime.now(ET) + timedelta(days=1)).replace(microsecond=0)
        stub_feed(cal, monkeypatch, [feed_row('CPI m/m', soon.isoformat())])
        cal._fetch_calendar()
        assert cal._empty_fetches == 0
