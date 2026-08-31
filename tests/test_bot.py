"""
Tests for nq_sentiment's pure logic and its network-facing helpers.

Every network call is stubbed — the `bot` fixture makes a real one an error.
"""
from datetime import datetime, timedelta

import pytest
import pytz

ET = pytz.timezone('US/Eastern')


def _et(year, month, day, hour, minute=0):
    return ET.localize(datetime(year, month, day, hour, minute))


@pytest.fixture
def frozen_clock(monkeypatch):
    """Pin nq_sentiment's view of 'now'. It does `from datetime import datetime`,
    so the name to patch lives on the module."""
    def _freeze(bot, when):
        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return when.astimezone(tz) if tz else when.replace(tzinfo=None)
        monkeypatch.setattr(bot, 'datetime', FrozenDatetime)
    return _freeze


class FakeResponse:
    def __init__(self, content=b'', status_code=200, text=''):
        self.content = content
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>Fed holds rates steady as inflation cools</title></item>
<item><title>Nasdaq futures climb on strong earnings</title></item>
<item><title>Short</title></item>
</channel></rss>"""


class TestGenerateSignal:
    def test_missing_inputs_return_insufficient_data(self, bot, sentiment, volatility):
        assert "Insufficient data" in bot.generate_signal(None, volatility(0.09))
        assert "Insufficient data" in bot.generate_signal(sentiment(30.0), None)
        assert "Insufficient data" in bot.generate_signal(None, None)

    def test_strong_bullish(self, bot, sentiment, volatility):
        assert "STRONG BULLISH" in bot.generate_signal(sentiment(30.0), volatility(0.15))

    def test_weak_bullish(self, bot, sentiment, volatility):
        assert "WEAK BULLISH" in bot.generate_signal(sentiment(30.0), volatility(0.04))

    def test_strong_bearish(self, bot, sentiment, volatility):
        assert "STRONG BEARISH" in bot.generate_signal(sentiment(-30.0), volatility(0.15))

    def test_weak_bearish(self, bot, sentiment, volatility):
        assert "WEAK BEARISH" in bot.generate_signal(sentiment(-30.0), volatility(0.04))

    def test_high_volatility_uncertain_direction(self, bot, sentiment, volatility):
        assert "HIGH VOLATILITY" in bot.generate_signal(sentiment(5.0), volatility(0.15))

    def test_neutral_fallback(self, bot, sentiment, volatility):
        assert "NEUTRAL" in bot.generate_signal(sentiment(5.0), volatility(0.08))

    @pytest.mark.parametrize("score", [20.0, -20.0])
    def test_sentiment_threshold_is_exclusive(self, bot, sentiment, volatility, score):
        """s > 20 / s < -20, so exactly 20 is not 'strong'."""
        assert "STRONG" not in bot.generate_signal(sentiment(score), volatility(0.15))

    def test_dead_zone_between_thresholds_is_neutral(self, bot, sentiment, volatility):
        """Volatility in [0.06, 0.10] with strong sentiment hits no specific
        branch and falls through to NEUTRAL — worth pinning so a future
        threshold edit is a deliberate choice."""
        assert "NEUTRAL" in bot.generate_signal(sentiment(50.0), volatility(0.08))


class TestTradingHours:
    @pytest.mark.parametrize("hour,expected", [
        (7, False),   # before pre-market
        (8, True),    # pre-market open
        (12, True),
        (16, True),   # close boundary is inclusive
        (17, False),
    ])
    def test_weekday_hours(self, bot, frozen_clock, hour, expected):
        frozen_clock(bot, _et(2026, 8, 26, hour))  # Wednesday
        assert bot.is_trading_time() is expected

    @pytest.mark.parametrize("day", [29, 30])  # Sat, Sun
    def test_weekends_are_closed(self, bot, frozen_clock, day):
        frozen_clock(bot, _et(2026, 8, day, 12))
        assert bot.is_trading_time() is False

    def test_before_open_waits_until_today(self, bot, frozen_clock):
        frozen_clock(bot, _et(2026, 8, 26, 6))  # Wed 6am, opens 8am
        assert bot.time_until_next_session() == pytest.approx(2 * 3600, abs=60)

    def test_after_close_waits_until_tomorrow(self, bot, frozen_clock):
        frozen_clock(bot, _et(2026, 8, 26, 17))  # Wed 5pm -> Thu 8am
        assert bot.time_until_next_session() == pytest.approx(15 * 3600, abs=60)

    def test_friday_evening_waits_until_monday(self, bot, frozen_clock):
        frozen_clock(bot, _et(2026, 8, 28, 17))  # Fri 5pm -> Mon 8am
        assert bot.time_until_next_session() == pytest.approx(63 * 3600, abs=60)

    def test_saturday_waits_until_monday(self, bot, frozen_clock):
        frozen_clock(bot, _et(2026, 8, 29, 12))  # Sat noon -> Mon 8am
        assert bot.time_until_next_session() == pytest.approx(44 * 3600, abs=60)

    def test_wait_is_always_positive(self, bot, frozen_clock):
        """A non-positive wait would spin the loop hot."""
        for day in range(24, 31):
            for hour in (0, 7, 9, 16, 23):
                frozen_clock(bot, _et(2026, 8, day, hour))
                assert bot.time_until_next_session() > 0


class TestAnalyzeSentiment:
    def test_empty_headlines_return_none(self, bot):
        assert bot.analyze_sentiment([]) is None

    def test_scores_bullish_and_bearish(self, bot, monkeypatch):
        class FakeVectorizer:
            def transform(self, headlines):
                return headlines

        class FakeClassifier:
            def predict(self, X):
                return ['Bullish', 'Bullish', 'Bullish', 'Bearish']

        monkeypatch.setattr(bot, 'vectorizer', FakeVectorizer())
        monkeypatch.setattr(bot, 'sentiment_classifier', FakeClassifier())

        result = bot.analyze_sentiment(['a', 'b', 'c', 'd'])
        assert result['bullish'] == 3
        assert result['bearish'] == 1
        assert result['total'] == 4
        assert result['score'] == pytest.approx(50.0)  # (3-1)/4 * 100

    def test_classifier_failure_returns_none_not_raise(self, bot, monkeypatch):
        class Broken:
            def transform(self, headlines):
                raise ValueError("vectorizer not fitted")

        monkeypatch.setattr(bot, 'vectorizer', Broken())
        assert bot.analyze_sentiment(['a']) is None


class TestScrapeNews:
    """Regression coverage for fix #4 (timeouts) and the silent `except: pass`."""

    def test_passes_a_timeout_to_every_request(self, bot, monkeypatch):
        seen = []

        def fake_get(url, **kwargs):
            seen.append(kwargs.get('timeout'))
            return FakeResponse(RSS)

        monkeypatch.setattr(bot.requests, 'get', fake_get)
        bot.scrape_news()

        assert len(seen) == len(bot.NEWS_FEEDS)
        assert all(t == bot.FEED_TIMEOUT for t in seen), \
            "a feed request without a timeout can hang the whole loop"

    def test_deduplicates_and_filters_short_titles(self, bot, monkeypatch):
        monkeypatch.setattr(bot.requests, 'get', lambda url, **kw: FakeResponse(RSS))
        headlines = bot.scrape_news()

        # Same two feeds' worth of content repeated across all 16 feeds.
        assert len(headlines) == 2
        assert "Short" not in headlines

    def test_one_bad_feed_does_not_lose_the_others(self, bot, monkeypatch):
        def fake_get(url, **kwargs):
            if 'bloomberg' in url:
                raise TimeoutError("connect timed out")
            return FakeResponse(RSS)

        monkeypatch.setattr(bot.requests, 'get', fake_get)
        assert len(bot.scrape_news()) == 2

    def test_feed_failures_are_logged(self, bot, monkeypatch, caplog):
        """The old bare `except: pass` meant a permanently dead feed was invisible."""
        monkeypatch.setattr(
            bot.requests, 'get',
            lambda url, **kw: (_ for _ in ()).throw(TimeoutError("timed out"))
        )
        with caplog.at_level('WARNING', logger='nq_bot'):
            assert bot.scrape_news() == []
        assert 'feeds failed' in caplog.text

    def test_http_error_status_is_treated_as_failure(self, bot, monkeypatch):
        monkeypatch.setattr(bot.requests, 'get', lambda url, **kw: FakeResponse(b'', 429))
        assert bot.scrape_news() == []


class TestSendDiscord:
    def test_passes_a_timeout(self, bot, monkeypatch, sentiment, volatility):
        captured = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return FakeResponse(status_code=204)

        monkeypatch.setattr(bot.requests, 'post', fake_post)
        monkeypatch.setattr(bot, 'DISCORD_WEBHOOK_URL', 'https://example.invalid/hook')
        bot.send_discord(sentiment(30.0), volatility(0.12))

        assert captured.get('timeout') == bot.DISCORD_TIMEOUT

    def test_non_204_response_is_logged(self, bot, monkeypatch, caplog, sentiment, volatility):
        monkeypatch.setattr(
            bot.requests, 'post',
            lambda url, **kw: FakeResponse(status_code=429, text='rate limited')
        )
        monkeypatch.setattr(bot, 'DISCORD_WEBHOOK_URL', 'https://example.invalid/hook')

        with caplog.at_level('WARNING', logger='nq_bot'):
            bot.send_discord(sentiment(0.0), volatility(0.08))
        assert '429' in caplog.text

    def test_missing_inputs_send_nothing(self, bot, volatility):
        # requests.post is the fixture's raising stub, so this asserts no call.
        bot.send_discord(None, volatility(0.09))
        bot.send_discord(None, None)


class TestEventContext:
    """Regression coverage for fix #1 — get_upcoming_events was never imported,
    so this whole block raised NameError into a swallowed warning."""

    def test_builds_a_block_without_raising(self, bot, monkeypatch):
        soon = bot.datetime.now(bot.ET) + timedelta(days=3)
        monkeypatch.setattr(bot, 'get_events_within', lambda **kw: [])
        monkeypatch.setattr(
            bot, 'get_upcoming_events',
            lambda **kw: [{'name': 'CPI', 'datetime_et': soon, 'impact': 'high'}]
        )

        result = bot.get_event_context_for_discord()
        assert 'CPI' in result
        assert 'Upcoming' in result

    def test_imminent_event_is_flagged(self, bot, monkeypatch):
        imminent = {
            'name': 'FOMC Minutes',
            'datetime_et': bot.datetime.now(bot.ET) + timedelta(minutes=45),
            'impact': 'high',
        }
        monkeypatch.setattr(bot, 'get_events_within', lambda **kw: [imminent])
        monkeypatch.setattr(bot, 'get_upcoming_events', lambda **kw: [imminent])

        result = bot.get_event_context_for_discord()
        assert 'FOMC Minutes' in result
        assert 'min' in result

    def test_no_events_yields_empty_string(self, bot, monkeypatch):
        monkeypatch.setattr(bot, 'get_events_within', lambda **kw: [])
        monkeypatch.setattr(bot, 'get_upcoming_events', lambda **kw: [])
        assert bot.get_event_context_for_discord() == ""

    def test_calendar_failure_degrades_quietly(self, bot, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("finnhub down")
        monkeypatch.setattr(bot, 'get_events_within', boom)
        assert bot.get_event_context_for_discord() == ""

    def test_the_module_actually_exports_what_the_bot_imports(self):
        """Catches the class of bug directly: a name used but not imported."""
        import calendar_events
        import nq_sentiment
        for name in ('get_upcoming_events', 'get_events_within'):
            assert hasattr(nq_sentiment, name), f"{name} is not imported into nq_sentiment"
            assert hasattr(calendar_events, name)


class TestThrottleConfig:
    """Fix #2: the off-schedule shift check used to run on every 30s tick,
    hitting all 16 feeds each time."""

    def test_shift_check_is_less_frequent_than_the_tick(self, bot):
        assert bot.SHIFT_CHECK_INTERVAL > bot.LOOP_TICK_SECONDS

    def test_daily_scrape_volume_is_bounded(self, bot):
        session_seconds = 8 * 3600
        max_scrapes = session_seconds / bot.SHIFT_CHECK_INTERVAL + 8  # + hourly runs
        requests_per_day = max_scrapes * len(bot.NEWS_FEEDS)
        assert requests_per_day < 2000, (
            f"{requests_per_day:.0f} feed requests/day is enough to get rate-limited"
        )
