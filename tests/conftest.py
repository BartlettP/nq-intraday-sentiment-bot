"""
Shared pytest fixtures.

src/ isn't a package (the bot's modules import each other by bare name, e.g.
`from database import ...`), so it goes on sys.path directly.
"""
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fresh, isolated predictions database.

    database.DB_PATH is read inside get_connection() on every call, so pointing
    it at a temp file redirects the whole module without touching data/.
    """
    import database

    monkeypatch.setattr(database, 'DB_PATH', str(tmp_path / 'test_predictions.db'))
    database.init_db()
    return database


@pytest.fixture
def bot(monkeypatch):
    """The nq_sentiment module, importable without models or network.

    Importing it is now side-effect free (no model load, no loop, no exit) —
    that's what the main()/run_loop() split bought us.
    """
    import nq_sentiment

    # Nothing in the tested functions should reach the network; make it loud
    # if something tries.
    def _no_network(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(nq_sentiment.requests, 'get', _no_network)
    monkeypatch.setattr(nq_sentiment.requests, 'post', _no_network)
    return nq_sentiment


@pytest.fixture
def sentiment():
    """A representative sentiment dict as analyze_sentiment() returns it."""
    def _make(score, total=100):
        bullish = int(round((score / 100 * total + total) / 2))
        bearish = total - bullish
        return {
            'score': score,
            'bullish': bullish,
            'bearish': bearish,
            'total': total,
            'bullish_pct': bullish / total * 100,
            'bearish_pct': bearish / total * 100,
        }
    return _make


@pytest.fixture
def volatility():
    """A representative volatility dict as predict_intraday_volatility() returns it."""
    def _make(predicted, current=0.08, nq_price=20000.0):
        return {
            'predicted_next_hour': predicted,
            'current': current,
            'last_4h_avg': current,
            'change': predicted - current,
            'nq_price': nq_price,
        }
    return _make
