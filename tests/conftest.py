import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def reset_telegram_globals(monkeypatch):
    try:
        import src.monitoring.telegram_bot as telegram_bot
    except ImportError:
        yield
        return

    telegram_bot._sent_messages.clear()
    telegram_bot._pending_fingerprints.clear()
    yield
    telegram_bot._sent_messages.clear()
    telegram_bot._pending_fingerprints.clear()
