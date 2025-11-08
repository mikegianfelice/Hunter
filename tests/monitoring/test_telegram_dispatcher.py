import time

import pytest

import src.monitoring.telegram_bot as telegram_bot


def test_dispatcher_retries_until_success(monkeypatch):
    attempts = {"count": 0}

    def fake_send(job):
        attempts["count"] += 1
        return attempts["count"] >= 2

    dispatcher = telegram_bot.TelegramDispatcher(max_retries=3, base_backoff=0.01)
    monkeypatch.setattr(telegram_bot, "_send_message_sync", fake_send)

    dispatcher.start()
    dispatcher.submit({"payload": {"chat_id": "1", "text": "hi"}, "message_type": "test", "deduplicate": False})
    dispatcher._queue.join()
    dispatcher.stop()

    assert attempts["count"] == 2


def test_dispatcher_gives_up(monkeypatch):
    attempts = {"count": 0}

    def fake_send(job):
        attempts["count"] += 1
        return False

    dispatcher = telegram_bot.TelegramDispatcher(max_retries=2, base_backoff=0.01)
    monkeypatch.setattr(telegram_bot, "_send_message_sync", fake_send)

    dispatcher.start()
    dispatcher.submit({"payload": {"chat_id": "1", "text": "fail"}, "message_type": "fail", "deduplicate": False})
    dispatcher._queue.join()
    dispatcher.stop()

    assert attempts["count"] == 3  # initial + retries


def test_send_message_deduplicates(monkeypatch):
    monkeypatch.setattr(telegram_bot, "TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr(telegram_bot, "TELEGRAM_CHAT_ID", "chat")

    calls = {"count": 0}

    def fake_send(job):
        calls["count"] += 1
        return True

    monkeypatch.setattr(telegram_bot, "_send_message_sync", fake_send)

    result_first = telegram_bot.send_telegram_message("Hello world", async_mode=False)
    result_second = telegram_bot.send_telegram_message("Hello world", async_mode=False)

    assert result_first is True
    assert result_second is True
    assert calls["count"] == 1
