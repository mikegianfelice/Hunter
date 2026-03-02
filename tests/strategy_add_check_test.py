import src.core.strategy as strategy


def test_not_held_adds_disabled_allows_new_entry(monkeypatch):
    monkeypatch.setattr("src.storage.positions.load_positions", lambda: {})
    monkeypatch.setattr("src.utils.position_sync.create_position_key", lambda address: "pos-key")

    token = {"address": "SomeTokenAddress", "priceUsd": 1.0}
    cfg = {"enable_adds_if_held": False, "position_size_multiplier": 1.0}

    allowed, mult, reason = strategy._check_add_to_position_allowed(token, cfg)
    assert allowed is True
    assert reason == "new_position"
    assert mult == 1.0


def test_held_adds_disabled_blocks_add(monkeypatch):
    monkeypatch.setattr(
        "src.storage.positions.load_positions",
        lambda: {"pos-key": {"entry_price": 1.0, "position_size_usd": 100.0}},
    )
    monkeypatch.setattr("src.utils.position_sync.create_position_key", lambda address: "pos-key")

    token = {"address": "SomeTokenAddress", "priceUsd": 1.2}
    cfg = {"enable_adds_if_held": False}

    allowed, mult, reason = strategy._check_add_to_position_allowed(token, cfg)
    assert allowed is False
    assert reason == "adds_disabled"
    assert mult == 0.0


def test_not_held_adds_enabled_allows_new_entry(monkeypatch):
    monkeypatch.setattr("src.storage.positions.load_positions", lambda: {})
    monkeypatch.setattr("src.utils.position_sync.create_position_key", lambda address: "pos-key")

    token = {"address": "SomeTokenAddress", "priceUsd": 1.0}
    cfg = {"enable_adds_if_held": True, "position_size_multiplier": 0.75}

    allowed, mult, reason = strategy._check_add_to_position_allowed(token, cfg)
    assert allowed is True
    assert reason == "new_position"
    assert mult == 0.75
