import json
import importlib

import pytest


def _reload_with_tmp_paths(module, tmp_path, monkeypatch):
    import src.storage.db as db

    db_path = tmp_path / "hunter_state.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    importlib.reload(db)

    monkeypatch.setattr(module, "_PRICE_MEMORY_JSON", tmp_path / "price_memory.json")
    return importlib.reload(module)


def test_price_memory_roundtrip(tmp_path, monkeypatch):
    import src.storage.price_memory as price_memory

    price_memory = _reload_with_tmp_paths(price_memory, tmp_path, monkeypatch)

    data = {"0xABCDEF": {"ts": 123, "price": 1.23}}
    price_memory.save_price_memory(data)

    stored = price_memory.load_price_memory()
    assert stored == {"0xabcdef": {"ts": 123, "price": 1.23}}

    snapshot = json.loads((tmp_path / "price_memory.json").read_text())
    assert snapshot == {"0xabcdef": {"ts": 123, "price": 1.23}}


def test_price_memory_refresh_from_json(tmp_path, monkeypatch):
    import src.storage.price_memory as price_memory

    price_memory = _reload_with_tmp_paths(price_memory, tmp_path, monkeypatch)

    snapshot_path = tmp_path / "price_memory.json"
    snapshot_path.write_text(json.dumps({"0xabc": {"ts": 50, "price": 2.5}}))

    reloaded = price_memory.load_price_memory()
    assert reloaded == {"0xabc": {"ts": 50, "price": 2.5}}
