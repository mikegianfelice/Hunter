import json
import importlib
import sys


def _reload_sol_price(tmp_path, monkeypatch):
    import src.storage.db as db

    db_path = tmp_path / "hunter_state.db"
    monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
    importlib.reload(db)

    module_name = "src.storage.sol_price"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_SOL_PRICE_JSON", tmp_path / "sol_price_cache.json", raising=False)
    return module


def test_sol_price_cache_roundtrip(tmp_path, monkeypatch):
    sol_price = _reload_sol_price(tmp_path, monkeypatch)

    sol_price.save_sol_price_cache(42.5, timestamp=100.0)

    payload = sol_price.load_sol_price_cache()
    assert payload["price"] == 42.5
    assert payload["timestamp"] == 100.0

    snapshot_path = tmp_path / "sol_price_cache.json"
    assert snapshot_path.exists()
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["price"] == 42.5
    assert snapshot["timestamp"] == 100.0


def test_sol_price_refresh_from_json(tmp_path, monkeypatch):
    sol_price = _reload_sol_price(tmp_path, monkeypatch)

    snapshot_path = tmp_path / "sol_price_cache.json"
    snapshot_path.write_text(json.dumps({"price": 10.0, "timestamp": 123.0}))

    payload = sol_price.load_sol_price_cache()
    assert payload["price"] == 10.0
    assert payload["timestamp"] == 123.0
