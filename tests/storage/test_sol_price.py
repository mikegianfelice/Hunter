import json
import importlib


def _reload_with_tmp_paths(module, tmp_path, monkeypatch):
    import src.storage.db as db

    db_path = tmp_path / "hunter_state.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    importlib.reload(db)

    monkeypatch.setattr(module, "_SOL_PRICE_JSON", tmp_path / "sol_price_cache.json")
    return importlib.reload(module)


def test_sol_price_cache_roundtrip(tmp_path, monkeypatch):
    import src.storage.sol_price as sol_price

    sol_price = _reload_with_tmp_paths(sol_price, tmp_path, monkeypatch)

    sol_price.save_sol_price_cache(42.5, timestamp=100.0)

    payload = sol_price.load_sol_price_cache()
    assert payload["price"] == 42.5
    assert payload["timestamp"] == 100.0

    snapshot = json.loads((tmp_path / "sol_price_cache.json").read_text())
    assert snapshot["price"] == 42.5
    assert snapshot["timestamp"] == 100.0


def test_sol_price_refresh_from_json(tmp_path, monkeypatch):
    import src.storage.sol_price as sol_price

    sol_price = _reload_with_tmp_paths(sol_price, tmp_path, monkeypatch)

    snapshot_path = tmp_path / "sol_price_cache.json"
    snapshot_path.write_text(json.dumps({"price": 10.0, "timestamp": 123.0}))

    payload = sol_price.load_sol_price_cache()
    assert payload["price"] == 10.0
    assert payload["timestamp"] == 123.0
