"""
Dry-run simulation tests for exit gating, token cooldown/blacklist,
and price-health tracking.  No network calls required.

Run:  python -m pytest tests/test_exit_gating.py -v
"""

import time
import pytest


# ---------------------------------------------------------------------------
# 1. Soft-exit profit floor gating
# ---------------------------------------------------------------------------

class TestSoftExitGating:
    """Verify _soft_exit_gated helper blocks or allows exits correctly."""

    def _gate(self, gain, exit_name="test", per_exit_key="test_min_profit", config=None):
        from src.monitoring.monitor_position import _soft_exit_gated
        cfg = config or {"soft_exit_min_profit": 0.05, per_exit_key: 0.05}
        return _soft_exit_gated(gain, exit_name, per_exit_key, cfg)

    def test_below_floor_blocked(self):
        assert self._gate(0.02) is False, "2% gain should be gated when floor is 5%"

    def test_at_floor_allowed(self):
        assert self._gate(0.05) is True, "5% gain should pass when floor is 5%"

    def test_above_floor_allowed(self):
        assert self._gate(0.10) is True, "10% gain should pass"

    def test_negative_gain_blocked(self):
        assert self._gate(-0.01) is False, "negative gain must be gated"

    def test_per_exit_overrides_global(self):
        """Per-exit floor higher than global should take precedence."""
        cfg = {"soft_exit_min_profit": 0.03, "high_min": 0.08}
        assert self._gate(0.05, per_exit_key="high_min", config=cfg) is False

    def test_global_overrides_low_per_exit(self):
        """Global floor higher than per-exit should take precedence."""
        cfg = {"soft_exit_min_profit": 0.07, "low_min": 0.02}
        assert self._gate(0.05, per_exit_key="low_min", config=cfg) is False


# ---------------------------------------------------------------------------
# 2. Token cooldown / blacklist / daily cap
# ---------------------------------------------------------------------------

class TestTokenTradeState:
    """Verify cooldown, blacklist, and daily-cap logic."""

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        """Ensure a clean state for every test."""
        from src.utils import token_trade_state as tts
        tts.reset_all()
        # Override config values for deterministic tests
        monkeypatch.setattr("src.utils.token_trade_state.get_config_float",
                            lambda key, default=0: {
                                "token_cooldown_minutes": 5,  # 5 min for tests
                                "blacklist_hours": 1,
                            }.get(key, default))
        monkeypatch.setattr("src.utils.token_trade_state.get_config_int",
                            lambda key, default=0: {
                                "blacklist_after_consecutive_losses": 2,
                                "max_trades_per_token_per_day": 3,
                            }.get(key, default))
        yield

    def test_fresh_token_allowed(self):
        from src.utils.token_trade_state import is_token_allowed
        allowed, reason = is_token_allowed("MINT123", "solana")
        assert allowed is True
        assert reason == ""

    def test_cooldown_after_close(self):
        from src.utils.token_trade_state import record_trade_close, is_token_allowed
        record_trade_close("MINT123", "solana", pnl_usd=0.50)
        allowed, reason = is_token_allowed("MINT123", "solana")
        assert allowed is False
        assert "cooldown" in reason

    def test_blacklist_after_consecutive_losses(self):
        from src.utils.token_trade_state import record_trade_close, is_token_allowed
        record_trade_close("MINT456", "solana", pnl_usd=-0.10)
        # First loss → cooldown but no blacklist
        state = _get_state("MINT456", "solana")
        assert state["consecutive_losses"] == 1
        assert state["blacklist_until"] == 0

        record_trade_close("MINT456", "solana", pnl_usd=-0.20)
        # Second loss → blacklisted
        state = _get_state("MINT456", "solana")
        assert state["consecutive_losses"] == 2
        assert state["blacklist_until"] > time.time()

        allowed, reason = is_token_allowed("MINT456", "solana")
        assert allowed is False
        assert "blacklisted" in reason

    def test_win_resets_consecutive_losses(self):
        from src.utils.token_trade_state import record_trade_close
        record_trade_close("MINT789", "solana", pnl_usd=-0.10)
        record_trade_close("MINT789", "solana", pnl_usd=0.50)  # win
        state = _get_state("MINT789", "solana")
        assert state["consecutive_losses"] == 0

    def test_daily_cap(self):
        from src.utils.token_trade_state import record_trade_close, is_token_allowed
        for _ in range(3):
            record_trade_close("MINTCAP", "solana", pnl_usd=0.01)
        # After 3 trades, should hit daily cap.
        # Note: also on cooldown, but daily_cap should appear in reason if cooldown expires.
        # Force-expire cooldown for this test by patching state.
        from src.utils.token_trade_state import _get as _tts_get
        rec = _tts_get("MINTCAP", "solana")
        rec["cooldown_until"] = 0  # clear cooldown
        allowed, reason = is_token_allowed("MINTCAP", "solana")
        assert allowed is False
        assert "daily_cap" in reason


def _get_state(token_address, chain_id):
    from src.utils.token_trade_state import get_state
    return get_state(token_address, chain_id)


# ---------------------------------------------------------------------------
# 3. Price health helper
# ---------------------------------------------------------------------------

class TestPriceHealth:
    """Validate that the price health dict logic is internally consistent."""

    def test_valid_price_clears_invalid_since(self):
        """Simulate valid price updates."""
        import math
        health = {"last_valid_price": 0.0, "last_valid_ts": 0.0, "invalid_since": None}

        price = 0.001234
        is_valid = (
            price is not None
            and isinstance(price, (int, float))
            and not math.isnan(price)
            and price > 0
            and abs(price - 0.000001) > 1e-9
        )
        assert is_valid is True
        if is_valid:
            health["last_valid_price"] = price
            health["last_valid_ts"] = time.time()
            health["invalid_since"] = None

        assert health["last_valid_price"] == price
        assert health["invalid_since"] is None

    def test_zero_price_marks_invalid_since(self):
        import math
        health = {"last_valid_price": 0.001, "last_valid_ts": time.time() - 10, "invalid_since": None}

        price = 0
        is_valid = (
            price is not None
            and isinstance(price, (int, float))
            and not math.isnan(price)
            and price > 0
            and abs(price - 0.000001) > 1e-9
        )
        assert is_valid is False
        if not is_valid:
            if health["invalid_since"] is None:
                health["invalid_since"] = time.time()

        assert health["invalid_since"] is not None
        assert health["last_valid_price"] == 0.001  # unchanged

    def test_fallback_price_rejected(self):
        """The sentinel 0.000001 must be treated as invalid."""
        import math
        price = 0.000001
        is_valid = (
            price is not None
            and isinstance(price, (int, float))
            and not math.isnan(price)
            and price > 0
            and abs(price - 0.000001) > 1e-9
        )
        assert is_valid is False
