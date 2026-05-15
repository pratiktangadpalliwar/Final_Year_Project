import pytest

from server.app.round_manager import RoundManager


def test_register_and_active_count():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_01", "Bank One", 100_000)
    rm.register("bank_02", "Bank Two", 200_000)
    assert rm.active_node_count() == 2
    assert "bank_01" in rm.registered


def test_quorum_size():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    for i in range(7):
        rm.register(f"bank_{i:02d}", f"Bank {i}", 100_000)
    # max(3, 7*0.6=4.2 → 5) → 5
    assert rm.quorum_size() == 5


def test_quorum_size_floor_at_min_nodes():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    for i in range(2):
        rm.register(f"bank_{i:02d}", f"Bank {i}", 100_000)
    # max(3, 2*0.6=1.2 → 2) → 3
    assert rm.quorum_size() == 3


def test_flag_decreases_trust_and_eventually_suspends():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_x", "X", 100)
    assert rm.trust_scores["bank_x"] == pytest.approx(1.0)
    for _ in range(5):
        rm.flag_node("bank_x")
    # 1.0 * 0.6^5 ≈ 0.0778 < 0.2 → suspended
    assert "bank_x" in rm.suspended


def test_reward_clamps_at_one():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_x", "X", 100)
    for _ in range(10):
        rm.reward_node("bank_x")
    assert rm.trust_scores["bank_x"] <= 1.0


def test_checkpoint_roundtrip():
    rm = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm.register("bank_01", "Bank One", 100)
    rm.register("bank_02", "Bank Two", 200)
    rm.flag_node("bank_02")
    rm.current_round = 5
    rm.cumulative_eps_global = 12.5
    snapshot = rm.checkpoint_dict()
    rm2 = RoundManager(min_nodes=3, quorum_pct=0.6)
    rm2.restore_from_dict(snapshot)
    assert rm2.current_round == 5
    assert rm2.cumulative_eps_global == 12.5
    assert rm2.trust_scores["bank_02"] == rm.trust_scores["bank_02"]
    assert rm2.registered.keys() == rm.registered.keys()
