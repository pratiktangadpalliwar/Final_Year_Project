"""Auto-loop: collect → validate → aggregate → DP → eval → persist → broadcast → wait.

restore_state() reads the latest checkpoint + control snapshot from S3 on
boot, so a server pod restart resumes mid-experiment.
"""
from __future__ import annotations

import asyncio
import logging
import time

from server.app.aggregator import Aggregator
from server.app.config import Settings
from server.app.control_plane import ControlPlane
from server.app.dp_engine import DPEngine
from server.app.model import FraudDetectionModel
from server.app.round_manager import RoundManager
from server.app.routers.client import get_pending
from server.app.routers.metrics import push_bank_metrics, push_global_metrics
from server.app.storage import Storage
from server.app.validator import UpdateValidator
from server.app.ws_hub import WsHub

_LOG = logging.getLogger("fl.round_loop")
_CHECKPOINT_KEY = "control/state.json"


def restore_state(*, rm: RoundManager, cp: ControlPlane, storage: Storage, settings: Settings) -> None:
    """Boot: load latest RoundManager checkpoint and ControlPlane snapshot from S3."""
    latest = storage.latest_round(prefix="checkpoints/round_")
    if latest is not None:
        try:
            rm.restore_from_dict(storage.get_json(f"checkpoints/round_{latest:04d}.json"))
            _LOG.info("restored RoundManager from round %d", latest)
        except Exception:
            _LOG.exception("checkpoint restore failed")
            raise
    try:
        cp.restore_from_dict(storage.get_json(_CHECKPOINT_KEY))
        _LOG.info("restored ControlPlane snapshot")
    except Exception:
        _LOG.info("no ControlPlane snapshot found; starting fresh")


def _snapshot_control(cp: ControlPlane, storage: Storage) -> None:
    storage.put_json(_CHECKPOINT_KEY, cp.snapshot_dict())


async def run_one_round(
    *,
    rm: RoundManager,
    cp: ControlPlane,
    storage: Storage,
    hub: WsHub,
    settings: Settings,
    target_round: int,
    deadline_s: int | None = None,
) -> None:
    """Drives a single round end-to-end. Tests call this directly; run_round_loop
    calls it in a while-True."""
    pending = get_pending()
    cp.global_.state = "collecting"
    cp.global_.current_round = target_round
    await hub.broadcast({"type": "round_started", "round": target_round, "quorum_size": rm.quorum_size()})

    deadline = (deadline_s if deadline_s is not None else settings.round_timeout_s)
    end = time.time() + deadline
    while time.time() < end:
        n_received = sum(1 for (r, _) in pending if r == target_round)
        if n_received >= rm.quorum_size():
            break
        await asyncio.sleep(0.5)

    submissions = [v for (r, _), v in list(pending.items()) if r == target_round]
    for (r, bid) in list(pending):
        if r == target_round:
            del pending[(r, bid)]

    if len(submissions) < rm.min_nodes:
        cp.global_.state = "stalled"
        await hub.broadcast({"type": "round_stalled", "round": target_round, "received": len(submissions)})
        _snapshot_control(cp, storage)
        return

    updates = [storage.get_weights(s["weights_key"]) for s in submissions]
    n_samples = [int(s["n_samples"]) for s in submissions]
    bank_ids = [s["bank_id"] for s in submissions]

    validator = UpdateValidator()
    valid, suspicious = validator.score(updates)

    suspicious_set = {id(u) for u in suspicious}
    for u, bid in zip(updates, bank_ids, strict=True):
        if id(u) in suspicious_set:
            rm.flag_node(bid)
            await hub.broadcast({"type": "event", "level": "warn", "msg": f"flagged {bid}"})
        else:
            rm.reward_node(bid)

    cp.global_.state = "aggregating"
    aggregator = Aggregator()
    valid_n = [n for u, n in zip(updates, n_samples, strict=True) if id(u) not in suspicious_set]
    if not valid_n:
        valid, valid_n = updates, n_samples
    suspicious_pct = len(suspicious) / max(1, len(updates))
    aggregated, method = aggregator.aggregate(valid, valid_n, suspicious_pct=suspicious_pct, n_total=len(updates))

    dp = DPEngine(epsilon=settings.dp_epsilon, delta=settings.dp_delta, clip_norm=settings.dp_clip_norm)
    new_global = dp.privatize(aggregated)
    rm.cumulative_eps_global += settings.dp_epsilon
    for bid in bank_ids:
        rm.add_eps(bid, settings.dp_epsilon)

    storage.put_weights(f"models/global_round_{target_round:04d}.pt", new_global)
    rm.current_round = target_round
    storage.put_json(f"checkpoints/round_{target_round:04d}.json", rm.checkpoint_dict())

    for s in submissions:
        push_bank_metrics(s["bank_id"], {"round": target_round, **s["metrics"]})
        await hub.broadcast({"type": "bank_update", "bank_id": s["bank_id"], "round": target_round, "metrics": s["metrics"]})
    push_global_metrics({
        "round": target_round,
        "method": method,
        "n_participants": len(updates),
        "n_suspicious": len(suspicious),
    })
    cp.global_.state = "idle"
    _snapshot_control(cp, storage)
    await hub.broadcast({"type": "round_completed", "round": target_round, "method": method})


async def run_round_loop(
    *,
    rm: RoundManager,
    cp: ControlPlane,
    storage: Storage,
    hub: WsHub,
    settings: Settings,
) -> None:
    """Forever loop. Cancelled on app shutdown."""
    while True:
        if cp.global_.paused:
            await asyncio.sleep(1)
            continue
        if storage.latest_round(prefix="models/global_round_") is None:
            storage.put_weights("models/global_round_0000.pt", FraudDetectionModel().get_weights())
        target = cp.global_.current_round + 1
        try:
            await run_one_round(rm=rm, cp=cp, storage=storage, hub=hub, settings=settings, target_round=target)
        except Exception:
            _LOG.exception("round %d errored", target)
            await asyncio.sleep(5)
            continue
        await asyncio.sleep(cp.global_.inter_round_delay_s)
