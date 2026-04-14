"""
server/app.py
-------------
FL Aggregation Server — REST API

Endpoints:
  POST /register          - Bank node registers itself
  GET  /model/global      - Download current global model weights
  POST /model/update      - Submit local model update
  GET  /round/status      - Get current round status
  GET  /health            - Health check
  GET  /metrics           - Training metrics
  POST /admin/rollback    - Manual rollback trigger

Datasets are loaded locally on each bank node via kubectl cp or volume mounts.
No upload endpoint needed — data never leaves the bank node.
"""

import os
import re
import json
import logging
import threading
from datetime import datetime
from flask import Flask, request, jsonify

from round_manager import RoundManager
from aggregator   import FedAvgAggregator
from dp_engine    import DifferentialPrivacy
from storage      import S3Storage, DynamoDBState
from utils        import setup_logging, validate_update

app    = Flask(__name__)
logger = setup_logging("fl-server")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# ── CORS (required for browser-based dashboard) ───────────────────
@app.after_request
def _add_cors(response):
    origin = os.environ.get("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Origin"]  = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Admin-Token"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def _options_handler(path):
    return jsonify({}), 200


def _sanitize_key(key):
    """Return key if safe for S3/local storage, else None."""
    if not isinstance(key, str) or not key:
        return None
    if ".." in key or key.startswith("/"):
        return None
    if not re.match(r'^[a-zA-Z0-9/_\-\.]+$', key):
        return None
    return key


# ── Globals (initialised in main) ─────────────────────────────────
round_manager : RoundManager   = None
aggregator    : FedAvgAggregator = None
storage       : S3Storage      = None
db_state      : DynamoDBState  = None


# ══════════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════════
@app.route("/register", methods=["POST"])
def register():
    """Bank node registers with the FL server."""
    data     = request.get_json()
    bank_id  = data.get("bank_id")
    bank_name= data.get("bank_name", bank_id)
    n_samples= data.get("n_samples", 0)

    if not bank_id:
        return jsonify({"error": "bank_id required"}), 400

    result = round_manager.register_node(
        bank_id   = bank_id,
        bank_name = bank_name,
        n_samples = n_samples,
    )

    logger.info(f"Node registered: {bank_id} ({n_samples} samples)")
    return jsonify(result), 200


# ══════════════════════════════════════════════════════════════════
# MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════
@app.route("/model/global", methods=["GET"])
def get_global_model():
    """Return current global model weights + round info."""
    bank_id = request.args.get("bank_id")
    if not bank_id:
        return jsonify({"error": "bank_id required"}), 400

    if not round_manager.is_registered(bank_id):
        return jsonify({"error": "Node not registered"}), 403

    weights_key = storage.get_latest_model_key()
    if not weights_key:
        return jsonify({"error": "No global model available yet"}), 404

    weights_url = storage.get_presigned_url(weights_key)

    return jsonify({
        "round"       : round_manager.current_round,
        "weights_url" : weights_url,
        "weights_key" : weights_key,
        "model_version": round_manager.model_version,
        "min_nodes"   : round_manager.min_nodes_per_round,
    }), 200


# ══════════════════════════════════════════════════════════════════
# MODEL UPDATE SUBMISSION
# ══════════════════════════════════════════════════════════════════
@app.route("/model/update", methods=["POST"])
def submit_update():
    """
    Bank submits its local model update.

    Expects JSON:
    {
        "bank_id"      : "bank_01_retail_urban",
        "round"        : 5,
        "weights_key"  : "updates/bank_01_round_005.pt",
        "n_samples"    : 400000,
        "local_metrics": {"loss": 0.12, "accuracy": 0.94}
    }
    """
    data       = request.get_json()
    bank_id    = data.get("bank_id")
    round_num  = data.get("round")
    weights_key= data.get("weights_key")
    n_samples  = data.get("n_samples", 0)
    metrics    = data.get("local_metrics", {})

    # ── Validation ────────────────────────────────────────────────
    if not all([bank_id, round_num is not None, weights_key]):
        return jsonify({"error": "bank_id, round, weights_key required"}), 400

    weights_key = _sanitize_key(weights_key)
    if not weights_key:
        return jsonify({"error": "invalid weights_key"}), 400

    if not round_manager.is_registered(bank_id):
        return jsonify({"error": "Node not registered"}), 403

    if round_num != round_manager.current_round:
        return jsonify({
            "error"        : "Stale round",
            "submitted"    : round_num,
            "current_round": round_manager.current_round,
        }), 409

    if round_manager.is_suspended(bank_id):
        return jsonify({"error": f"Node {bank_id} is suspended"}), 403

    # ── Validate update integrity ─────────────────────────────────
    validation = validate_update(
        bank_id     = bank_id,
        weights_key = weights_key,
        storage     = storage,
        round_manager = round_manager,
    )
    if not validation["valid"]:
        round_manager.flag_node(bank_id, reason=validation["reason"])
        logger.warning(f"Update rejected from {bank_id}: {validation['reason']}")
        return jsonify({"error": validation["reason"], "flagged": True}), 422

    # ── Accept update ─────────────────────────────────────────────
    accepted = round_manager.accept_update(
        bank_id     = bank_id,
        weights_key = weights_key,
        n_samples   = n_samples,
        metrics     = metrics,
    )

    logger.info(f"Update accepted from {bank_id} | Round {round_num} | "
                f"{round_manager.updates_received}/{round_manager.active_nodes} nodes")

    # ── Trigger aggregation if quorum reached ─────────────────────
    if round_manager.quorum_reached():
        thread = threading.Thread(target=_run_aggregation, daemon=True)
        thread.start()

    return jsonify({
        "accepted"         : True,
        "round"            : round_num,
        "updates_received" : round_manager.updates_received,
        "nodes_total"      : round_manager.active_nodes,
        "quorum_reached"   : round_manager.quorum_reached(),
    }), 200


# ══════════════════════════════════════════════════════════════════
# ROUND STATUS
# ══════════════════════════════════════════════════════════════════
@app.route("/round/status", methods=["GET"])
def round_status():
    """Return current FL round status."""
    return jsonify(round_manager.get_status()), 200


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════
@app.route("/metrics", methods=["GET"])
def metrics():
    """Return training metrics history."""
    return jsonify({
        "current_round"  : round_manager.current_round,
        "model_version"  : round_manager.model_version,
        "metrics_history": round_manager.metrics_history,
        "node_trust"     : round_manager.trust_scores,
        "suspended_nodes": list(round_manager.suspended_nodes),
    }), 200


# ══════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"       : "healthy",
        "round"        : round_manager.current_round,
        "active_nodes" : round_manager.active_nodes,
        "timestamp"    : datetime.utcnow().isoformat(),
    }), 200


# ══════════════════════════════════════════════════════════════════
# ADMIN — ROLLBACK
# ══════════════════════════════════════════════════════════════════
@app.route("/admin/rollback", methods=["POST"])
def admin_rollback():
    """Manually trigger rollback to previous checkpoint."""
    token = request.headers.get("X-Admin-Token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    data          = request.get_json() or {}
    target_round  = data.get("target_round")

    success = round_manager.rollback(target_round=target_round)
    if success:
        logger.warning(f"Manual rollback triggered to round {target_round}")
        return jsonify({"rolled_back": True, "round": round_manager.current_round}), 200
    return jsonify({"error": "Rollback failed"}), 500


# ══════════════════════════════════════════════════════════════════
# AGGREGATION (background thread)
# ══════════════════════════════════════════════════════════════════
def _run_aggregation():
    """Run FedAvg aggregation in background when quorum reached."""
    try:
        logger.info(f"Starting aggregation for round {round_manager.current_round}")

        # Collect accepted updates
        updates = round_manager.get_pending_updates()

        # Run robust aggregation
        new_weights, agg_info = aggregator.aggregate(updates)

        if new_weights is None:
            logger.error("Aggregation failed — no valid updates")
            return

        # Validate new model (anti-corruption check)
        valid, reason = round_manager.validate_new_model(new_weights)
        if not valid:
            logger.warning(f"New model failed validation: {reason} — rolling back")
            round_manager.rollback()
            return

        # Save checkpoint BEFORE applying
        storage.save_checkpoint(
            weights    = new_weights,
            round_num  = round_manager.current_round,
            agg_info   = agg_info,
        )

        # Advance to next round
        round_manager.advance_round(
            new_weights = new_weights,
            agg_info    = agg_info,
        )

        # Persist round state to DynamoDB
        db_state.save_round_state(round_manager.get_status())

        logger.info(f"Round {round_manager.current_round - 1} complete → "
                    f"Round {round_manager.current_round} started")

    except Exception as e:
        logger.error(f"Aggregation error: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════
def create_app():
    global round_manager, aggregator, storage, db_state

    # Config from environment
    cfg = {
        "s3_bucket"        : os.environ.get("S3_BUCKET",         "fl-models"),
        "dynamodb_table"   : os.environ.get("DYNAMODB_TABLE",    "fl-rounds"),
        "min_nodes"        : int(os.environ.get("MIN_NODES",      "3")),
        "max_rounds"       : int(os.environ.get("MAX_ROUNDS",     "50")),
        "quorum_pct"       : float(os.environ.get("QUORUM_PCT",   "0.6")),
        "rollback_threshold": float(os.environ.get("ROLLBACK_THRESHOLD", "0.05")),
        "dp_epsilon"       : float(os.environ.get("DP_EPSILON",   "5.0")),
        "dp_delta"         : float(os.environ.get("DP_DELTA",     "1e-5")),
        "dp_clip_norm"     : float(os.environ.get("DP_CLIP_NORM", "0.5")),
        "input_dim"        : int(os.environ.get("INPUT_DIM",      "19")),
        "hidden_dims"      : [64, 32, 16],
        "aws_region"       : os.environ.get("AWS_REGION",        "us-east-1"),
    }

    logger.info("Initialising FL Server components...")

    storage      = S3Storage(cfg)
    db_state     = DynamoDBState(cfg)
    dp           = DifferentialPrivacy(cfg["dp_epsilon"], cfg["dp_delta"], cfg["dp_clip_norm"])
    aggregator   = FedAvgAggregator(dp=dp, cfg=cfg)
    round_manager= RoundManager(cfg=cfg, storage=storage, aggregator=aggregator)
    round_manager.set_aggregation_callback(_run_aggregation)

    round_manager.initialise()
    logger.info(f"FL Server ready | Min nodes: {cfg['min_nodes']} | "
                f"Max rounds: {cfg['max_rounds']}")

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(
        host  = "0.0.0.0",
        port  = int(os.environ.get("PORT", "8080")),
        debug = False,
    )
