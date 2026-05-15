from unittest.mock import MagicMock

import pytest
import torch

from client.app.round_runner import RoundRunner


def _make_weights():
    from client.app.model import FraudDetectionModel
    return FraudDetectionModel().get_weights()


def test_runner_skips_when_no_new_round():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 0, "dataset_version": 1,
        "fault": "none", "weights_key": "models/global_round_0000.pt",
    }
    storage = MagicMock()
    trainer = MagicMock()
    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=storage, trainer=trainer,
        dataset_loader=lambda: (torch.randn(10, 19), torch.zeros(10)),
        last_round_seen=0,
    )
    r.tick()
    trainer.train.assert_not_called()
    server.post_update.assert_not_called()


def test_runner_trains_and_uploads_on_new_round():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 1, "dataset_version": 1,
        "fault": "none", "weights_key": "models/global_round_0000.pt",
    }
    server.get_global.return_value = {
        "round": 0, "weights_key": "models/global_round_0000.pt",
        "weights_url": "https://signed",
    }
    storage = MagicMock()
    storage.get_weights_from_url.return_value = _make_weights()
    storage.put_weights = MagicMock()
    trainer = MagicMock()
    trainer.train.return_value = MagicMock(weights=_make_weights(), metrics={"val_auc": 0.8})

    # need at least 2 samples for stratify, and enough positives. Use 32 samples with mix.
    X = torch.randn(32, 19)
    y = torch.cat([torch.zeros(28), torch.ones(4)])

    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=storage, trainer=trainer,
        dataset_loader=lambda: (X, y),
        last_round_seen=0,
    )
    r.tick()
    trainer.train.assert_called_once()
    storage.put_weights.assert_called_once()
    server.post_update.assert_called_once()
    args, _ = server.post_update.call_args
    assert args[0]["bank_id"] == "bank_01"
    assert args[0]["round"] == 1
    assert args[0]["weights_key"].startswith("updates/bank_01/round_0001")


def test_runner_crashes_on_crash_fault():
    server = MagicMock()
    server.get_control.return_value = {
        "paused": False, "current_round": 1, "dataset_version": 1,
        "fault": "crash", "weights_key": "models/global_round_0000.pt",
    }
    r = RoundRunner(
        bank_id="bank_01",
        server=server, storage=MagicMock(), trainer=MagicMock(),
        dataset_loader=lambda: (torch.randn(10, 19), torch.zeros(10)),
        last_round_seen=0,
    )
    with pytest.raises(SystemExit):
        r.tick()
