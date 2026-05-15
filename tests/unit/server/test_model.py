import hashlib
from pathlib import Path

import torch

from server.app.model import INPUT_DIM, FraudDetectionModel


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_server_and_client_model_files_are_byte_identical():
    """Hand-shared file: any divergence will silently break weight transfer."""
    root = Path(__file__).resolve().parents[3]
    server_hash = _sha256(root / "server" / "app" / "model.py")
    client_hash = _sha256(root / "client" / "app" / "model.py")
    assert server_hash == client_hash, (
        "server/app/model.py and client/app/model.py have diverged. "
        "Edit BOTH identically."
    )


def test_model_input_output_shape():
    m = FraudDetectionModel()
    m.eval()
    x = torch.randn(8, INPUT_DIM)
    y = m(x)
    assert y.shape == (8, 1)


def test_get_set_weights_roundtrip():
    m1, m2 = FraudDetectionModel(), FraudDetectionModel()
    w = m1.get_weights()
    m2.set_weights(w)
    m1.eval(); m2.eval()
    x = torch.randn(4, INPUT_DIM)
    assert torch.allclose(m1(x), m2(x), atol=1e-6)
