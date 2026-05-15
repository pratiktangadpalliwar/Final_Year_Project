from server.app.control_plane import ControlPlane


def test_initial_state():
    cp = ControlPlane()
    assert cp.global_.paused is False
    assert cp.global_.current_round == 0
    assert cp.global_.state == "idle"
    assert cp.banks == {}


def test_pause_resume():
    cp = ControlPlane()
    cp.pause()
    assert cp.global_.paused is True
    cp.resume()
    assert cp.global_.paused is False


def test_set_fault_creates_bank_entry_if_missing():
    cp = ControlPlane()
    cp.set_fault("bank_04", "byzantine")
    assert cp.banks["bank_04"].fault == "byzantine"
    assert cp.banks["bank_04"].dataset_version == 1


def test_bump_dataset_version_increments():
    cp = ControlPlane()
    cp.bump_dataset_version("bank_03")
    cp.bump_dataset_version("bank_03")
    assert cp.banks["bank_03"].dataset_version == 3  # starts at 1, +2 bumps


def test_reset_rounds_clears_state():
    cp = ControlPlane()
    cp.global_.current_round = 42
    cp.global_.state = "collecting"
    cp.reset_rounds()
    assert cp.global_.current_round == 0
    assert cp.global_.state == "idle"


def test_snapshot_and_restore_roundtrip():
    cp = ControlPlane()
    cp.set_fault("bank_01", "straggle")
    cp.bump_dataset_version("bank_01")
    cp.global_.current_round = 7
    snap = cp.snapshot_dict()

    cp2 = ControlPlane()
    cp2.restore_from_dict(snap)
    assert cp2.global_.current_round == 7
    assert cp2.banks["bank_01"].fault == "straggle"
    assert cp2.banks["bank_01"].dataset_version == 2
