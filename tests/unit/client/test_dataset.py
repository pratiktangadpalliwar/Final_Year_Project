import pandas as pd
import pytest

from client.app.dataset import assert_dataset_present, load_dataset


def test_assert_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_dataset_present(tmp_path / "nope.csv", min_size_bytes=1)


def test_assert_too_small_raises(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("a")
    with pytest.raises(ValueError):
        assert_dataset_present(p, min_size_bytes=1024)


def test_assert_passes_for_normal_file(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text("x" * 2000)
    assert_dataset_present(p, min_size_bytes=1024)


def test_load_dataset_returns_dataframe(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
    df = load_dataset(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
