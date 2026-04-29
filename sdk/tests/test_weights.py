import os
import hashlib
import pytest
from unittest.mock import patch
from deepsafe_sdk.manifest import WeightEntry
from deepsafe_sdk.weights import ensure_weights, compute_sha256


def test_compute_sha256(tmp_path):
    filepath = os.path.join(str(tmp_path), "test.bin")
    content = b"test content for hashing"
    with open(filepath, "wb") as f:
        f.write(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_sha256(filepath) == expected


def test_ensure_weights_skips_existing(tmp_path):
    content = b"fake model weights"
    sha = hashlib.sha256(content).hexdigest()
    filepath = os.path.join(str(tmp_path), "model.pth")
    with open(filepath, "wb") as f:
        f.write(content)
    entry = WeightEntry(
        url="https://example.com/model.pth", path="model.pth", sha256=sha
    )
    with patch("deepsafe_sdk.weights.urllib.request.urlretrieve") as mock_dl:
        ensure_weights([entry], str(tmp_path))
        mock_dl.assert_not_called()


def test_ensure_weights_downloads_missing(tmp_path):
    content = b"downloaded weights"
    sha = hashlib.sha256(content).hexdigest()
    entry = WeightEntry(
        url="https://example.com/model.pth", path="model.pth", sha256=sha
    )

    def fake_download(url, dest):
        with open(dest, "wb") as f:
            f.write(content)

    with patch(
        "deepsafe_sdk.weights.urllib.request.urlretrieve", side_effect=fake_download
    ):
        ensure_weights([entry], str(tmp_path))
    filepath = os.path.join(str(tmp_path), "model.pth")
    assert os.path.exists(filepath)


def test_ensure_weights_checksum_mismatch(tmp_path):
    entry = WeightEntry(
        url="https://example.com/model.pth", path="model.pth", sha256="wrong_hash"
    )

    def fake_download(url, dest):
        with open(dest, "wb") as f:
            f.write(b"some data")

    with patch(
        "deepsafe_sdk.weights.urllib.request.urlretrieve", side_effect=fake_download
    ):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            ensure_weights([entry], str(tmp_path))


def test_ensure_weights_skips_empty_url(tmp_path):
    entry = WeightEntry(url="", path="weights/model.pth", sha256="")
    with patch("deepsafe_sdk.weights.urllib.request.urlretrieve") as mock_dl:
        ensure_weights([entry], str(tmp_path))
        mock_dl.assert_not_called()
