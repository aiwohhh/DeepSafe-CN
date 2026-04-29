import os
import pytest
import yaml
from deepsafe_sdk.manifest import load_manifest


def _write_manifest(tmp_path, data):
    path = os.path.join(tmp_path, "model.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_load_valid_manifest(tmp_path):
    data = {
        "name": "test_model",
        "version": "1.0.0",
        "media_type": "image",
        "model_class": "detector.TestDetector",
        "port": 5008,
        "weights": [
            {
                "url": "https://example.com/model.pth",
                "path": "weights/model.pth",
                "sha256": "abc123",
            }
        ],
        "environment": {"PRELOAD_MODEL": "false", "MODEL_TIMEOUT": "600"},
        "dependencies": ["timm==0.9.2"],
    }
    path = _write_manifest(str(tmp_path), data)
    manifest = load_manifest(path)
    assert manifest.name == "test_model"
    assert manifest.media_type == "image"
    assert manifest.port == 5008
    assert len(manifest.weights) == 1
    assert manifest.weights[0].sha256 == "abc123"


def test_load_minimal_manifest(tmp_path):
    data = {
        "name": "minimal_model",
        "media_type": "video",
        "model_class": "detector.Detector",
        "port": 7002,
    }
    path = _write_manifest(str(tmp_path), data)
    manifest = load_manifest(path)
    assert manifest.name == "minimal_model"
    assert manifest.weights == []
    assert manifest.environment == {}
    assert manifest.dependencies == []


def test_invalid_media_type(tmp_path):
    data = {
        "name": "bad",
        "media_type": "pdf",
        "model_class": "d.D",
        "port": 5000,
    }
    path = _write_manifest(str(tmp_path), data)
    with pytest.raises(ValueError, match="media_type"):
        load_manifest(path)


def test_missing_required_field(tmp_path):
    data = {"name": "incomplete"}
    path = _write_manifest(str(tmp_path), data)
    with pytest.raises((ValueError, KeyError)):
        load_manifest(path)
