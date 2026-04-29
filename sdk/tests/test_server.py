import os
import base64
import io
import yaml
from PIL import Image
from fastapi.testclient import TestClient
from deepsafe_sdk.server import create_app
from deepsafe_sdk.manifest import load_manifest


def _make_manifest_dir(tmp_path):
    manifest_data = {
        "name": "stub_detector",
        "version": "1.0.0",
        "media_type": "image",
        "model_class": "stub.StubDetector",
        "port": 5099,
    }
    manifest_path = os.path.join(str(tmp_path), "model.yaml")
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_data, f)

    stub_path = os.path.join(str(tmp_path), "stub.py")
    with open(stub_path, "w") as f:
        f.write(
            "from deepsafe_sdk.image import ImageModel\n"
            "from deepsafe_sdk.types import PredictionResult\n"
            "class StubDetector(ImageModel):\n"
            "    def load(self):\n"
            "        self.model = 'loaded'\n"
            "    def predict(self, input_data, threshold):\n"
            "        return self.make_result(probability=0.7, threshold=threshold)\n"
        )
    return manifest_path


def test_create_app_info_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "stub_detector"
    assert data["media_type"] == "image"


def test_create_app_health_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_name"] == "stub_detector"


def test_create_app_predict_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)

    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.post("/predict", json={"image_data": b64, "threshold": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "stub_detector"
    assert data["probability"] == 0.7
    assert data["prediction"] == 1
    assert data["class"] == "fake"
    assert "inference_time" in data


def test_create_app_unload_endpoint(tmp_path):
    manifest_path = _make_manifest_dir(tmp_path)
    manifest = load_manifest(manifest_path)
    app = create_app(manifest, str(tmp_path))
    client = TestClient(app)

    resp = client.post("/unload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("not_loaded", "unloaded")
