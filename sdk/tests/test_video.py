import base64
import os
import tempfile
import numpy as np
import pytest
from deepsafe_sdk.video import VideoModel
from deepsafe_sdk.types import PredictionResult


class FakeVideoModel(VideoModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        return self.make_result(probability=0.6, threshold=threshold)


def test_extract_frames_returns_list():
    m = FakeVideoModel(name="test", model_dir="/tmp")
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 25.0, (64, 64))
        for _ in range(30):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        frames = m.extract_frames(video_b64, num_frames=5)
        assert isinstance(frames, list)
        assert len(frames) == 5
        assert frames[0].shape[2] == 3  # RGB
    finally:
        os.unlink(tmp_path)


def test_frames_per_video_default():
    m = FakeVideoModel(name="test", model_dir="/tmp")
    assert m.frames_per_video == 15
