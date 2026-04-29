import base64
import io
import pytest
from PIL import Image
from deepsafe_sdk.image import ImageModel
from deepsafe_sdk.types import PredictionResult


class FakeImageModel(ImageModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        return self.make_result(probability=0.9, threshold=threshold)


def _make_test_image_b64():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_image():
    m = FakeImageModel(name="test", model_dir="/tmp")
    b64 = _make_test_image_b64()
    image = m.decode_image(b64)
    assert isinstance(image, Image.Image)
    assert image.size == (100, 100)
    assert image.mode == "RGB"


def test_decode_image_invalid():
    m = FakeImageModel(name="test", model_dir="/tmp")
    with pytest.raises(ValueError, match="decode"):
        m.decode_image("not_valid_base64!!!")


def test_predict_with_image():
    m = FakeImageModel(name="test", model_dir="/tmp")
    b64 = _make_test_image_b64()
    result = m.safe_predict(b64, 0.5)
    assert result.prediction == 1
    assert result.class_name == "fake"
