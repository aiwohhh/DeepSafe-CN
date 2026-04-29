from deepsafe_sdk.base import DeepSafeModel
from deepsafe_sdk.types import PredictionResult


class FakeModel(DeepSafeModel):
    def __init__(self, name="fake", model_dir="/tmp"):
        super().__init__(name=name, model_dir=model_dir)
        self.load_count = 0

    def load(self):
        self.load_count += 1
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        return self.make_result(probability=0.75, threshold=threshold)


def test_lazy_loading():
    m = FakeModel()
    assert not m.is_loaded
    result = m.safe_predict("data", 0.5)
    assert m.is_loaded
    assert m.load_count == 1
    assert result.probability == 0.75


def test_make_result():
    m = FakeModel(name="test")
    result = m.make_result(probability=0.8, threshold=0.5)
    assert result.prediction == 1
    assert result.class_name == "fake"
    assert result.model == "test"

    result_real = m.make_result(probability=0.3, threshold=0.5)
    assert result_real.prediction == 0
    assert result_real.class_name == "real"


def test_unload():
    m = FakeModel()
    m.safe_predict("data", 0.5)
    assert m.is_loaded
    m.unload()
    assert not m.is_loaded


def test_double_load_is_noop():
    m = FakeModel()
    m.safe_predict("data", 0.5)
    m.safe_predict("data", 0.5)
    assert m.load_count == 1


def test_weights_path():
    m = FakeModel(name="test", model_dir="/app/model")
    assert m.weights_path("weights/NPR.pth") == "/app/model/weights/NPR.pth"
