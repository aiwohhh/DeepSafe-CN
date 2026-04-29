from deepsafe_sdk.types import PredictionResult


def test_prediction_result_fields():
    result = PredictionResult(
        model="test_model",
        probability=0.85,
        prediction=1,
        class_name="fake",
        inference_time=0.123,
    )
    assert result.model == "test_model"
    assert result.probability == 0.85
    assert result.prediction == 1
    assert result.class_name == "fake"
    assert result.inference_time == 0.123


def test_prediction_result_serialization():
    result = PredictionResult(
        model="test_model",
        probability=0.3,
        prediction=0,
        class_name="real",
        inference_time=0.05,
    )
    d = result.model_dump()
    assert d["model"] == "test_model"
    assert d["probability"] == 0.3
    assert d["prediction"] == 0
    assert d["class_name"] == "real"
    d_alias = result.model_dump(by_alias=True)
    assert d_alias["class"] == "real"
