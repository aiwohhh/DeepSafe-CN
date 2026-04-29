from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    model: str
    probability: float = Field(ge=0.0, le=1.0)
    prediction: int = Field(ge=0, le=1)
    class_name: str = Field(serialization_alias="class")
    inference_time: float
