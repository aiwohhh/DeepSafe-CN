import gc
import logging
import os
import threading
import time
from abc import ABC, abstractmethod

from deepsafe_sdk.types import PredictionResult

logger = logging.getLogger(__name__)


class DeepSafeModel(ABC):
    def __init__(self, name: str = "", model_dir: str = ""):
        self._model_name = name
        self._model_dir = model_dir
        self._model = None
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Set self.model."""

    @abstractmethod
    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        """Run inference on base64-encoded input data."""

    def unload(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info(f"Unloading model '{self._model_name}'.")
                del self._model
                self._model = None
                gc.collect()

    def safe_predict(self, input_data: str, threshold: float) -> PredictionResult:
        self._ensure_loaded()
        start = time.time()
        result = self.predict(input_data, threshold)
        if result.inference_time == 0:
            result.inference_time = time.time() - start
        return result

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            self._last_used = time.time()
            return
        with self._lock:
            if self._model is not None:
                self._last_used = time.time()
                return
            logger.info(f"Loading model '{self._model_name}'...")
            self.load()
            self._last_used = time.time()
            logger.info(f"Model '{self._model_name}' loaded.")

    def check_idle_unload(self, timeout: int) -> None:
        if self._model is None:
            return
        if time.time() - self._last_used > timeout:
            self.unload()

    def weights_path(self, relative_path: str) -> str:
        return os.path.join(self._model_dir, relative_path)

    def make_result(self, probability: float, threshold: float) -> PredictionResult:
        prediction = 1 if probability >= threshold else 0
        return PredictionResult(
            model=self._model_name,
            probability=probability,
            prediction=prediction,
            class_name="fake" if prediction == 1 else "real",
            inference_time=0.0,
        )
