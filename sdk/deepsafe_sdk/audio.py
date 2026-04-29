import base64
import io
import logging
import wave
from typing import Tuple

import numpy as np

from deepsafe_sdk.base import DeepSafeModel

logger = logging.getLogger(__name__)


class AudioModel(DeepSafeModel):
    def decode_audio(self, base64_data: str) -> Tuple[np.ndarray, int]:
        try:
            audio_bytes = base64.b64decode(base64_data)
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as w:
                sample_rate = w.getframerate()
                n_frames = w.getnframes()
                raw = w.readframes(n_frames)
                sample_width = w.getsampwidth()

            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.uint8

            waveform = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            max_val = float(np.iinfo(dtype).max)
            waveform = waveform / max_val
            return waveform, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {e}") from e
