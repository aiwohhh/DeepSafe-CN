import base64
import io
import struct
import wave
import pytest
from deepsafe_sdk.audio import AudioModel
from deepsafe_sdk.types import PredictionResult


class FakeAudioModel(AudioModel):
    def load(self):
        self.model = "loaded"

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        waveform, sr = self.decode_audio(input_data)
        return self.make_result(probability=0.4, threshold=threshold)


def _make_wav_b64(duration_s=0.1, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        num_samples = int(duration_s * sample_rate)
        samples = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        w.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_audio():
    m = FakeAudioModel(name="test", model_dir="/tmp")
    b64 = _make_wav_b64()
    waveform, sr = m.decode_audio(b64)
    assert sr == 16000
    assert len(waveform) > 0


def test_decode_audio_invalid():
    m = FakeAudioModel(name="test", model_dir="/tmp")
    with pytest.raises(ValueError, match="decode"):
        m.decode_audio("not_valid_audio!!!")
