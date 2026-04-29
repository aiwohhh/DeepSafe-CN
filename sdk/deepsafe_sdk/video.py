import base64
import logging
import os
import tempfile
from typing import List

import cv2
import numpy as np

from deepsafe_sdk.base import DeepSafeModel

logger = logging.getLogger(__name__)


class VideoModel(DeepSafeModel):
    frames_per_video: int = 15

    def extract_frames(self, base64_data: str, num_frames: int = 0) -> List[np.ndarray]:
        if num_frames <= 0:
            num_frames = self.frames_per_video

        video_bytes = base64.b64decode(base64_data)
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"deepsafe_{os.urandom(8).hex()}.mp4"
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {tmp_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return []

            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            return frames
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
