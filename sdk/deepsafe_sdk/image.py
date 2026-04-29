import base64
import io
import logging

from PIL import Image, ImageFile

from deepsafe_sdk.base import DeepSafeModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ImageModel(DeepSafeModel):
    def decode_image(self, base64_data: str) -> Image.Image:
        try:
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {e}") from e
