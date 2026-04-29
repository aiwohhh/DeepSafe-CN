import hashlib
import logging
import os
import urllib.request
from typing import List

from deepsafe_sdk.manifest import WeightEntry

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(weights: List[WeightEntry], model_dir: str) -> None:
    for entry in weights:
        if not entry.url:
            logger.info(
                f"Weight entry '{entry.path}' has no URL. Skipping download (assumed present from Docker build)."
            )
            continue

        filepath = os.path.join(model_dir, entry.path)

        if os.path.exists(filepath):
            if entry.sha256:
                actual_sha = compute_sha256(filepath)
                if actual_sha == entry.sha256:
                    logger.info(
                        f"Weight file '{entry.path}' exists with correct checksum. Skipping."
                    )
                    continue
                else:
                    logger.warning(
                        f"Weight file '{entry.path}' checksum mismatch "
                        f"(expected {entry.sha256}, got {actual_sha}). Re-downloading."
                    )
            else:
                logger.info(
                    f"Weight file '{entry.path}' exists (no checksum to verify). Skipping."
                )
                continue

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logger.info(f"Downloading weight file '{entry.path}' from {entry.url}...")
        urllib.request.urlretrieve(entry.url, filepath)

        if entry.sha256:
            actual_sha = compute_sha256(filepath)
            if actual_sha != entry.sha256:
                os.remove(filepath)
                raise RuntimeError(
                    f"Checksum mismatch for '{entry.path}': expected {entry.sha256}, got {actual_sha}"
                )

        logger.info(f"Weight file '{entry.path}' ready.")
