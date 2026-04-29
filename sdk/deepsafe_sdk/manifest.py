from dataclasses import dataclass, field
from typing import List, Dict
import yaml

VALID_MEDIA_TYPES = {"image", "video", "audio"}


@dataclass
class WeightEntry:
    url: str
    path: str
    sha256: str


@dataclass
class ModelManifest:
    name: str
    media_type: str
    model_class: str
    port: int
    version: str = "0.0.0"
    weights: List[WeightEntry] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


def load_manifest(path: str) -> ModelManifest:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    for required in ("name", "media_type", "model_class", "port"):
        if required not in data:
            raise ValueError(f"Missing required field: {required}")

    if data["media_type"] not in VALID_MEDIA_TYPES:
        raise ValueError(
            f"Invalid media_type '{data['media_type']}'. "
            f"Must be one of: {VALID_MEDIA_TYPES}"
        )

    weights = [
        WeightEntry(url=w["url"], path=w["path"], sha256=w["sha256"])
        for w in data.get("weights", [])
    ]

    return ModelManifest(
        name=data["name"],
        version=data.get("version", "0.0.0"),
        media_type=data["media_type"],
        model_class=data["model_class"],
        port=data["port"],
        weights=weights,
        environment=data.get("environment", {}),
        dependencies=data.get("dependencies", []),
    )
