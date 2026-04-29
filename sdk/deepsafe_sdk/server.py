import importlib
import logging
import os
import sys
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deepsafe_sdk.base import DeepSafeModel
from deepsafe_sdk.manifest import ModelManifest, load_manifest
from deepsafe_sdk.weights import ensure_weights

logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    image_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class VideoInput(BaseModel):
    video_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


class AudioInput(BaseModel):
    audio_data: str
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


INPUT_MODELS = {
    "image": ImageInput,
    "video": VideoInput,
    "audio": AudioInput,
}

DATA_FIELD = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data",
}


def _import_model_class(model_class_path: str, model_dir: str) -> type:
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_app(manifest: ModelManifest, model_dir: str) -> FastAPI:
    model_cls = _import_model_class(manifest.model_class, model_dir)
    instance: DeepSafeModel = model_cls(name=manifest.name, model_dir=model_dir)

    preload = (
        manifest.environment.get(
            "PRELOAD_MODEL", os.environ.get("PRELOAD_MODEL", "false")
        ).lower()
        == "true"
    )
    model_timeout = int(
        manifest.environment.get(
            "MODEL_TIMEOUT", os.environ.get("MODEL_TIMEOUT", "600")
        )
    )

    InputModel = INPUT_MODELS[manifest.media_type]
    data_field = DATA_FIELD[manifest.media_type]

    app = FastAPI(
        title=f"DeepSafe Model: {manifest.name}",
        version=manifest.version,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def info():
        return {
            "name": manifest.name,
            "version": manifest.version,
            "media_type": manifest.media_type,
            "model_loaded": instance.is_loaded,
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_name": manifest.name,
            "model_loaded": instance.is_loaded,
        }

    @app.post("/predict")
    async def predict(input_data: InputModel):
        try:
            media_data = getattr(input_data, data_field)
            result = instance.safe_predict(media_data, input_data.threshold)
            return result.model_dump(by_alias=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/unload")
    async def unload():
        if not instance.is_loaded:
            return {
                "status": "not_loaded",
                "message": f"{manifest.name} is not loaded.",
            }
        instance.unload()
        return {"status": "unloaded", "message": f"{manifest.name} unloaded."}

    @app.on_event("startup")
    async def startup():
        if preload:
            logger.info(f"Preloading model '{manifest.name}'...")
            try:
                instance._ensure_loaded()
            except Exception as e:
                logger.error(f"Preload failed: {e}", exc_info=True)

        if model_timeout > 0:

            def idle_check():
                instance.check_idle_unload(model_timeout)
                if instance.is_loaded:
                    t = threading.Timer(model_timeout / 2.0, idle_check)
                    t.daemon = True
                    t.start()

            t = threading.Timer(model_timeout / 2.0, idle_check)
            t.daemon = True
            t.start()

    return app


def cli():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(prog="deepsafe")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--manifest", required=True, help="Path to model.yaml")

    args = parser.parse_args()

    if args.command == "serve":
        manifest_path = os.path.abspath(args.manifest)
        model_dir = os.path.dirname(manifest_path)
        manifest = load_manifest(manifest_path)

        logger.info(f"Starting DeepSafe model server: {manifest.name}")
        ensure_weights(manifest.weights, model_dir)

        app = create_app(manifest, model_dir)
        port = int(os.environ.get("MODEL_PORT", str(manifest.port)))
        uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
    else:
        parser.print_help()
