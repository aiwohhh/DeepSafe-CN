#!/usr/bin/env python3
"""
CLI tool to register a new detection model into the DeepSafe platform.

Automates:
  1. Adding model/health endpoints to config/deepsafe_config.json
  2. Adding a service entry to docker-compose.yml
  3. Optionally scaffolding a model directory with Dockerfile, app.py, requirements.txt

Usage:
  python scripts/add_model.py --name my_detector --media-type image --port 5008
  python scripts/add_model.py --name my_detector --media-type image --port 5008 --build-dir ./models/image/my_detector
"""

import argparse
import json
import os
import re
import sys
import textwrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "deepsafe_config.json")
COMPOSE_PATH = os.path.join(PROJECT_ROOT, "docker-compose.yml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MEDIA_TYPE_PAYLOAD_KEYS = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data",
}
PORT_RANGES = {"image": "5xxx", "video": "7xxx", "audio": "8xxx"}


def validate_name(name):
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        print(
            f"ERROR: Model name '{name}' must be lowercase snake_case (e.g., my_detector)"
        )
        sys.exit(1)


def get_used_ports(compose_content):
    """Extract all port mappings from docker-compose.yml (including commented-out ones)."""
    return set(int(m) for m in re.findall(r'"(\d+):\d+"', compose_content))


def update_config(name, media_type, port):
    """Add model and health endpoints to deepsafe_config.json."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if media_type not in config.get("media_types", {}):
        config.setdefault("media_types", {})[media_type] = {
            "model_endpoints": {},
            "health_endpoints": {},
            "supported_extensions": [],
        }

    mt = config["media_types"][media_type]

    if name in mt.get("model_endpoints", {}):
        print(
            f"WARNING: Model '{name}' already in config for '{media_type}'. Overwriting."
        )

    mt.setdefault("model_endpoints", {})[name] = f"http://{name}:{port}/predict"
    mt.setdefault("health_endpoints", {})[name] = f"http://{name}:{port}/health"

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"  [OK] Updated {CONFIG_PATH}")


def update_compose(name, media_type, port):
    """Add a service entry to docker-compose.yml."""
    with open(COMPOSE_PATH, "r") as f:
        content = f.read()

    container_name = f"deepsafe-{name.replace('_', '-')}"
    build_path = f"./models/{media_type}/{name}"

    service_block = textwrap.dedent(
        f"""\
  {name}:
    build: {build_path}
    container_name: {container_name}
    ports:
      - "{port}:{port}"
    restart: unless-stopped
    environment:
      - MODEL_PORT={port}
      - PRELOAD_MODEL=false
      - MODEL_TIMEOUT=600
      - USE_GPU=false
    networks:
      - deepsafe-network

"""
    )

    # Insert before top-level "networks:" line
    networks_match = re.search(r"\nnetworks:\s*\n", content)
    if networks_match:
        insert_pos = networks_match.start()
        content = content[:insert_pos] + "\n" + service_block + content[insert_pos:]
    else:
        print("WARNING: Could not find 'networks:' section. Appending service at end.")
        content += "\n" + service_block

    # Add to api depends_on
    depends_pattern = re.compile(
        r"(  api:\s*\n(?:.*\n)*?    depends_on:\s*\n(?:      - \w+\s*\n)*)",
        re.MULTILINE,
    )
    match = depends_pattern.search(content)
    if match:
        insert_point = match.end()
        content = content[:insert_point] + f"      - {name}\n" + content[insert_point:]
    else:
        print("WARNING: Could not find api depends_on section. Add manually.")

    with open(COMPOSE_PATH, "w") as f:
        f.write(content)

    print(f"  [OK] Updated {COMPOSE_PATH}")


def scaffold_model(name, media_type, port):
    """Generate a model directory with Dockerfile, app.py, requirements.txt."""
    model_dir = os.path.join(MODELS_DIR, media_type, name)
    if os.path.exists(model_dir):
        print(f"  [SKIP] Directory already exists: {model_dir}")
        return

    os.makedirs(model_dir, exist_ok=True)

    payload_key = MEDIA_TYPE_PAYLOAD_KEYS[media_type]

    # requirements.txt
    with open(os.path.join(model_dir, "requirements.txt"), "w") as f:
        f.write(
            "fastapi>=0.100.0\nuvicorn[standard]>=0.23.0\npydantic>=2.0.0\npillow\nrequests\n"
        )

    # Dockerfile
    with open(os.path.join(model_dir, "Dockerfile"), "w") as f:
        f.write(
            textwrap.dedent(
                f"""\
            FROM python:3.9-slim

            WORKDIR /app

            RUN apt-get update && \\
                apt-get install -y --no-install-recommends git wget && \\
                apt-get clean && rm -rf /var/lib/apt/lists/*

            COPY requirements.txt .
            RUN pip install --no-cache-dir --upgrade pip && \\
                pip install --no-cache-dir -r requirements.txt

            # TODO: Add model-specific dependencies (e.g., torch) and download weights here

            COPY app.py .

            ENV MODEL_PORT={port}
            ENV USE_GPU=false
            ENV PRELOAD_MODEL=false
            ENV MODEL_TIMEOUT=600

            EXPOSE ${{MODEL_PORT}}
            CMD ["python", "app.py"]
        """
            )
        )

    # app.py
    with open(os.path.join(model_dir, "app.py"), "w") as f:
        f.write(
            textwrap.dedent(
                f'''\
            """
            {name} Model Service
            TODO: Implement your model loading and inference logic.
            """

            import os
            import base64
            import time
            import logging
            import sys
            from typing import Dict, Any, Optional
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel, Field
            import uvicorn

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
            logger = logging.getLogger(__name__)

            MODEL_NAME = "{name}"
            MODEL_PORT = int(os.environ.get("MODEL_PORT", {port}))

            app = FastAPI(title=f"{{MODEL_NAME}} Service", version="1.0.0")
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"], allow_credentials=True,
                allow_methods=["*"], allow_headers=["*"],
            )

            # Global model variable
            model = None


            class MediaInput(BaseModel):
                {payload_key}: str = Field(..., description="Base64 encoded {media_type} data")
                threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)


            def load_model():
                """TODO: Load your model weights here."""
                global model
                logger.info(f"Loading {{MODEL_NAME}} model...")
                # model = YourModel.load("weights/model.pth")
                model = "placeholder"
                logger.info(f"{{MODEL_NAME}} model loaded.")


            @app.on_event("startup")
            async def startup():
                if os.environ.get("PRELOAD_MODEL", "false").lower() == "true":
                    load_model()


            @app.get("/health")
            async def health():
                return {{
                    "status": "healthy" if model is not None else "not_loaded",
                    "model_name": MODEL_NAME,
                    "model_loaded": model is not None,
                }}


            @app.post("/predict", response_model=Dict[str, Any])
            async def predict(input_data: MediaInput):
                global model
                if model is None:
                    load_model()
                if model is None:
                    raise HTTPException(status_code=503, detail="Model failed to load")

                start_time = time.time()
                media_bytes = base64.b64decode(input_data.{payload_key})

                # TODO: Implement your inference logic here
                # probability = your_model_predict(media_bytes)
                probability = 0.5  # placeholder

                prediction = 1 if probability >= input_data.threshold else 0
                inference_time = time.time() - start_time

                return {{
                    "model": MODEL_NAME,
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "class": "fake" if prediction == 1 else "real",
                    "inference_time": float(inference_time),
                }}


            if __name__ == "__main__":
                uvicorn.run("app:app", host="0.0.0.0", port=MODEL_PORT, reload=False)
        '''
            )
        )

    print(f"  [OK] Scaffolded model at {model_dir}")
    print(f"       -> Edit app.py to implement your inference logic")
    print(f"       -> Edit Dockerfile to install dependencies and download weights")


def main():
    parser = argparse.ArgumentParser(
        description="Add a new model to the DeepSafe platform"
    )
    parser.add_argument("--name", required=True, help="Model name (snake_case)")
    parser.add_argument(
        "--media-type", required=True, choices=["image", "video", "audio"]
    )
    parser.add_argument(
        "--port", required=True, type=int, help="Port number for the model service"
    )
    parser.add_argument(
        "--no-scaffold", action="store_true", help="Skip Dockerfile/app.py generation"
    )
    args = parser.parse_args()

    validate_name(args.name)

    # Check port conflicts
    with open(COMPOSE_PATH, "r") as f:
        compose_content = f.read()
    used_ports = get_used_ports(compose_content)
    if args.port in used_ports:
        print(f"ERROR: Port {args.port} is already in use in docker-compose.yml")
        sys.exit(1)

    print(f"\nAdding model '{args.name}' ({args.media_type}) on port {args.port}...\n")

    # 1. Update config
    update_config(args.name, args.media_type, args.port)

    # 2. Update docker-compose.yml
    update_compose(args.name, args.media_type, args.port)

    # 3. Scaffold model directory
    if not args.no_scaffold:
        scaffold_model(args.name, args.media_type, args.port)

    print(f"\nDone! Next steps:")
    print(
        f"  1. Implement inference logic in models/{args.media_type}/{args.name}/app.py"
    )
    print(
        f"  2. Add dependencies to models/{args.media_type}/{args.name}/requirements.txt"
    )
    print(f"  3. Update the Dockerfile with weight downloads")
    print(f"  4. Run: make start")
    print(f"  5. Retrain ensemble: make retrain MEDIA_TYPE={args.media_type}")


if __name__ == "__main__":
    main()
