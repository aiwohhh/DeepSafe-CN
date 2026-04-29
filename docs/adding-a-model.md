# Adding a New Model to DeepSafe

DeepSafe uses a shared SDK so you only write the model-specific logic. Everything else — the HTTP server, health checks, lazy loading, thread safety — is handled for you.

## Quick Start

A complete model contribution is 3 files + 2 config entries:

```
models/image/my_model/
├── model.yaml      # What your model is
├── detector.py     # How your model works
├── Dockerfile      # How to build it
└── requirements.txt # Extra pip dependencies
```

## Step 1: Create model.yaml

```yaml
name: my_model
version: "1.0.0"
media_type: image          # image | video | audio
model_class: detector.MyDetector
port: 5020

weights:
  - url: https://example.com/weights.pth
    path: weights/model.pth
    sha256: abc123def456...

environment:
  PRELOAD_MODEL: "false"
  MODEL_TIMEOUT: "600"
  USE_GPU: "false"

dependencies:
  - timm==0.9.2
```

Fields reference:
| Field | Required | Description |
|-------|----------|-------------|
| name | Yes | Unique model identifier |
| media_type | Yes | image, video, or audio |
| model_class | Yes | Python import path to your class (relative to model dir) |
| port | Yes | Unique port number |
| version | No | Semver string (default: 0.0.0) |
| weights | No | List of weight files to download |
| environment | No | Default environment variables |
| dependencies | No | Extra pip packages |

## Step 2: Write detector.py

Extend the base class for your media type:

### Image Model

```python
import torch
from deepsafe_sdk import ImageModel, PredictionResult

class MyDetector(ImageModel):
    def load(self):
        self.model = torch.load(self.weights_path("weights/model.pth"))
        self.model.eval()

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)      # base64 -> PIL Image (RGB)
        tensor = self.preprocess(image)              # your preprocessing
        with torch.no_grad():
            prob = self.model(tensor).sigmoid().item()
        return self.make_result(probability=prob, threshold=threshold)
```

### Video Model

```python
import numpy as np
from deepsafe_sdk import VideoModel, PredictionResult

class MyVideoDetector(VideoModel):
    def load(self):
        self.model = load_my_model(self.weights_path("weights/model.pth"))

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        frames = self.extract_frames(input_data)    # base64 -> list of RGB numpy arrays
        scores = [self.score_frame(f) for f in frames]
        prob = float(np.mean(scores))
        return self.make_result(probability=prob, threshold=threshold)
```

### Audio Model

```python
from deepsafe_sdk import AudioModel, PredictionResult

class MyAudioDetector(AudioModel):
    def load(self):
        self.model = load_my_model(self.weights_path("weights/model.pth"))

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        waveform, sr = self.decode_audio(input_data)  # base64 -> (numpy array, sample_rate)
        prob = self.run_inference(waveform, sr)
        return self.make_result(probability=prob, threshold=threshold)
```

What the base classes give you:
- self.decode_image(b64) — base64 to PIL Image (ImageModel)
- self.extract_frames(b64, num_frames=15) — base64 video to frame list (VideoModel)
- self.decode_audio(b64) — base64 WAV to numpy array (AudioModel)
- self.weights_path(relative) — resolve weight file path
- self.make_result(probability, threshold) — build standard response with auto timing
- Lazy loading, thread safety, timeout unloading — all automatic

## Step 3: Create Dockerfile

Copy this template and change the paths and port:

```dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY models/image/my_model/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/image/my_model/ /app/model/
WORKDIR /app/model

EXPOSE 5020
CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
```

## Step 4: Register in Docker Compose

Add to docker-compose.yml:

```yaml
  my_model:
    build:
      context: .
      dockerfile: models/image/my_model/Dockerfile
    ports:
      - "5020:5020"
    environment:
      - PRELOAD_MODEL=false
      - MODEL_TIMEOUT=600
      - USE_GPU=false
    networks:
      - deepsafe-network
    restart: unless-stopped
```

## Step 5: Register in Config

Add to config/deepsafe_config.json under the matching media type:

```json
{
  "model_endpoints": {
    "my_model": "http://my_model:5020/predict"
  },
  "health_endpoints": {
    "my_model": "http://my_model:5020/health"
  }
}
```

## Verify

```bash
docker compose up -d --build my_model
curl http://localhost:5020/health
```

Your model is now part of the DeepSafe ensemble.

## Automated Setup (Steps 4-5)

Instead of manually editing docker-compose.yml and config, use:

```bash
make add-model NAME=my_model MEDIA_TYPE=image PORT=5020
```

This auto-generates the config and compose entries. Then:

```bash
make start       # Build and start all services
make health      # Verify your model is running
make retrain MEDIA_TYPE=image  # Retrain ensemble with the new model
```

## Ensemble Retraining

After adding a model, retrain the stacking ensemble to include it:

```bash
# Full pipeline: feature generation + training
make retrain MEDIA_TYPE=image

# With custom dataset
make retrain MEDIA_TYPE=image DATASET_DIR=./my_dataset/images

# Use Optuna optimizer
make retrain MEDIA_TYPE=image OPTIMIZER=optuna TRIALS=100
```

## Checklist

- [ ] model.yaml with name, media_type, model_class, port
- [ ] detector.py extending ImageModel/VideoModel/AudioModel
- [ ] Dockerfile using SDK + deepsafe serve
- [ ] requirements.txt with model-specific dependencies
- [ ] Entry in docker-compose.yml (auto: `make add-model`)
- [ ] Entry in config/deepsafe_config.json (auto: `make add-model`)
- [ ] Ensemble retrained (auto: `make retrain`)
