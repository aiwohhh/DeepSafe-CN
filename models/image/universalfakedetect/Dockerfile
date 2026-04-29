FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir --upgrade pip "setuptools<70" wheel

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY models/image/universalfakedetect/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/image/universalfakedetect/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git universalfakedetect && \
    mkdir -p universalfakedetect/pretrained_weights

# Download CLIP model weights (HuggingFace mirror with original source fallback)
RUN python -c "import torch; import os; os.makedirs('/root/.cache/clip', exist_ok=True)" && \
    (wget -O /root/.cache/clip/ViT-L-14.pt \
    https://huggingface.co/siddharthksah/DeepSafe-weights/resolve/main/universalfakedetect/ViT-L-14.pt \
    || wget -O /root/.cache/clip/ViT-L-14.pt \
    https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

# Download the FC weights (HuggingFace mirror with original source fallback)
RUN (wget -O universalfakedetect/pretrained_weights/fc_weights.pth \
    https://huggingface.co/siddharthksah/DeepSafe-weights/resolve/main/universalfakedetect/fc_weights.pth \
    || wget -O universalfakedetect/pretrained_weights/fc_weights.pth \
    https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth)

ENV MODEL_PORT=5004
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

EXPOSE 5004

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
