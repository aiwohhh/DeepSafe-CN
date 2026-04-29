FROM python:3.9-slim

WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g; s|http://security.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake \
    libgl1-mesa-dev ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-0 gstreamer1.0-plugins-base \
    libjpeg-dev libpng-dev libtiff-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_NONFREE=0
ENV QT_QPA_PLATFORM=offscreen

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir opencv-python-headless

COPY models/video/cross_efficient_vit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/video/cross_efficient_vit/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git model_code

# Set up weight directories
RUN mkdir -p /app/model/model_code/efficient-vit/pretrained_models && \
    mkdir -p /app/model/model_code/cross-efficient-vit/pretrained_models

# Download model weights (HuggingFace mirror with Google Drive fallback)
RUN (wget -O /app/model/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth \
    https://huggingface.co/siddharthksah/DeepSafe-weights/resolve/main/cross_efficient_vit/cross_efficient_vit.pth && \
    wget -O /app/model/model_code/efficient-vit/pretrained_models/efficient_vit.pth \
    https://huggingface.co/siddharthksah/DeepSafe-weights/resolve/main/cross_efficient_vit/efficient_vit.pth) \
    || (echo "HuggingFace download failed, falling back to Google Drive..." && \
    mkdir -p /app/model/model_code/gdrive_weights && \
    for i in 1 2 3; do \
        gdown --folder https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1 -O /app/model/model_code/gdrive_weights --remaining-ok && break; \
        echo "gdown attempt $i failed, retrying in 10s..."; \
        sleep 10; \
    done && \
    [ -f /app/model/model_code/gdrive_weights/cross_efficient_vit.pth ] && \
        mv /app/model/model_code/gdrive_weights/cross_efficient_vit.pth /app/model/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth; \
    [ -f /app/model/model_code/gdrive_weights/efficient_vit.pth ] && \
        mv /app/model/model_code/gdrive_weights/efficient_vit.pth /app/model/model_code/efficient-vit/pretrained_models/efficient_vit.pth; \
    rm -rf /app/model/model_code/gdrive_weights)

ENV PYTHONPATH=/app/model/model_code/cross-efficient-vit:/app/model/model_code/efficient-vit:/app/model/model_code/preprocessing:/app/model/model_code/cross-efficient-vit/efficient_net:${PYTHONPATH}

ENV MODEL_PORT=7001
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV USE_GPU="false"
ENV DEFAULT_MODEL_VARIANT="cross_efficient_vit"
ENV FRAMES_PER_VIDEO="15"

EXPOSE 7001

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
