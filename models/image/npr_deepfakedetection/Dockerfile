FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY sdk/ /app/sdk/
RUN pip install --no-cache-dir /app/sdk/

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY models/image/npr_deepfakedetection/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY models/image/npr_deepfakedetection/ /app/model/

WORKDIR /app/model

RUN git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git npr_deepfakedetection && \
    mkdir -p npr_deepfakedetection/weights && \
    (wget -O npr_deepfakedetection/weights/NPR.pth \
    https://huggingface.co/siddharthksah/DeepSafe-weights/resolve/main/npr_deepfakedetection/NPR.pth \
    || wget -O npr_deepfakedetection/weights/NPR.pth \
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth)

ENV MODEL_PORT=5001
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

EXPOSE 5001

CMD ["deepsafe", "serve", "--manifest", "model.yaml"]
