import os
import sys
import gc
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from albumentations import Compose, PadIfNeeded

from deepsafe_sdk import VideoModel, PredictionResult

app_dir = os.path.dirname(os.path.abspath(__file__))
for subpath in [
    "model_code/cross-efficient-vit",
    "model_code/efficient-vit",
    "model_code/preprocessing",
    "model_code/cross-efficient-vit/efficient_net",
]:
    abs_path = os.path.join(app_dir, subpath)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

from cross_efficient_vit import CrossEfficientViT
from efficient_vit import EfficientViT
from facenet_pytorch import MTCNN
from transforms.albu import IsotropicResize

IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
FACE_THRESHOLDS = [0.7, 0.8, 0.8]
MTCNN_MIN_FACE_SIZE = 40


class CrossEfficientViTDetector(VideoModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frames_per_video = int(os.environ.get("FRAMES_PER_VIDEO", "15"))
        use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.variant = os.environ.get("DEFAULT_MODEL_VARIANT", "cross_efficient_vit")
        self.face_detector = None
        self.face_transform = None
        self.config = None

    def load(self):
        model_paths = {
            "cross_efficient_vit": self.weights_path(
                "model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth"
            ),
            "efficient_vit": self.weights_path(
                "model_code/efficient-vit/pretrained_models/efficient_vit.pth"
            ),
        }
        config_paths = {
            "cross_efficient_vit": self.weights_path(
                "model_code/cross-efficient-vit/configs/architecture.yaml"
            ),
            "efficient_vit": self.weights_path(
                "model_code/efficient-vit/configs/architecture.yaml"
            ),
        }

        model_path = model_paths[self.variant]
        config_path = config_paths[self.variant]

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        image_size = self.config["model"]["image-size"]

        if self.variant == "cross_efficient_vit":
            net = CrossEfficientViT(config=self.config)
        else:
            channels = 1280
            if self.config["model"].get("selected_efficient_net", 0) == 7:
                channels = 2560
            net = EfficientViT(
                config=self.config,
                channels=channels,
                selected_efficient_net=self.config["model"].get(
                    "selected_efficient_net", 0
                ),
            )

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=True)
        net.to(self.device)
        net.eval()
        self.model = net

        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            thresholds=FACE_THRESHOLDS,
            min_face_size=MTCNN_MIN_FACE_SIZE,
            select_largest=False,
        )
        self.face_transform = Compose(
            [
                IsotropicResize(
                    max_side=image_size,
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_CUBIC,
                ),
                PadIfNeeded(
                    min_height=image_size,
                    min_width=image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ]
        )

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        frames = self.extract_frames(input_data, num_frames=self.frames_per_video)
        if not frames:
            return self.make_result(probability=0.5, threshold=threshold)

        all_scores = []
        for frame in frames:
            boxes, probs, landmarks = self.face_detector.detect(frame, landmarks=True)
            if boxes is None:
                continue
            for box, prob, _ in zip(boxes, probs, landmarks):
                if prob < FACE_THRESHOLDS[2]:
                    continue
                xmin, ymin, xmax, ymax = [int(b) for b in box]
                w_face, h_face = xmax - xmin, ymax - ymin
                pad_h, pad_w = h_face // 3, w_face // 3
                crop_xmin = max(0, xmin - pad_w)
                crop_ymin = max(0, ymin - pad_h)
                crop_xmax = min(frame.shape[1], xmax + pad_w)
                crop_ymax = min(frame.shape[0], ymax + pad_h)
                face_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                if face_crop.size == 0:
                    continue

                transformed = self.face_transform(image=face_crop)["image"]
                tensor = (
                    torch.from_numpy(transformed.astype(np.float32)).permute(2, 0, 1)
                    / 255.0
                )
                tensor = IMAGENET_NORMALIZE(tensor).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    score = torch.sigmoid(logits).squeeze().item()
                all_scores.append(score)

        if not all_scores:
            return self.make_result(probability=0.5, threshold=threshold)

        probability = float(np.mean(all_scores))
        return self.make_result(probability=probability, threshold=threshold)

    def unload(self):
        super().unload()
        self.face_detector = None
        self.face_transform = None
        self.config = None
        gc.collect()
