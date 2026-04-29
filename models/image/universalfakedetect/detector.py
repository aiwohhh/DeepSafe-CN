import os
import sys
import torch
from torchvision import transforms
from deepsafe_sdk import ImageModel, PredictionResult

current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, "universalfakedetect")
if model_code_path not in sys.path:
    sys.path.append(model_code_path)

from models import get_model


class UniversalFakeDetector(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def load(self):
        weights_path = self.weights_path(
            "universalfakedetect/pretrained_weights/fc_weights.pth"
        )
        net = get_model("CLIP:ViT-L/14")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        net.fc.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = self.model(tensor).sigmoid().flatten().item()
        return self.make_result(probability=probability, threshold=threshold)
