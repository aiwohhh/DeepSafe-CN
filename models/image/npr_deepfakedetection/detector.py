import os
import sys
import torch
import torchvision.transforms as transforms
from deepsafe_sdk import ImageModel, PredictionResult

# Add model code to path for resnet50 import
MODEL_REPO_SUBDIR = "npr_deepfakedetection"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, MODEL_REPO_SUBDIR)
if model_code_path not in sys.path:
    sys.path.insert(0, model_code_path)

from networks.resnet import resnet50


class NPRDetector(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load(self):
        weights_path = self.weights_path("npr_deepfakedetection/weights/NPR.pth")
        net = resnet50(num_classes=1)
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()
        return self.make_result(probability=probability, threshold=threshold)
