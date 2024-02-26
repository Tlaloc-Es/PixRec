import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from src.steps.image_embeddings.adapters.interface import EmbeddingCalculator


class Resnet50EmbeddingCalculator(EmbeddingCalculator):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]  # TODO research about these values
    )

    def __init__(self):
        self._download_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def calculate_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(self.device)
        with torch.no_grad():
            output = self.model(input_batch)
        return output[0].cpu().numpy()

    def _download_model(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        features = nn.Sequential(*list(model.children())[:-1])
        self.model = nn.Sequential(features, nn.Flatten())
