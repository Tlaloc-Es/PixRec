from typing import Union

import cv2
import numpy as np
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.downloader import DownloaderGit
from src.steps.img_to_text.adapters.interface import ImageToText


class Blip(ImageToText):
    def __init__(self):
        model_path = DownloaderGit().download(
            "blip", "https://huggingface.co/Salesforce/blip-image-captioning-large"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(
            self.device
        )

    def describe(self, image: Union[str, np.array]) -> str:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
