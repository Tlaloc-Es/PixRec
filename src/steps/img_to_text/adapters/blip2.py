from typing import Union

import cv2
import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from src.steps.img_to_text.adapters.interface import ImageToText


class Blip2(ImageToText):
    def __init__(
        self, *, use_nucleus_sampling: bool = True, num_captions: int = 1, **kwargs
    ):
        self.use_nucleus_sampling = use_nucleus_sampling
        self.num_captions = num_captions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=self.device,
        )

        self.model = model
        self.vis_processors = vis_processors

    def describe(self, image: Union[str, np.array]) -> str:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        torch_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

        return self.model.generate(
            {"image": torch_image},
            use_nucleus_sampling=self.use_nucleus_sampling,
            num_captions=self.num_captions,
        )[0]
