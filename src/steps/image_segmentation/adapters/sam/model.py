from typing import Any, List

import cv2
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from src.downloader import Downloader
from src.steps.image_segmentation.adapters.interface import ImageSegmentation
from src.steps.image_segmentation.adapters.sam.tools import show_anns


class SAM(ImageSegmentation):
    models = {
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "hash": "4b8939a88964f0f4ff5f5b2642c598a6",
        },
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "hash": "01ec64d29a2fca3f0661936605ae66f8",
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "hash": "0b3195507c641ddb6910d2bb5adee89c",
        },
    }

    def __init__(
        self,
        *,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        model_selected: str = "vit_h",
        **kwargs,
    ):
        self.model_selected = model_selected
        self._download_model()
        self.model = sam_model_registry[self.model_selected](checkpoint=self.checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    def segment(self, image_path: str) -> List[Any]:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        return masks

    def masks_to_image(self, masks, image):
        return show_anns(masks, image)

    def _download_model(self):
        url = self.models[self.model_selected]["url"]
        hash = self.models[self.model_selected]["hash"]
        self.checkpoint = Downloader().download(
            "sam",
            self.model_selected,
            hash,
            url,
        )
