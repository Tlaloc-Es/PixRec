from src.steps.image_segmentation.adapters import SAM
from src.steps.image_segmentation.adapters.interface import ImageSegmentation


def ImageSegmentationFactory(image_segmentation: str) -> ImageSegmentation:
    return {"SAM": SAM}[image_segmentation]
