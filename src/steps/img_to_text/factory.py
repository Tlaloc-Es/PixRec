from src.steps.img_to_text.adapters import Blip
from src.steps.img_to_text.adapters.interface import ImageToText


def ImageToTextFactory(image_to_text: str) -> ImageToText:
    return {"Blip": Blip}[image_to_text]
