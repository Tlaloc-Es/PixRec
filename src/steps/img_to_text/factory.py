from src.steps.img_to_text.adapters import Blip, Blip2
from src.steps.img_to_text.adapters.interface import ImageToText


def ImageToTextFactory(image_to_text: str) -> ImageToText:
    return {"Blip": Blip, "Blip2": Blip2}[image_to_text]
