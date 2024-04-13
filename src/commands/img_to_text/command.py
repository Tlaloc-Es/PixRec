import logging
from glob import iglob

import click
import cv2

from src.settings import load_config
from src.steps.img_to_text.factory import ImageToTextFactory


@click.command()
@click.option("-c", "--config_path", required=True)
def img_to_text_command(config_path):
    raw_config = load_config(config_path)
    pipeline_config = raw_config["pipeline"]

    input_path = pipeline_config["input_path"]

    steps_config = pipeline_config["steps"]

    img_to_text_config = steps_config.get("img-to-text")
    img_to_text_model = ImageToTextFactory(img_to_text_config.get("selected"))(
        **img_to_text_config.get("params", {})
    )
    logging.info(f"Img to text model: {img_to_text_model}")

    img_to_text(input_path, img_to_text_model)


def img_to_text(input_path, img_to_text_model):
    for image_path in iglob(input_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_description = img_to_text_model.describe(image_path)
        logging.info(f"Image description: {image_description}")
