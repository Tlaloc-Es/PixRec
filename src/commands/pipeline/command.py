import logging

import click

from src.settings import load_config
from src.steps.image_segmentation.factory import ImageSegmentationFactory
from src.steps.img_to_text.factory import ImageToTextFactory


@click.command()
@click.option("-c", "--config_path", required=True)
def pipeline_command(config_path):
    raw_config = load_config(config_path)
    pipeline_config = raw_config["pipeline"]
    steps_config = pipeline_config["steps"]

    image_segmentation_config = steps_config.get("image-segmentation")

    segmentation_model = ImageSegmentationFactory(
        image_segmentation_config.get("selected")
    )(**image_segmentation_config.get("params", {}))
    logging.info(f"Segmentation model: {segmentation_model}")

    img_to_text_config = steps_config.get("img-to-text")
    img_to_text_model = ImageToTextFactory(img_to_text_config.get("selected"))(
        **img_to_text_config.get("params", {})
    )
    logging.info(f"Img to text model: {img_to_text_model}")


def pipeline(segmentation_model, img_to_text_config, input_image_path):
    pass
