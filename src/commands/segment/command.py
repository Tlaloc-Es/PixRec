import logging
import os
from glob import iglob

import click
import cv2

from src.settings import load_config
from src.steps.image_segmentation.factory import ImageSegmentationFactory


@click.command()
@click.option("-c", "--config_path", required=True)
def segment_command(config_path):
    raw_config = load_config(config_path)
    pipeline_config = raw_config["pipeline"]

    output_path = pipeline_config["output_path"]
    input_path = pipeline_config["input_path"]

    steps_config = pipeline_config["steps"]

    image_segmentation_config = steps_config.get("image-segmentation")

    segmentation_model = ImageSegmentationFactory(
        image_segmentation_config.get("selected")
    )(**image_segmentation_config.get("params", {}))
    logging.info(f"Segmentation model: {segmentation_model}")

    segment(input_path, output_path, segmentation_model)


def segment(input_path, output_path, segmentation_model):
    for image_path in iglob(input_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = segmentation_model.segment(image_path)
        images = segmentation_model.masks_to_image(masks, image)

        for i, img in enumerate(images):
            filename = os.path.basename(image_path)
            output_folder_path = os.path.join(output_path, filename)
            output_file_path = os.path.join(output_folder_path, f"{i}.jpg")
            os.makedirs(output_folder_path, exist_ok=True)
            print(output_file_path)
            cv2.imwrite(output_file_path, img)
