import numpy as np


def show_anns(anns, reference_image):
    images = []
    if len(anns) == 0:
        return images

    for ann in anns:
        img_shape = ann["segmentation"].shape[:2]
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

        m = ann["segmentation"]
        color_mask = reference_image[m]
        img[m] = color_mask

        images.append(img)

    return images
