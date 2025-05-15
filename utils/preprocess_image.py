import numpy as np
import cv2
from library.logger import get_logger

logger = get_logger(__name__)


def process_digit(arr: np.ndarray) -> np.ndarray:
    logger.info("Resizing and padding the image.")
    height = 28
    new_width = int((height * arr.shape[1]) / arr.shape[0])
    resized_img = cv2.resize(arr, (new_width, height))

    pad_width = 28 - resized_img.shape[1]
    left_padding = pad_width // 2
    right_padding = pad_width - left_padding

    if pad_width > 0:
        padded_img = np.pad(
            resized_img,
            ((0, 0), (left_padding, right_padding)),
            mode="constant",
            constant_values=255,
        )
    else:
        padded_img = resized_img

    return padded_img
