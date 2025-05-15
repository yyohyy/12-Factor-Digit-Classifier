import numpy as np
from utils.preprocess_image import process_digit


def test_process_digit():
    test_image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)

    processed = process_digit(test_image)

    assert processed.shape == (28, 28)
    assert processed.dtype == np.uint8
