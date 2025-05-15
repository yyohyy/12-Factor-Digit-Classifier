import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from model.model import DigitClassifierModel
from utils.preprocess_image import process_digit
from library.logger import get_logger
from library.variable import variables

logger = get_logger(__name__)


class DigitClassifier:
    def __init__(self):
        self.model = DigitClassifierModel()
        model_path = variables.MODEL_PATH
        logger.info(f"Loading model from {model_path}")
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()
        logger.info("Model loaded and set to eval mode")

    def predict_digit(self, image_bytes: bytes) -> int:
        logger.info("Reading and processing input image.")
        pil_image = Image.open(BytesIO(image_bytes)).convert("L")
        img_arr = np.array(pil_image)
        resized = cv2.resize(img_arr, (28, 28))
        padded_img = process_digit(resized)
        inv_img = 255 - padded_img

        tensor_img = torch.tensor(inv_img.reshape(1, 1, 28, 28), dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor_img)
        predicted_digit = torch.argmax(output, dim=1).item()
        logger.info(f"Predicted digit: {predicted_digit}")
        return predicted_digit
