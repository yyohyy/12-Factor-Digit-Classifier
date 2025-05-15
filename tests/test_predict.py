import torch
import numpy as np
from model.model import DigitClassifierModel


def test_digit_classifier():
    model = DigitClassifierModel()
    model.eval()

    dummy_input = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 10)
    assert torch.all(torch.isfinite(output))
