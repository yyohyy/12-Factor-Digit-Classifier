from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import numpy as np
import io

client = TestClient(app)


def test_predict_digit_endpoint():
    image = Image.fromarray(np.uint8(np.ones((28, 28)) * 255))
    image.putpixel((14, 14), 0)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict_digit", files={"image": ("digit.png", buf, "image/png")}
    )

    assert response.status_code == 200
    assert "Prediction" in response.json()
    assert isinstance(response.json()["Prediction"], int)
