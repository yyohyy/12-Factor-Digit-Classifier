from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from model.predict import DigitClassifier
from library.logger import get_logger

app = FastAPI()
model = DigitClassifier()
logger = get_logger(__name__)


@app.post("/predict_digit")
async def predict(image: UploadFile = File(...)):
    try:
        logger.info(f"Prediction digit for file: {image.filename}")
        contents = await image.read()
        digit = model.predict_digit(contents)
        logger.info(f"Prediction result: {digit}")
        return {"Prediction": digit}
    except Exception as e:
        logger.info(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
