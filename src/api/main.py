from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import io

from src.api.inference import Classifier, CLASSES, IMAGE_SIZE
from src.api.schemas import HealthResponse, MetadataResponse, PredictionResponse

app = FastAPI(
    title="EuroSAT Land-Use Classifier",
    description="Classifies Sentinel-2 satellite image patches into 10 land-use classes.",
    version="1.0.0",
)

# Loaded once at startup, not per request
classifier = Classifier()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=classifier is not None)


@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    return MetadataResponse(
        classes=CLASSES,
        input_size=list(IMAGE_SIZE),
        model_format="onnx",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Reject anything that is not a readable image
    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    result = classifier.predict(image)
    return PredictionResponse(**result)