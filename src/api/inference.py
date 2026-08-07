import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Must match the values used during training, otherwise the model receives a
# distribution it was never trained on and accuracy silently collapses.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = (64, 64)


def preprocess(image: Image.Image) -> np.ndarray:
    # Uploaded images may be grayscale or RGBA. Force 3 channels.
    image = image.convert("RGB").resize(IMAGE_SIZE)

    # Scale 0-255 to 0-1, then normalize. Same recipe as build_transform().
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD

    # PIL gives height-width-channel. The model expects channel-height-width.
    array = array.transpose(2, 0, 1)

    # Add the batch dimension: the model always expects a batch.
    return array[np.newaxis, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    # Subtracting the max prevents overflow on large logits. Standard trick.
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class Classifier:
    # Loading the ONNX session is slow, so it happens once when the object is
    # created, not on every request.
    def __init__(self, model_path: str = "artifacts/model.onnx"):
        self.session = ort.InferenceSession(model_path)

    def predict(self, image: Image.Image) -> dict:
        tensor = preprocess(image)
        logits = self.session.run(["logits"], {"input": tensor})[0][0]
        probs = softmax(logits)

        best = int(probs.argmax())
        return {
            "predicted_class": CLASSES[best],
            "confidence": float(probs[best]),
            "probabilities": {
                name: float(p) for name, p in zip(CLASSES, probs)
            },
        }


if __name__ == "__main__":
    import sys

    clf = Classifier()
    result = clf.predict(Image.open(sys.argv[1]))
    print(result)