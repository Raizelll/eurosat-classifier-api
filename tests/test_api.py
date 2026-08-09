import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def make_image(size=(64, 64), color=(0, 128, 0)):
    """Build a fake image in memory, so tests need no data files."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200

    body = response.json()
    assert len(body["classes"]) == 10
    assert body["input_size"] == [64, 64]


def test_predict_returns_valid_response():
    response = client.post(
        "/predict", files={"file": ("test.jpg", make_image(), "image/jpeg")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in body["probabilities"]
    assert 0.0 <= body["confidence"] <= 1.0
    # Probabilities must sum to 1
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-5) == 1.0


def test_predict_rejects_non_image():
    response = client.post(
        "/predict", files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400


def test_predict_accepts_other_sizes():
    """A user may upload a 200x200 image: preprocessing must resize it."""
    response = client.post(
        "/predict", files={"file": ("big.jpg", make_image(size=(200, 200)), "image/jpeg")}
    )
    assert response.status_code == 200


def test_predict_requires_file():
    response = client.post("/predict")
    assert response.status_code == 422