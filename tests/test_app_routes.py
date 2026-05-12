import io
from PIL import Image


def test_homepage_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"ALAsKA" in response.data


def test_predict_without_image_returns_400(client):
    response = client.post("/predict")
    data = response.get_json()

    assert response.status_code == 400
    assert data["error"] == "No image uploaded"


def test_predict_with_invalid_file_returns_400(client):
    fake_file = io.BytesIO(b"this is not an image")

    response = client.post(
        "/predict",
        data={"image": (fake_file, "fake.txt")},
        content_type="multipart/form-data",
    )

    data = response.get_json()

    assert response.status_code == 400
    assert data["error"] == "Could not read image"


def test_predict_with_valid_image_returns_prediction(client):
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/predict",
        data={"image": (img_bytes, "test.jpg")},
        content_type="multipart/form-data",
    )

    data = response.get_json()

    assert response.status_code == 200
    assert "prediction" in data
    assert "confidence" in data
    assert "top5" in data
    assert isinstance(data["top5"], list)
    assert len(data["top5"]) == 5