from flask import Flask, request, jsonify, send_from_directory
import torch, json
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import io, os

app = Flask(__name__, static_folder=".")

# ── Load model once at startup ──────────────────────────────────────────
IMG_SIZE = 128
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("asl_classes.json") as f:
    classes = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("asl_resnet18.pt", map_location=device))
model = model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "recognize.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top5 = torch.topk(probs, 5)

    results = [
        {"letter": classes[idx], "confidence": round(float(p), 4)}
        for idx, p in zip(top5.indices.tolist(), top5.values.tolist())
    ]

    return jsonify({"prediction": results[0]["letter"],
                    "confidence": results[0]["confidence"],
                    "top5": results})

if __name__ == "__main__":
    print(f"Running on device: {device}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)