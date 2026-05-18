from flask import Flask, request, jsonify, send_from_directory
import torch, json
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import io, os, tempfile, cv2
from itertools import product as cart_product
from englishtoglossified import translate_to_asl_gloss
from spellchecker import SpellChecker
import anthropic
import replicate

spell = SpellChecker()

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load model once at startup ──────────────────────────────────────────
IMG_SIZE = 128
PRE_SIZE = 200
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("asl_classes.json") as f:
    classes = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("asl_resnet18.pt", map_location=device))
model = model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((PRE_SIZE, PRE_SIZE)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Page routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/quiz")
def quiz():
    return send_from_directory(".", "test.html")

@app.route("/recognize")
def recognize():
    return send_from_directory(".", "recognize.html")

@app.route("/video")
def video():
    return send_from_directory(".", "video.html")

@app.route("/animations")
def animations():
    return send_from_directory(".", "animations.html")

# ── API routes ──────────────────────────────────────────────────────────
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

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    gloss = translate_to_asl_gloss(text)
    letters = [ch for ch in gloss if ch.isalpha()]

    return jsonify({"gloss": gloss, "letters": letters})

@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        file.save(tmp.name)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps * 0.15))  # sample more often

        CONF_THRESHOLD = 0.4  # discard low-confidence predictions

        raw_predictions = []   # top-1 per frame (for display)
        frame_top5 = []        # top-5 per frame (for permutation search)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Center-crop to square (hand is usually centered)
                w, h = pil_img.size
                short = min(w, h)
                left = (w - short) // 2
                top = (h - short) // 2
                pil_img = pil_img.crop((left, top, left + short, top + short))

                x = tf(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    top5 = torch.topk(probs, 5)

                top1_letter = classes[top5.indices[0].item()]
                top1_conf = float(top5.values[0])

                # Filter out non-letter classes
                t5 = []
                for idx, p in zip(top5.indices.tolist(), top5.values.tolist()):
                    l = classes[idx]
                    if l not in ("del", "nothing", "space"):
                        t5.append({"letter": l, "confidence": round(float(p), 3)})

                if top1_letter not in ("del", "nothing", "space") and top1_conf >= CONF_THRESHOLD:
                    raw_predictions.append({"letter": top1_letter, "confidence": round(top1_conf, 3)})
                    frame_top5.append(t5)

            frame_idx += 1

        cap.release()

        # Group consecutive frames by top-1 prediction
        MIN_RUN = 3
        groups = []
        for i, pred in enumerate(raw_predictions):
            if groups and groups[-1]["letter"] == pred["letter"]:
                groups[-1]["count"] += 1
                groups[-1]["frame_indices"].append(i)
            else:
                groups.append({"letter": pred["letter"], "count": 1, "frame_indices": [i]})

        # Keep only groups that meet the minimum run length
        valid_groups = [g for g in groups if g["count"] >= MIN_RUN]
        deduped_str = "".join(g["letter"] for g in valid_groups)

        # For each valid group, find top 5 candidate letters by total confidence
        candidates_per_pos = []
        for group in valid_groups:
            letter_scores = {}
            for fi in group["frame_indices"]:
                for pred in frame_top5[fi]:
                    letter_scores[pred["letter"]] = letter_scores.get(pred["letter"], 0) + pred["confidence"]
            sorted_letters = sorted(letter_scores.items(), key=lambda x: -x[1])
            candidates_per_pos.append([l for l, _ in sorted_letters[:5]])

        # Try all permutations of candidates to find a real word
        corrected = deduped_str.lower()
        if candidates_per_pos:
            # Cap candidates per position to avoid explosion on long words
            max_cands = 5 if len(candidates_per_pos) <= 6 else 3
            limited = [c[:max_cands] for c in candidates_per_pos]

            all_combos = ["".join(combo).lower() for combo in cart_product(*limited)]
            known = spell.known(all_combos)

            if known:
                # Pick the first known word (product order = highest-confidence first)
                for word in all_combos:
                    if word in known:
                        corrected = word
                        break
            else:
                # Fall back to spell correction on top-1 sequence
                corrected = spell.correction(deduped_str.lower()) or deduped_str.lower()

        return jsonify({
            "raw": raw_predictions,
            "deduplicated": deduped_str,
            "corrected": corrected
        })

    finally:
        os.unlink(tmp.name)

@app.route("/generate-animation", methods=["POST"])
def generate_animation():
    data = request.get_json()
    if not data or "word" not in data:
        return jsonify({"error": "No word provided"}), 400

    word = data["word"].strip().upper()
    if not word:
        return jsonify({"error": "Empty word"}), 400

    # Step 1: Use Claude to generate a video prompt description (with fallback)
    description = None
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        print(f"[Animation] ANTHROPIC_API_KEY set: {bool(api_key)}")
        claude = anthropic.Anthropic(api_key=api_key)
        msg = claude.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f'Describe the ASL sign for the word "{word}" as a single, detailed video prompt. '
                    "Describe the complete motion from start to finish: starting hand shape and position, "
                    "the movement trajectory, and the ending position. Be specific about hand shape, "
                    "palm orientation, finger positions, and movement direction. "
                    "Write it as one continuous description suitable for a text-to-video AI model. "
                    "Return ONLY valid JSON with no extra text, in this exact format:\n"
                    '{"description": "the full motion description"}'
                )
            }]
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
        claude_data = json.loads(raw)
        description = claude_data["description"]
    except Exception as e:
        print(f"[Animation] Claude failed, using fallback: {type(e).__name__}: {e}")
        description = f"Performing the ASL sign for the word {word}, showing the hand shape, position, and movement from start to finish"

    # Step 2: Generate a video using Replicate Wan 2.5
    prompt = (
        "A close-up video of a person wearing a blue sweatshirt performing American Sign Language with their hands. "
        f"{description}. "
        "Front-facing view, plain white background, realistic, studio lighting, "
        "detailed hands, smooth continuous motion."
    )

    try:
        print(f"[Animation] Generating video for '{word}'...")
        output = replicate.run(
            "wavespeedai/wan-2.1-t2v-480p",
            input={
                "prompt": prompt,
            }
        )
        video_url = str(output) if output else None
        print(f"[Animation] Video URL: {video_url}")
    except Exception as e:
        return jsonify({"error": f"Replicate API error: {str(e)}"}), 500

    if not video_url:
        return jsonify({"error": "No video generated"}), 500

    return jsonify({"word": word, "description": description, "video_url": video_url})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Running on device: {device}")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=False, host="0.0.0.0", port=port)
