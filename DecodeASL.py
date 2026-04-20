import torch, json
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import sys

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0).to(device)  # shape: (1, 3, 128, 128)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    top5 = torch.topk(probs, 5)

print(f"Prediction: {classes[top5.indices[0]]}  (confidence {top5.values[0]:.3f})")
print("Top 5:")
for idx, p in zip(top5.indices, top5.values):
    print(f"  {classes[idx]}: {p:.3f}")
