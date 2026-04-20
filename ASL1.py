import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from pathlib import Path
import json

DATA_DIR = Path("asl_alphabet_train/asl_alphabet_train")
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
print(f"Classes: {full_ds.classes}")
print(f"Total images: {len(full_ds)}")

n_val = len(full_ds) // 5
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(full_ds.classes))
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        if batch_idx % 100 == 0:
            print(f"  epoch {epoch+1} batch {batch_idx}/{len(train_loader)}  loss={loss.item():.4f}")

    train_acc = correct / total

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total += imgs.size(0)
    val_acc = v_correct / v_total

    print(f"Epoch {epoch+1}/{EPOCHS}  loss={running_loss/total:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

torch.save(model.state_dict(), "asl_resnet18.pt")
with open("asl_classes.json", "w") as f:
    json.dump(full_ds.classes, f)
print("Saved asl_resnet18.pt and asl_classes.json")
