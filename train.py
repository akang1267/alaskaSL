import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, random_split
import json, os

# ── Config ───────────────────────────────────────────────────────────────
DATA_DIR = "asl_data/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = 128
PRE_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ── Transforms ───────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((PRE_SIZE, PRE_SIZE)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((PRE_SIZE, PRE_SIZE)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR)
classes = full_dataset.classes
print(f"Classes ({len(classes)}): {classes}")

# Save class list
with open("asl_classes.json", "w") as f:
    json.dump(classes, f)

# Split 85% train / 15% val
n_total = len(full_dataset)
n_val = int(n_total * 0.15)
n_train = n_total - n_val
train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))

# Apply different transforms
train_set.dataset.transform = train_tf
# For val we need a wrapper since both splits share the same dataset object
class ValSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is a PIL image since ImageFolder loads it
        # But with transform already set on parent, we need raw image
        return self.transform(img), label

# Actually, let's just use two separate ImageFolder instances
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

# Use the same split indices
train_set = torch.utils.data.Subset(train_dataset, train_set.indices)
val_set = torch.utils.data.Subset(val_dataset, val_set.indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# ── Model ────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# ── Training loop ────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_loss = running_loss / total
    train_acc = correct / total

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "asl_resnet18.pt")
        print(f"  -> Saved best model (val acc: {val_acc:.4f})")

    scheduler.step()

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
print("Weights saved to asl_resnet18.pt")
