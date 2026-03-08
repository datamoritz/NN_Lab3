"""
train.py — VizWiz binary answerability classifier
Designed to run on Google Colab with data stored as zip files on Google Drive.

─────────────────────────────────────────────────────────────
COLAB SETUP  (run these in a notebook cell before importing)
─────────────────────────────────────────────────────────────
# 1. Mount Drive and clone the repo
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo (skip if already cloned)
!git clone https://github.com/datamoritz/NN_Lab3.git /content/NN_Lab3
%cd /content/NN_Lab3

# 3. Unzip data from Drive into /content/data/
import zipfile, pathlib
DATA_DIR = pathlib.Path("/content/data")
DATA_DIR.mkdir(exist_ok=True)

DRIVE_DATA = pathlib.Path("/content/drive/MyDrive/NN Lab3/Data")
for zip_name, dest in [
    ("train.zip",       DATA_DIR / "train"),
    ("val.zip",         DATA_DIR / "val"),
    ("test.zip",        DATA_DIR / "test"),
    ("Annotations.zip", DATA_DIR / "Annotations"),
]:
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(DRIVE_DATA / zip_name) as zf:
        zf.extractall(dest)
    print(f"Extracted {zip_name}")

# 4. Run training
!python src/train.py
─────────────────────────────────────────────────────────────
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Allow `python src/train.py` from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import VizWizBinaryDataset, build_vocab
from src.model import VizWizBinaryClassifier


def find_image_dir(base: Path) -> Path:
    """Return the directory that actually contains .jpg files.
    Handles zips that extract into a subdirectory (e.g. train/train/*.jpg)."""
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")

# -------------------------------------------------------
# Config
# -------------------------------------------------------

# Paths — point to wherever data was extracted on Colab (or locally)
DATA_ROOT      = Path("/content/data")
TRAIN_IMAGE_DIR = DATA_ROOT / "train"
VAL_IMAGE_DIR   = DATA_ROOT / "val"
_ann_base       = DATA_ROOT / "Annotations"
# Handle zips that extract into a subdirectory
if not (_ann_base / "train.json").exists():
    _ann_base = next(_ann_base.iterdir())
TRAIN_ANN_PATH  = _ann_base / "train.json"
VAL_ANN_PATH    = _ann_base / "val.json"

# Where to save the best checkpoint
CHECKPOINT_PATH = Path("/content/best_model.pt")

# Dataset
MAX_TRAIN_SAMPLES = 2000  # assignment cap; set None for full split
MAX_VAL_SAMPLES   = 1000
MAX_LEN           = 20

# DataLoader
BATCH_SIZE  = 128   # larger batch fits easily on a GPU
NUM_WORKERS = 4    # safe on Colab; set 0 if you hit issues

# Model
EMBED_DIM  = 256
NUM_HEADS  = 4
NUM_LAYERS = 2
DROPOUT    = 0.3

# Training
NUM_EPOCHS = 3
LR         = 1e-3
WEIGHT_DECAY = 1e-4

# -------------------------------------------------------
# Device
# -------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Training on: {DEVICE}")

# -------------------------------------------------------
# Annotations
# -------------------------------------------------------
with open(TRAIN_ANN_PATH) as f:
    all_train = json.load(f)

TRAIN_IMAGE_DIR = find_image_dir(TRAIN_IMAGE_DIR)
train_available = {p.name for p in TRAIN_IMAGE_DIR.glob("*.jpg")}
train_annotations = [a for a in all_train if a["image"] in train_available]
if MAX_TRAIN_SAMPLES is not None:
    train_annotations = train_annotations[:MAX_TRAIN_SAMPLES]

with open(VAL_ANN_PATH) as f:
    all_val = json.load(f)

VAL_IMAGE_DIR = find_image_dir(VAL_IMAGE_DIR)
val_available = {p.name for p in VAL_IMAGE_DIR.glob("*.jpg")}
val_annotations = [a for a in all_val if a["image"] in val_available]
if MAX_VAL_SAMPLES is not None:
    val_annotations = val_annotations[:MAX_VAL_SAMPLES]

print(f"Train: {len(train_annotations)} samples | Val: {len(val_annotations)} samples")

# -------------------------------------------------------
# Transforms
# -------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------------
# Vocabulary (built from training questions only)
# -------------------------------------------------------
questions = [a["question"] for a in train_annotations]
vocab = build_vocab(questions, min_freq=1)
print(f"Vocab size: {len(vocab)}")

# -------------------------------------------------------
# Datasets & DataLoaders
# -------------------------------------------------------
train_dataset = VizWizBinaryDataset(
    annotations=train_annotations,
    image_dir=TRAIN_IMAGE_DIR,
    vocab=vocab,
    max_len=MAX_LEN,
    transform=train_transform,
)
val_dataset = VizWizBinaryDataset(
    annotations=val_annotations,
    image_dir=VAL_IMAGE_DIR,
    vocab=vocab,
    max_len=MAX_LEN,
    transform=val_transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# -------------------------------------------------------
# Model
# -------------------------------------------------------
model = VizWizBinaryClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=MAX_LEN,
    dropout=DROPOUT,
).to(DEVICE)

num_pos = sum(int(a["answerable"]) for a in train_annotations)
num_neg = len(train_annotations) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(DEVICE)
print(f"Class balance — pos: {num_pos}, neg: {num_neg}, pos_weight: {pos_weight.item():.3f}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    # ---- Train ----
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images, tokens), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # ---- Validate ----
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            tokens = batch["tokens"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            preds = (model(images, tokens).sigmoid() >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc  = correct / total
    avg_loss = train_loss / len(train_loader)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
        + (" <- best" if is_best else "")
    )

# -------------------------------------------------------
# Final Evaluation (best checkpoint)
# -------------------------------------------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

correct = total = tp = tn = fp = fn = 0

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        preds = (model(images, tokens).sigmoid() >= 0.5).float()

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"\nFinal val accuracy (best checkpoint): {accuracy:.4f}")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"Best val acc seen during training:    {best_val_acc:.4f}")
