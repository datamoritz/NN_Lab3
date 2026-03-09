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

# -------------------------------------------------------
# Fast mode — set True for quick hyperparameter search:
#   - 128x128 images  (~4x fewer pixels to process)
#   - 5000 train / 2000 val samples
#   - 10 epochs
# Set False for the final full-quality run.
# -------------------------------------------------------
FAST_MODE = True

IMG_SIZE          = 128        if FAST_MODE else 224
MAX_TRAIN_SAMPLES = 5_000      if FAST_MODE else 10_000
MAX_VAL_SAMPLES   = 2_000      if FAST_MODE else None
NUM_EPOCHS        = 10         if FAST_MODE else 20
MAX_LEN           = 20

# DataLoader
BATCH_SIZE  = 256
NUM_WORKERS = 4

# Model
EMBED_DIM  = 256
NUM_HEADS  = 4
NUM_LAYERS = 2
DROPOUT    = 0.3

# Training
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
LABEL_SMOOTH  = 0.05
# Override the auto-computed pos_weight (neg/pos ratio ~0.37).
# Higher value penalises false positives more → improves TNR.
# Set to None to use the automatic class-balance ratio.
POS_WEIGHT_OVERRIDE = 1.0

# -------------------------------------------------------
# Device
# -------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(
    f"\n{'='*60}\n"
    f"  MODE        : {'FAST' if FAST_MODE else 'FULL'}\n"
    f"  Device      : {DEVICE}\n"
    f"  --- Data ---\n"
    f"  IMG_SIZE    : {IMG_SIZE}px\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  BATCH_SIZE  : {BATCH_SIZE}  |  NUM_WORKERS: {NUM_WORKERS}\n"
    f"  --- Model ---\n"
    f"  EMBED_DIM   : {EMBED_DIM}  |  NUM_HEADS: {NUM_HEADS}  |  NUM_LAYERS: {NUM_LAYERS}\n"
    f"  DROPOUT     : {DROPOUT}\n"
    f"  --- Training ---\n"
    f"  NUM_EPOCHS  : {NUM_EPOCHS}  |  WARMUP: {WARMUP_EPOCHS}\n"
    f"  LR          : {LR} (text_enc: {LR*0.3})  |  WEIGHT_DECAY: {WEIGHT_DECAY}\n"
    f"  LABEL_SMOOTH: {LABEL_SMOOTH}  |  POS_WEIGHT: {POS_WEIGHT_OVERRIDE or 'auto'}\n"
    f"{'='*60}\n"
)

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
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Answerability augmentation: simulate blurry/low-quality images that
    # are characteristic of unanswerable VizWiz samples.
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
auto_pw = num_neg / num_pos
pw_value = POS_WEIGHT_OVERRIDE if POS_WEIGHT_OVERRIDE is not None else auto_pw
pos_weight = torch.tensor([pw_value], dtype=torch.float32).to(DEVICE)
print(f"Class balance — pos: {num_pos}, neg: {num_neg}, auto_pw: {auto_pw:.3f}, using pw: {pw_value:.3f}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Label smoothing applied manually to targets (see config section above)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Separate LRs: Transformer (text encoder) is more sensitive to large updates
# and benefits from a lower LR than the CNN and fusion layers.
optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(), "lr": LR},
    {"params": model.text_encoder.parameters(),  "lr": LR * 0.3},
    {"params": model.fusion.parameters(),         "lr": LR},
    {"params": model.classifier.parameters(),     "lr": LR},
], weight_decay=WEIGHT_DECAY)

# Mixed precision scaler — ~2x faster on CUDA, no effect on CPU/MPS
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

# Cosine decay with a non-zero floor so LR never stalls at 0
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
# Linear warmup: scale LR from 0 -> 1 over WARMUP_EPOCHS
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)

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
        # Smooth labels: 0 -> LABEL_SMOOTH, 1 -> 1 - LABEL_SMOOTH
        smooth_labels = labels * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss = criterion(model(images, tokens), smooth_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
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
# Final Evaluation (best checkpoint, threshold = 0.5)
# -------------------------------------------------------
THRESHOLD = 0.5

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

tp = tn = fp = fn = 0

with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        preds = (model(images, tokens).sigmoid() >= THRESHOLD).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

accuracy = (tp + tn) / (tp + tn + fp + fn)
tpr = tp / (tp + fn + 1e-8)
tnr = tn / (tn + fp + 1e-8)
bal_acc = 0.5 * (tpr + tnr)

print(f"\nFinal val accuracy (threshold=0.5): {accuracy:.4f}")
print(f"Balanced accuracy:                  {bal_acc:.4f}  (TPR={tpr:.4f}, TNR={tnr:.4f})")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"Best val acc seen during training:  {best_val_acc:.4f}")
