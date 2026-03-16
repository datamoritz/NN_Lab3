"""
train_ch1.py — Challenge 1: Binary answerability (CNN + Transformer)
               10,000 training samples | saves best_ch1.pt | auto-infers test set

═══════════════════════════════════════════════════════════
COLAB WORKFLOW (run once per session in a notebook cell):

    from google.colab import drive
    drive.mount('/content/drive')

    # clone repo (skip if already done)
    !git clone https://github.com/datamoritz/NN_Lab3.git /content/NN_Lab3
    %cd /content/NN_Lab3
    !git pull          # ← pull latest changes

    # unzip data (skip if already done)
    import zipfile, pathlib
    DATA_DIR   = pathlib.Path("/content/data");  DATA_DIR.mkdir(exist_ok=True)
    DRIVE_DATA = pathlib.Path("/content/drive/MyDrive/NN Lab3/Data")
    for zip_name, dest in [
        ("train.zip",       DATA_DIR/"train"),
        ("val.zip",         DATA_DIR/"val"),
        ("test.zip",        DATA_DIR/"test"),
        ("Annotations.zip", DATA_DIR/"Annotations"),
    ]:
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(DRIVE_DATA/zip_name) as zf: zf.extractall(dest)
        print(f"Extracted {zip_name}")

    # train + auto-infer
    !python src/train_ch1.py --name moritz_knoedler
═══════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import VizWizBinaryDataset, build_vocab, encode_text
from src.model import VizWizBinaryClassifier


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="moritz_knoedler",
                    help="firstname_lastname used in output .pkl filename")
args = parser.parse_args()

STUDENT_NAME = args.name


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def find_image_dir(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
DATA_ROOT       = Path("/content/data")
TRAIN_IMAGE_DIR = DATA_ROOT / "train"
VAL_IMAGE_DIR   = DATA_ROOT / "val"
_ann_base       = DATA_ROOT / "Annotations"
if not (_ann_base / "train.json").exists():
    _ann_base = next(_ann_base.iterdir())
TRAIN_ANN_PATH  = _ann_base / "train.json"
VAL_ANN_PATH    = _ann_base / "val.json"

CHECKPOINT_PATH = Path("/content/best_ch1.pt")
THRESHOLD_PATH  = Path("/content/best_ch1_threshold.pt")

FAST_MODE = False
IMG_SIZE          = 128   if FAST_MODE else 224
MAX_TRAIN_SAMPLES = 5_000 if FAST_MODE else 10_000
MAX_VAL_SAMPLES   = 2_000 if FAST_MODE else None
NUM_EPOCHS        = 10    if FAST_MODE else 20
MAX_LEN           = 20

BATCH_SIZE  = 256
NUM_WORKERS = 4
EMBED_DIM   = 256
NUM_HEADS   = 4
NUM_LAYERS  = 2
DROPOUT     = 0.3

LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
WARMUP_EPOCHS       = 2
LABEL_SMOOTH        = 0.05
POS_WEIGHT_OVERRIDE = None


# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(
    f"\n{'='*60}\n"
    f"  CHALLENGE   : 1  (binary answerability — CNN+Transformer)\n"
    f"  MODE        : {'FAST' if FAST_MODE else 'FULL'}\n"
    f"  Device      : {DEVICE}\n"
    f"  IMG_SIZE    : {IMG_SIZE}px\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  EMBED_DIM   : {EMBED_DIM}  |  NUM_HEADS: {NUM_HEADS}  |  NUM_LAYERS: {NUM_LAYERS}\n"
    f"  DROPOUT     : {DROPOUT}\n"
    f"  NUM_EPOCHS  : {NUM_EPOCHS}  |  WARMUP: {WARMUP_EPOCHS}\n"
    f"  LR          : {LR}  |  WEIGHT_DECAY: {WEIGHT_DECAY}\n"
    f"  CHECKPOINT  : {CHECKPOINT_PATH}\n"
    f"{'='*60}\n"
)


# ──────────────────────────────────────────────────────────
# Annotations
# ──────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
test_transform = val_transform


# ──────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────
vocab = build_vocab([a["question"] for a in train_annotations], min_freq=1)
print(f"Vocab size: {len(vocab)}")


# ──────────────────────────────────────────────────────────
# Datasets & DataLoaders
# ──────────────────────────────────────────────────────────
train_dataset = VizWizBinaryDataset(
    annotations=train_annotations, image_dir=TRAIN_IMAGE_DIR,
    vocab=vocab, max_len=MAX_LEN, transform=train_transform,
)
val_dataset = VizWizBinaryDataset(
    annotations=val_annotations, image_dir=VAL_IMAGE_DIR,
    vocab=vocab, max_len=MAX_LEN, transform=val_transform,
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


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────
model = VizWizBinaryClassifier(
    vocab_size=len(vocab), embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS, max_len=MAX_LEN, dropout=DROPOUT,
).to(DEVICE)

num_pos = sum(int(a["answerable"]) for a in train_annotations)
num_neg = len(train_annotations) - num_pos
auto_pw = num_neg / num_pos
pw_value = POS_WEIGHT_OVERRIDE if POS_WEIGHT_OVERRIDE is not None else auto_pw
pos_weight = torch.tensor([pw_value], dtype=torch.float32).to(DEVICE)
print(f"Class balance — pos: {num_pos}, neg: {num_neg}, auto_pw: {auto_pw:.3f}, using pw: {pw_value:.3f}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(), "lr": LR},
    {"params": model.text_encoder.parameters(),  "lr": LR * 0.3},
    {"params": model.fusion.parameters(),         "lr": LR},
    {"params": model.classifier.parameters(),     "lr": LR},
], weight_decay=WEIGHT_DECAY)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
)


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
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
    is_best  = val_acc > best_val_acc

    if is_best:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
        + (" ← best" if is_best else "")
    )


# ──────────────────────────────────────────────────────────
# Threshold scan on val
# ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        all_probs.append(model(images, tokens).sigmoid().cpu())
        all_labels.append(labels.cpu())

all_probs  = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

best_thresh, best_accuracy = 0.5, 0.0
for t in [i / 100 for i in range(30, 71)]:
    preds = (all_probs >= t).float()
    acc = (preds == all_labels).float().mean().item()
    if acc > best_accuracy:
        best_accuracy = acc
        best_thresh   = t

preds = (all_probs >= best_thresh).float()
tp = ((preds == 1) & (all_labels == 1)).sum().item()
tn = ((preds == 0) & (all_labels == 0)).sum().item()
fp = ((preds == 1) & (all_labels == 0)).sum().item()
fn = ((preds == 0) & (all_labels == 1)).sum().item()
final_acc = (tp + tn) / (tp + tn + fp + fn)
tpr = tp / (tp + fn + 1e-8)
tnr = tn / (tn + fp + 1e-8)
bal_acc = 0.5 * (tpr + tnr)

torch.save({"threshold": best_thresh}, THRESHOLD_PATH)


# ──────────────────────────────────────────────────────────
# Auto-inference on test set (indices 100–199)
# ──────────────────────────────────────────────────────────
test_ann_path = _ann_base / "test.json"
with open(test_ann_path) as f:
    all_test = json.load(f)
test_annotations = all_test[100:200]
for ann in test_annotations:
    ann.setdefault("answerable", 0)

TEST_IMAGE_DIR = find_image_dir(DATA_ROOT / "test")
test_dataset = VizWizBinaryDataset(
    annotations=test_annotations, image_dir=TEST_IMAGE_DIR,
    vocab=vocab, max_len=MAX_LEN, transform=test_transform,
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        tokens = batch["tokens"].to(DEVICE)
        probs  = model(images, tokens).sigmoid()
        preds  = (probs >= best_thresh).float().squeeze(1)
        all_preds.append(preds.cpu())

predictions = torch.cat(all_preds).long()
assert len(predictions) == 100

out_pkl = Path(f"/content/{STUDENT_NAME}_challenge1.pkl")
torch.save(predictions, out_pkl)


# ──────────────────────────────────────────────────────────
# Final verbose summary
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHALLENGE 1 — TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best val accuracy (during training) : {best_val_acc:.4f}")
print(f"  Final val accuracy (best threshold) : {final_acc:.4f}")
print(f"  Balanced accuracy                   : {bal_acc:.4f}  (TPR={tpr:.4f}, TNR={tnr:.4f})")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"  Best threshold                      : {best_thresh:.2f}")
print(f"  Answerable predictions (1) / 100    : {predictions.sum().item()}")
print(f"  Checkpoint saved to                 : {CHECKPOINT_PATH}")
print(f"  Threshold saved to                  : {THRESHOLD_PATH}")
print(f"  Submission file                     : {out_pkl}")
print(f"{'='*60}\n")
