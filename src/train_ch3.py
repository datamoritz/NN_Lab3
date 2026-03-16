"""
train_ch3.py — Challenge 3: Binary answerability (CLIP features)
               10,000 training samples | saves best_ch3.pt | auto-infers test set

COLAB USAGE (after data + CLIP features are available):
    !python src/train_ch3.py --name moritz_knoedler

Expected feature paths:
    /content/clip_features/VizWiz_train_CLIP_Image.pkl
    /content/clip_features/VizWiz_train_CLIP_Text.pkl
    /content/clip_features/VizWiz_val_CLIP_Image.pkl
    /content/clip_features/VizWiz_val_CLIP_Text.pkl
    /content/clip_features/VizWiz_test_CLIP_Image.pkl
    /content/clip_features/VizWiz_test_CLIP_Text.pkl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_models import CLIPBinaryClassifier
from clip_dataset import CLIPBinaryDataset


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="moritz_knoedler",
                    help="firstname_lastname used in output .pkl filename")
args = parser.parse_args()

STUDENT_NAME = args.name


# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
DATA_ROOT = Path("/content/data")
FEAT_ROOT = Path("/content/clip_features")

ANN_PATH = DATA_ROOT / "Annotations"
if not (ANN_PATH / "train.json").exists():
    ANN_PATH = next(ANN_PATH.iterdir())

CHECKPOINT_PATH = Path("/content/best_ch3.pt")
THRESHOLD_PATH  = Path("/content/best_ch3_threshold.pt")

MAX_TRAIN_SAMPLES = 10_000
MAX_VAL_SAMPLES   = None

HIDDEN_DIM   = 256
DROPOUT      = 0.3
BATCH_SIZE   = 512
NUM_EPOCHS   = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05


# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"\n{'='*60}\n"
    f"  CHALLENGE   : 3  (binary answerability — CLIP features)\n"
    f"  Device      : {device}\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  HIDDEN_DIM  : {HIDDEN_DIM}  |  DROPOUT: {DROPOUT}\n"
    f"  NUM_EPOCHS  : {NUM_EPOCHS}  |  LR: {LR}\n"
    f"  CHECKPOINT  : {CHECKPOINT_PATH}\n"
    f"{'='*60}\n"
)


# ──────────────────────────────────────────────────────────
# Load features and annotations
# ──────────────────────────────────────────────────────────
vis_train = torch.load(FEAT_ROOT / "VizWiz_train_CLIP_Image.pkl", map_location="cpu")
txt_train = torch.load(FEAT_ROOT / "VizWiz_train_CLIP_Text.pkl",  map_location="cpu")
vis_val   = torch.load(FEAT_ROOT / "VizWiz_val_CLIP_Image.pkl",   map_location="cpu")
txt_val   = torch.load(FEAT_ROOT / "VizWiz_val_CLIP_Text.pkl",    map_location="cpu")
print(f"Train features: {vis_train.shape} | Val features: {vis_val.shape}")

with open(ANN_PATH / "train.json") as f:
    all_train = json.load(f)
with open(ANN_PATH / "val.json") as f:
    all_val = json.load(f)

train_indices = list(range(min(MAX_TRAIN_SAMPLES, len(all_train))))
train_anns    = [all_train[i] for i in train_indices]
val_indices   = list(range(len(all_val))) if MAX_VAL_SAMPLES is None else list(range(MAX_VAL_SAMPLES))
val_anns      = [all_val[i] for i in val_indices]
print(f"Train: {len(train_anns)} | Val: {len(val_anns)}")


# ──────────────────────────────────────────────────────────
# Class balance
# ──────────────────────────────────────────────────────────
num_pos    = sum(int(a["answerable"]) for a in train_anns)
num_neg    = len(train_anns) - num_pos
pos_weight = torch.tensor([num_neg / num_pos]).to(device)
print(f"Class balance — pos: {num_pos}, neg: {num_neg}, pos_weight: {pos_weight.item():.3f}")


# ──────────────────────────────────────────────────────────
# Datasets & Loaders
# ──────────────────────────────────────────────────────────
train_dataset = CLIPBinaryDataset(train_anns, train_indices, vis_train, txt_train)
val_dataset   = CLIPBinaryDataset(val_anns,   val_indices,   vis_val,   txt_val)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────
model     = CLIPBinaryClassifier(feat_dim=512, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        vis    = batch["vis"].to(device)
        txt    = batch["txt"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(vis, txt)
        smooth_labels = labels * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
        loss = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            vis    = batch["vis"].to(device)
            txt    = batch["txt"].to(device)
            labels = batch["label"].to(device)
            preds  = (model(vis, txt).sigmoid() >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc  = correct / total
    avg_loss = train_loss / len(train_loader)
    is_best  = val_acc > best_val_acc

    if is_best:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
          + (" ← best" if is_best else ""))


# ──────────────────────────────────────────────────────────
# Threshold scan
# ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        vis    = batch["vis"].to(device)
        txt    = batch["txt"].to(device)
        labels = batch["label"].to(device)
        all_probs.append(model(vis, txt).sigmoid().cpu())
        all_labels.append(labels.cpu())
all_probs  = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

best_thresh, best_acc = 0.5, 0.0
for t in [i / 100 for i in range(30, 71)]:
    preds = (all_probs >= t).float()
    acc   = (preds == all_labels).float().mean().item()
    if acc > best_acc:
        best_acc, best_thresh = acc, t

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
vis_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Image.pkl", map_location="cpu")
txt_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Text.pkl",  map_location="cpu")
vis_test = vis_test[100:200].to(device)
txt_test = txt_test[100:200].to(device)
print(f"\nTest features loaded: {vis_test.shape}")

with torch.no_grad():
    probs = model(vis_test, txt_test).sigmoid().squeeze(1)
    preds = (probs >= best_thresh).long()

assert len(preds) == 100

out_pkl = Path(f"/content/{STUDENT_NAME}_challenge3.pkl")
torch.save(preds, out_pkl)


# ──────────────────────────────────────────────────────────
# Final verbose summary
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHALLENGE 3 — TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best val accuracy (during training) : {best_val_acc:.4f}")
print(f"  Final val accuracy (best threshold) : {final_acc:.4f}")
print(f"  Balanced accuracy                   : {bal_acc:.4f}  (TPR={tpr:.4f}, TNR={tnr:.4f})")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"  Best threshold                      : {best_thresh:.2f}")
print(f"  Answerable predictions (1) / 100    : {preds.sum().item()}")
print(f"  Checkpoint saved to                 : {CHECKPOINT_PATH}")
print(f"  Threshold saved to                  : {THRESHOLD_PATH}")
print(f"  Submission file                     : {out_pkl}")
print(f"{'='*60}\n")
