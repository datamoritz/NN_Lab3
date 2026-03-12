"""
train_clip_binary.py — Train CLIP-based binary classifier for Challenge 3.

COLAB USAGE:
    !python src/train_clip_binary.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from clip_models import CLIPBinaryClassifier
from clip_dataset import CLIPBinaryDataset, vizwiz_accuracy

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT    = Path("/content/data")
FEAT_ROOT    = Path("/content/clip_features")

ANN_PATH     = DATA_ROOT / "Annotations"
if not (ANN_PATH / "train.json").exists():
    ANN_PATH = next(ANN_PATH.iterdir())

CHECKPOINT_PATH   = Path("/content/best_clip_binary_75.pt")
THRESHOLD_PATH    = Path("/content/best_clip_binary_threshold_75.pt")

MAX_TRAIN_SAMPLES = 15_392
MAX_VAL_SAMPLES   = None

HIDDEN_DIM  = 256
DROPOUT     = 0.3
BATCH_SIZE  = 512
NUM_EPOCHS  = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05

# -------------------------------------------------------
# Device
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# -------------------------------------------------------
# Load features and annotations
# -------------------------------------------------------
vis_train = torch.load(FEAT_ROOT / "VizWiz_train_CLIP_Image.pkl", map_location="cpu")
txt_train = torch.load(FEAT_ROOT / "VizWiz_train_CLIP_Text.pkl",  map_location="cpu")
vis_val   = torch.load(FEAT_ROOT / "VizWiz_val_CLIP_Image.pkl",   map_location="cpu")
txt_val   = torch.load(FEAT_ROOT / "VizWiz_val_CLIP_Text.pkl",    map_location="cpu")
print(f"Train features: {vis_train.shape} | Val features: {vis_val.shape}")

with open(ANN_PATH / "train.json") as f:
    all_train = json.load(f)
with open(ANN_PATH / "val.json") as f:
    all_val = json.load(f)

# Keep track of original indices for feature lookup
train_indices  = list(range(min(MAX_TRAIN_SAMPLES, len(all_train))))
train_anns     = [all_train[i] for i in train_indices]

val_indices    = list(range(len(all_val))) if MAX_VAL_SAMPLES is None else list(range(MAX_VAL_SAMPLES))
val_anns       = [all_val[i] for i in val_indices]

print(f"Train: {len(train_anns)} | Val: {len(val_anns)}")

# -------------------------------------------------------
# Class balance
# -------------------------------------------------------
num_pos    = sum(int(a["answerable"]) for a in train_anns)
num_neg    = len(train_anns) - num_pos
pos_weight = torch.tensor([num_neg / num_pos]).to(device)
print(f"Class balance — pos: {num_pos}, neg: {num_neg}, pos_weight: {pos_weight.item():.3f}")

# -------------------------------------------------------
# Datasets & Loaders
# -------------------------------------------------------
train_dataset = CLIPBinaryDataset(train_anns, train_indices, vis_train, txt_train)
val_dataset   = CLIPBinaryDataset(val_anns,   val_indices,   vis_val,   txt_val)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -------------------------------------------------------
# Model
# -------------------------------------------------------
model     = CLIPBinaryClassifier(feat_dim=512, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
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
        loss   = criterion(logits, smooth_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validate
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
          + (" <- best" if is_best else ""))

# Threshold scan
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
accuracy = (tp + tn) / (tp + tn + fp + fn)
tpr = tp / (tp + fn + 1e-8)
tnr = tn / (tn + fp + 1e-8)

print(f"\nBest threshold: {best_thresh:.2f}")
print(f"Val accuracy:   {accuracy:.4f}  (TPR={tpr:.4f}, TNR={tnr:.4f})")
print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
torch.save({"threshold": best_thresh}, THRESHOLD_PATH)
print(f"Checkpoint: {CHECKPOINT_PATH} | Threshold: {THRESHOLD_PATH}")
