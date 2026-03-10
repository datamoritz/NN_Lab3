"""
train_clip_answer.py — Train CLIP-based answer classifier for Challenge 4.

COLAB USAGE:
    !python src/train_clip_answer.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from clip_models import CLIPAnswerClassifier
from clip_dataset import CLIPAnswerDataset, build_clip_answer_vocab, vizwiz_accuracy

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT  = Path("/content/data")
FEAT_ROOT  = Path("/content/clip_features")

ANN_PATH   = DATA_ROOT / "Annotations"
if not (ANN_PATH / "train.json").exists():
    ANN_PATH = next(ANN_PATH.iterdir())

CHECKPOINT_PATH = Path("/content/best_clip_answer.pt")

MAX_TRAIN_SAMPLES = 10_000
MAX_VAL_SAMPLES   = None
TOP_K_ANSWERS     = 1000   # closed answer vocabulary size

HIDDEN_DIM   = 512
DROPOUT      = 0.3
BATCH_SIZE   = 512
NUM_EPOCHS   = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1

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

with open(ANN_PATH / "train.json") as f:
    all_train = json.load(f)
with open(ANN_PATH / "val.json") as f:
    all_val = json.load(f)

train_indices = list(range(min(MAX_TRAIN_SAMPLES, len(all_train))))
train_anns    = [all_train[i] for i in train_indices]
val_indices   = list(range(len(all_val))) if MAX_VAL_SAMPLES is None else list(range(MAX_VAL_SAMPLES))
val_anns      = [all_val[i] for i in val_indices]

print(f"Train: {len(train_anns)} | Val: {len(val_anns)}")

# -------------------------------------------------------
# Build answer vocabulary (top-K, "unanswerable" included)
# -------------------------------------------------------
answer_vocab, top_answers = build_clip_answer_vocab(train_anns, top_k=TOP_K_ANSWERS)
print(f"Answer vocabulary: {len(top_answers)} classes  (top-{TOP_K_ANSWERS})")
print(f"  'unanswerable' index: {answer_vocab.get('unanswerable', 'NOT IN VOCAB')}")

# -------------------------------------------------------
# Datasets & Loaders
# -------------------------------------------------------
train_dataset = CLIPAnswerDataset(train_anns, train_indices, vis_train, txt_train, answer_vocab, top_answers)
val_dataset   = CLIPAnswerDataset(val_anns,   val_indices,   vis_val,   txt_val,   answer_vocab, top_answers)
def collate_fn(batch):
    """Keep 'answers' (list of dicts) as a plain Python list — not collated by torch."""
    from torch.utils.data import default_collate
    answers = [item.pop("answers") for item in batch]
    collated = default_collate(batch)
    collated["answers"] = answers
    return collated

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                           collate_fn=collate_fn)

# -------------------------------------------------------
# Model
# -------------------------------------------------------
# +1 for answers outside the top-K vocabulary → mapped to unk_idx at training
num_classes = len(top_answers)
model       = CLIPAnswerClassifier(feat_dim=512, hidden_dim=HIDDEN_DIM,
                                    num_answers=num_classes, dropout=DROPOUT).to(device)
# Ignore samples whose target answer is outside the vocabulary (unk_idx)
criterion = nn.CrossEntropyLoss(ignore_index=num_classes, label_smoothing=LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
best_val_vizwiz = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        vis        = batch["vis"].to(device)
        txt        = batch["txt"].to(device)
        ans_idx    = torch.tensor(batch["answer_idx"], dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(vis, txt)              # [B, K]
        loss   = criterion(logits, ans_idx)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validate — VizWiz accuracy
    model.eval()
    total = 0
    sum_vizwiz = 0.0

    with torch.no_grad():
        for batch in val_loader:
            vis        = batch["vis"].to(device)
            txt        = batch["txt"].to(device)
            all_answers = batch["answers"]

            logits     = model(vis, txt)               # [B, K]
            pred_idxs  = logits.argmax(dim=-1).cpu()   # [B]

            for i in range(len(pred_idxs)):
                pred_text = top_answers[pred_idxs[i].item()]
                vz = vizwiz_accuracy(pred_text, all_answers[i])
                sum_vizwiz += vz
                total += 1

    val_vizwiz = sum_vizwiz / total if total > 0 else 0.0
    avg_loss   = train_loss / len(train_loader)
    is_best    = val_vizwiz > best_val_vizwiz

    if is_best:
        best_val_vizwiz = val_vizwiz
        torch.save({
            "model_state":  model.state_dict(),
            "answer_vocab": answer_vocab,
            "top_answers":  top_answers,
            "hidden_dim":   HIDDEN_DIM,
        }, CHECKPOINT_PATH)

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
          f"VizWiz: {val_vizwiz:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
          + (" <- best" if is_best else ""))

print(f"\nBest val VizWiz: {best_val_vizwiz:.4f}")
print(f"Checkpoint saved to {CHECKPOINT_PATH}")
