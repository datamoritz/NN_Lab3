"""
train_ch4.py — Challenge 4: Answer classification (CLIP features, closed-set top-1000)
               10,000 training samples | saves best_ch4.pt | auto-infers test set

COLAB USAGE (after data + CLIP features are available):
    !python src/train_ch4.py --name moritz_knoedler

NOTE: Challenge 4 inference uses the CLIP binary classifier from Ch3 as a gate.
      If best_ch3.pt + best_ch3_threshold.pt exist they are used automatically.
      Otherwise the answer classifier runs without gating.
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

from clip_models import CLIPAnswerClassifier, CLIPBinaryClassifier
from clip_dataset import CLIPAnswerDataset, CLIPBinaryDataset, build_clip_answer_vocab, vizwiz_accuracy_clip


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

CHECKPOINT_PATH = Path("/content/best_ch4.pt")

# Optional Ch3 gate paths
CH3_CHECKPOINT = Path("/content/best_ch3.pt")
CH3_THRESHOLD  = Path("/content/best_ch3_threshold.pt")

MAX_TRAIN_SAMPLES = 10_000
MAX_VAL_SAMPLES   = None
TOP_K_ANSWERS     = 1000

HIDDEN_DIM   = 512
DROPOUT      = 0.3
BATCH_SIZE   = 512
NUM_EPOCHS   = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1


# ──────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(
    f"\n{'='*60}\n"
    f"  CHALLENGE   : 4  (answer classification — CLIP features)\n"
    f"  Device      : {device}\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  TOP_K       : {TOP_K_ANSWERS}  |  HIDDEN_DIM: {HIDDEN_DIM}\n"
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
# Answer vocabulary (top-K)
# ──────────────────────────────────────────────────────────
answer_vocab, top_answers = build_clip_answer_vocab(train_anns, top_k=TOP_K_ANSWERS)
print(f"Answer vocabulary: {len(top_answers)} classes  (top-{TOP_K_ANSWERS})")
print(f"  'unanswerable' index: {answer_vocab.get('unanswerable', 'NOT IN VOCAB')}")


# ──────────────────────────────────────────────────────────
# Datasets & Loaders
# ──────────────────────────────────────────────────────────
def collate_fn(batch):
    from torch.utils.data import default_collate
    answers = [item.pop("answers") for item in batch]
    collated = default_collate(batch)
    collated["answers"] = answers
    return collated

train_dataset = CLIPAnswerDataset(train_anns, train_indices, vis_train, txt_train, answer_vocab, top_answers)
val_dataset   = CLIPAnswerDataset(val_anns,   val_indices,   vis_val,   txt_val,   answer_vocab, top_answers)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                           collate_fn=collate_fn)


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────
num_classes = len(top_answers)
model       = CLIPAnswerClassifier(feat_dim=512, hidden_dim=HIDDEN_DIM,
                                   num_answers=num_classes, dropout=DROPOUT).to(device)
criterion   = nn.CrossEntropyLoss(ignore_index=num_classes, label_smoothing=LABEL_SMOOTH)
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ──────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────
best_val_vizwiz = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        vis     = batch["vis"].to(device)
        txt     = batch["txt"].to(device)
        ans_idx = torch.tensor(batch["answer_idx"], dtype=torch.long).to(device)
        optimizer.zero_grad()
        logits = model(vis, txt)
        loss   = criterion(logits, ans_idx)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    model.eval()
    total = 0
    sum_vizwiz = 0.0
    with torch.no_grad():
        for batch in val_loader:
            vis         = batch["vis"].to(device)
            txt         = batch["txt"].to(device)
            all_answers = batch["answers"]
            logits      = model(vis, txt)
            pred_idxs   = logits.argmax(dim=-1).cpu()
            for i in range(len(pred_idxs)):
                pred_text = top_answers[pred_idxs[i].item()]
                sum_vizwiz += vizwiz_accuracy_clip(pred_text, all_answers[i])
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
          + (" ← best" if is_best else ""))


# ──────────────────────────────────────────────────────────
# Auto-inference on test set (indices 100–199)
# ──────────────────────────────────────────────────────────
ckpt         = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

vis_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Image.pkl", map_location="cpu")
txt_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Text.pkl",  map_location="cpu")
vis_test = vis_test[100:200].to(device)
txt_test = txt_test[100:200].to(device)
print(f"\nTest features loaded: {vis_test.shape}")

# Load Ch3 binary gate if available
gate_model = None
gate_threshold = None
if CH3_CHECKPOINT.exists() and CH3_THRESHOLD.exists():
    print(f"Loading Ch3 CLIP binary gate from {CH3_CHECKPOINT} ...")
    gate_model = CLIPBinaryClassifier(feat_dim=512, hidden_dim=256, dropout=0.3).to(device)
    gate_model.load_state_dict(torch.load(CH3_CHECKPOINT, map_location=device))
    gate_model.eval()
    gate_threshold = torch.load(CH3_THRESHOLD, map_location="cpu")["threshold"]
    print(f"Gate threshold: {gate_threshold:.2f}")
else:
    print("Ch3 checkpoint not found — running answer classifier without binary gate.")

all_pred_text = []
gated_count   = 0

with torch.no_grad():
    logits    = model(vis_test, txt_test)    # [100, K]
    pred_idxs = logits.argmax(dim=-1)        # [100]

    if gate_model is not None:
        gate_probs = gate_model(vis_test, txt_test).sigmoid().squeeze(1)  # [100]

    for i in range(len(pred_idxs)):
        if gate_model is not None and gate_probs[i].item() < gate_threshold:
            pred_text = "unanswerable"
            gated_count += 1
        else:
            pred_text = top_answers[pred_idxs[i].item()]
        all_pred_text.append(pred_text)

assert len(all_pred_text) == 100

# Encode as class-index tensor (same format as original predict_challenge4.py)
# Re-map predictions through the vocab for gated "unanswerable" overrides
final_idxs = []
for t in all_pred_text:
    final_idxs.append(answer_vocab.get(t, pred_idxs[all_pred_text.index(t)].item()))
pred_tensor = torch.tensor(final_idxs, dtype=torch.long)

out_pkl = Path(f"/content/{STUDENT_NAME}_challenge4.pkl")
torch.save(pred_tensor, out_pkl)

# Human-readable
with open(ANN_PATH / "test.json") as f:
    all_test = json.load(f)
test_anns = all_test[100:200]

out_txt = Path(f"/content/{STUDENT_NAME}_challenge4_answers.txt")
with open(out_txt, "w") as f:
    for i, (ann, pred) in enumerate(zip(test_anns, all_pred_text)):
        f.write(f"[{100+i}] Q: {ann['question']}\n       Pred: {pred}\n\n")


# ──────────────────────────────────────────────────────────
# Final verbose summary
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHALLENGE 4 — TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best val VizWiz accuracy (Human Ans): {best_val_vizwiz:.4f}")
print(f"  Gate used                           : {'Yes (best_ch3.pt)' if gate_model else 'No'}")
if gate_model:
    print(f"  Gated as unanswerable (test)        : {gated_count}/100")
print(f"  Checkpoint saved to                 : {CHECKPOINT_PATH}")
print(f"  Submission file                     : {out_pkl}")
print(f"  Human-readable answers              : {out_txt}")
print(f"{'='*60}\n")
