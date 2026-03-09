"""
train_generator.py — Training script for VizWiz answer generation (Challenge 2/4)

COLAB SETUP (same data extraction as train_binary.py — skip if already done):
    %cd /content/NN_Lab3
    !git pull
    !python src/train_generator.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

_root = Path(__file__).resolve().parent.parent  # NN_Lab3/
_src  = Path(__file__).resolve().parent          # NN_Lab3/src/
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from src.dataset import (
    VizWizAnswerDataset, build_vocab, build_answer_vocab,
    get_majority_answer, encode_text,
)
from models.answer_generator import VizWizAnswerGenerator


def find_image_dir(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT       = Path("/content/data")
TRAIN_IMAGE_DIR = DATA_ROOT / "train"
VAL_IMAGE_DIR   = DATA_ROOT / "val"
_ann_base       = DATA_ROOT / "Annotations"
if not (_ann_base / "train.json").exists():
    _ann_base = next(_ann_base.iterdir())
TRAIN_ANN_PATH  = _ann_base / "train.json"
VAL_ANN_PATH    = _ann_base / "val.json"

CHECKPOINT_PATH  = Path("/content/best_generator.pt")
THRESHOLD_PATH   = Path("/content/best_threshold.pt")   # from binary training

FAST_MODE = True

IMG_SIZE          = 128   if FAST_MODE else 224
MAX_TRAIN_SAMPLES = 5_000 if FAST_MODE else 10_000
MAX_VAL_SAMPLES   = 2_000 if FAST_MODE else None
NUM_EPOCHS        = 10    if FAST_MODE else 20

Q_MAX_LEN   = 20
ANS_MAX_LEN = 12    # most VizWiz answers are 1–5 words; 12 tokens is safe

BATCH_SIZE   = 128
NUM_WORKERS  = 4
EMBED_DIM    = 256
NUM_HEADS    = 4
NUM_LAYERS   = 2
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

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
    f"  TASK        : Answer Generation (Challenge 2/4)\n"
    f"  MODE        : {'FAST' if FAST_MODE else 'FULL'}\n"
    f"  Device      : {DEVICE}\n"
    f"  IMG_SIZE    : {IMG_SIZE}px  |  ANS_MAX_LEN: {ANS_MAX_LEN}\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  EMBED_DIM   : {EMBED_DIM}  |  NUM_HEADS: {NUM_HEADS}  |  NUM_LAYERS: {NUM_LAYERS}\n"
    f"  LR          : {LR}  |  DROPOUT: {DROPOUT}\n"
    f"{'='*60}\n"
)

# -------------------------------------------------------
# Annotations
# -------------------------------------------------------
with open(TRAIN_ANN_PATH) as f:
    all_train = json.load(f)
with open(VAL_ANN_PATH) as f:
    all_val = json.load(f)

TRAIN_IMAGE_DIR = find_image_dir(TRAIN_IMAGE_DIR)
VAL_IMAGE_DIR   = find_image_dir(VAL_IMAGE_DIR)

train_available = {p.name for p in TRAIN_IMAGE_DIR.glob("*.jpg")}
train_annotations = [a for a in all_train if a["image"] in train_available]
if MAX_TRAIN_SAMPLES:
    train_annotations = train_annotations[:MAX_TRAIN_SAMPLES]

val_available = {p.name for p in VAL_IMAGE_DIR.glob("*.jpg")}
val_annotations = [a for a in all_val if a["image"] in val_available]
if MAX_VAL_SAMPLES:
    val_annotations = val_annotations[:MAX_VAL_SAMPLES]

print(f"Train: {len(train_annotations)} | Val: {len(val_annotations)}")

# -------------------------------------------------------
# Vocabularies
# -------------------------------------------------------
q_vocab   = build_vocab([a["question"] for a in train_annotations], min_freq=1)
ans_vocab = build_answer_vocab(train_annotations, min_freq=1)
inv_ans_vocab = {v: k for k, v in ans_vocab.items()}

SOS_IDX = ans_vocab["<sos>"]
EOS_IDX = ans_vocab["<eos>"]
PAD_IDX = ans_vocab["<pad>"]

print(f"Q vocab: {len(q_vocab)} | Ans vocab: {len(ans_vocab)}")

# -------------------------------------------------------
# Transforms
# -------------------------------------------------------
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

# -------------------------------------------------------
# Datasets & DataLoaders
# -------------------------------------------------------
train_dataset = VizWizAnswerDataset(
    train_annotations, TRAIN_IMAGE_DIR, q_vocab, ans_vocab,
    q_max_len=Q_MAX_LEN, ans_max_len=ANS_MAX_LEN, transform=train_transform,
)
val_dataset = VizWizAnswerDataset(
    val_annotations, VAL_IMAGE_DIR, q_vocab, ans_vocab,
    q_max_len=Q_MAX_LEN, ans_max_len=ANS_MAX_LEN, transform=val_transform,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

# -------------------------------------------------------
# Model
# -------------------------------------------------------
model = VizWizAnswerGenerator(
    q_vocab_size=len(q_vocab),
    ans_vocab_size=len(ans_vocab),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    q_max_len=Q_MAX_LEN,
    ans_max_len=ANS_MAX_LEN,
    dropout=DROPOUT,
    sos_idx=SOS_IDX,
    eos_idx=EOS_IDX,
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Ignore <pad> tokens in loss so they don't affect gradient
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(),   "lr": LR},
    {"params": model.question_encoder.parameters(),"lr": LR * 0.3},
    {"params": model.answer_decoder.parameters(),  "lr": LR},
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

# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    # ---- Train ----
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        images    = batch["image"].to(DEVICE)
        q_tokens  = batch["q_tokens"].to(DEVICE)
        ans_tokens = batch["ans_tokens"].to(DEVICE)  # [B, ANS_MAX_LEN]

        # Teacher forcing: input = ans_tokens[:, :-1], target = ans_tokens[:, 1:]
        decoder_input  = ans_tokens[:, :-1]   # [B, A-1]  starts with <sos>
        decoder_target = ans_tokens[:, 1:]    # [B, A-1]  ends with <eos>

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images, q_tokens, decoder_input)   # [B, A-1, V]
            # Reshape for CrossEntropyLoss: [B*(A-1), V] vs [B*(A-1)]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    scheduler.step()

    # ---- Validate (exact match accuracy on majority answer) ----
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in val_loader:
            images   = batch["image"].to(DEVICE)
            q_tokens = batch["q_tokens"].to(DEVICE)
            targets  = batch["answer_text"]   # list of strings

            pred_ids = model.greedy_decode(images, q_tokens)  # [B, ANS_MAX_LEN]

            for i in range(len(targets)):
                pred_tokens = [inv_ans_vocab.get(t.item(), "") for t in pred_ids[i]
                               if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                pred_text = " ".join(pred_tokens).strip()
                if pred_text == targets[i].strip():
                    correct += 1
                total += 1

    val_acc  = correct / total if total > 0 else 0.0
    avg_loss = train_loss / len(train_loader)
    is_best  = val_acc > best_val_acc

    if is_best:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "q_vocab":     q_vocab,
            "ans_vocab":   ans_vocab,
        }, CHECKPOINT_PATH)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
        f"Val ExactMatch: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        + (" <- best" if is_best else "")
    )

print(f"\nBest val exact-match accuracy: {best_val_acc:.4f}")
print(f"Checkpoint saved to {CHECKPOINT_PATH}")
