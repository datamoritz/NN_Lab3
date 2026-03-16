"""
train_ch2.py — Challenge 2: Answer generation (CNN + Transformer seq2seq)
               10,000 training samples | saves best_ch2.pt | auto-infers test set

COLAB USAGE (after data is already extracted):
    !python src/train_ch2.py --name moritz_knoedler

NOTE: Challenge 2 inference uses the binary classifier from Ch1 as a gate.
      If best_ch1.pt + best_ch1_threshold.pt exist they are used automatically.
      Otherwise the generator runs without gating.
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

from src.dataset import (
    VizWizAnswerDataset, build_vocab, build_answer_vocab,
    get_majority_answer, encode_text, encode_answer, vizwiz_accuracy,
)
from src.model import VizWizBinaryClassifier
from answer_generator import VizWizAnswerGenerator


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


def collate_fn(batch):
    from torch.utils.data import default_collate
    answers = [item.pop("answers") for item in batch]
    collated = default_collate(batch)
    collated["answers"] = answers
    return collated


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

CHECKPOINT_PATH = Path("/content/best_ch2.pt")

# Optional Ch1 gate paths
CH1_CHECKPOINT = Path("/content/best_ch1.pt")
CH1_THRESHOLD  = Path("/content/best_ch1_threshold.pt")

FAST_MODE = False
IMG_SIZE          = 128   if FAST_MODE else 224
MAX_TRAIN_SAMPLES = 5_000 if FAST_MODE else 10_000
MAX_VAL_SAMPLES   = 2_000 if FAST_MODE else None
NUM_EPOCHS        = 10    if FAST_MODE else 20

Q_MAX_LEN   = 20
ANS_MAX_LEN = 12

BATCH_SIZE   = 128
NUM_WORKERS  = 4
EMBED_DIM    = 256
NUM_HEADS    = 4
NUM_LAYERS   = 2
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2


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
    f"  CHALLENGE   : 2  (answer generation — seq2seq)\n"
    f"  MODE        : {'FAST' if FAST_MODE else 'FULL'}\n"
    f"  Device      : {DEVICE}\n"
    f"  IMG_SIZE    : {IMG_SIZE}px  |  ANS_MAX_LEN: {ANS_MAX_LEN}\n"
    f"  Train/Val   : {MAX_TRAIN_SAMPLES} / {MAX_VAL_SAMPLES or 'all'}\n"
    f"  EMBED_DIM   : {EMBED_DIM}  |  NUM_HEADS: {NUM_HEADS}  |  NUM_LAYERS: {NUM_LAYERS}\n"
    f"  LR          : {LR}  |  DROPOUT: {DROPOUT}\n"
    f"  CHECKPOINT  : {CHECKPOINT_PATH}\n"
    f"{'='*60}\n"
)


# ──────────────────────────────────────────────────────────
# Annotations
# ──────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────
# Vocabularies
# ──────────────────────────────────────────────────────────
q_vocab   = build_vocab([a["question"] for a in train_annotations], min_freq=1)
ans_vocab = build_answer_vocab(train_annotations, min_freq=4)
inv_ans_vocab = {v: k for k, v in ans_vocab.items()}
SOS_IDX = ans_vocab["<sos>"]
EOS_IDX = ans_vocab["<eos>"]
PAD_IDX = ans_vocab["<pad>"]
print(f"Q vocab: {len(q_vocab)} | Ans vocab: {len(ans_vocab)}")


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


# ──────────────────────────────────────────────────────────
# Datasets & DataLoaders
# ──────────────────────────────────────────────────────────
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
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                          collate_fn=collate_fn)


# ──────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────
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

class_weights = torch.ones(len(ans_vocab), device=DEVICE)
unanswerable_idx = ans_vocab.get("unanswerable")
if unanswerable_idx is not None:
    class_weights[unanswerable_idx] = 0.3
    print(f"Down-weighting 'unanswerable' (idx={unanswerable_idx}) to 0.3")
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, weight=class_weights, label_smoothing=0.1)

optimizer = torch.optim.AdamW([
    {"params": model.image_encoder.parameters(),   "lr": LR},
    {"params": model.question_encoder.parameters(), "lr": LR * 0.3},
    {"params": model.answer_decoder.parameters(),   "lr": LR},
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
best_val_vizwiz = 0.0

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images     = batch["image"].to(DEVICE)
        q_tokens   = batch["q_tokens"].to(DEVICE)
        ans_tokens = batch["ans_tokens"].to(DEVICE)

        decoder_input  = ans_tokens[:, :-1]
        decoder_target = ans_tokens[:, 1:]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images, q_tokens, decoder_input)
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

    model.eval()
    total = 0
    sum_vizwiz = 0.0
    exact_correct = 0

    with torch.no_grad():
        for batch in val_loader:
            images      = batch["image"].to(DEVICE)
            q_tokens    = batch["q_tokens"].to(DEVICE)
            targets     = batch["answer_text"]
            all_answers = batch["answers"]

            pred_ids = model.greedy_decode(images, q_tokens)

            for i in range(len(targets)):
                pred_tokens = [inv_ans_vocab.get(t.item(), "") for t in pred_ids[i]
                               if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                pred_text = " ".join(pred_tokens).strip()
                sum_vizwiz += vizwiz_accuracy(pred_text, all_answers[i])
                if pred_text == targets[i].strip():
                    exact_correct += 1
                total += 1

                if total <= 5:
                    vz = vizwiz_accuracy(pred_text, all_answers[i])
                    print(f"  [sample {total}] GT: '{targets[i]}' | Pred: '{pred_text}' | VizWiz: {vz:.2f}")

    vizwiz_acc = sum_vizwiz / total if total > 0 else 0.0
    exact_acc  = exact_correct / total if total > 0 else 0.0
    avg_loss   = train_loss / len(train_loader)
    is_best    = vizwiz_acc > best_val_vizwiz

    if is_best:
        best_val_vizwiz = vizwiz_acc
        torch.save({
            "model_state": model.state_dict(),
            "q_vocab":     q_vocab,
            "ans_vocab":   ans_vocab,
            "num_layers":  NUM_LAYERS,
            "embed_dim":   EMBED_DIM,
        }, CHECKPOINT_PATH)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | "
        f"VizWiz: {vizwiz_acc:.4f} | ExactMatch: {exact_acc:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
        + (" ← best" if is_best else "")
    )


# ──────────────────────────────────────────────────────────
# Auto-inference on test set (indices 100–199)
# ──────────────────────────────────────────────────────────
gen_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(gen_ckpt["model_state"])
model.eval()

test_ann_path = _ann_base / "test.json"
with open(test_ann_path) as f:
    all_test = json.load(f)
test_annotations = all_test[100:200]
for ann in test_annotations:
    ann.setdefault("answerable", 0)

TEST_IMAGE_DIR = find_image_dir(DATA_ROOT / "test")
test_dataset = VizWizAnswerDataset(
    test_annotations, TEST_IMAGE_DIR, q_vocab, ans_vocab,
    q_max_len=Q_MAX_LEN, ans_max_len=ANS_MAX_LEN, transform=val_transform,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load binary gate if available
gate_model = None
gate_threshold = None
binary_q_vocab = None
if CH1_CHECKPOINT.exists() and CH1_THRESHOLD.exists():
    print(f"\nLoading Ch1 binary gate from {CH1_CHECKPOINT} ...")
    binary_q_vocab = build_vocab([a["question"] for a in train_annotations], min_freq=1)
    gate_model = VizWizBinaryClassifier(
        vocab_size=len(binary_q_vocab), embed_dim=256, num_heads=4,
        num_layers=2, max_len=20, dropout=0.3,
    ).to(DEVICE)
    gate_model.load_state_dict(torch.load(CH1_CHECKPOINT, map_location=DEVICE))
    gate_model.eval()
    gate_threshold = torch.load(CH1_THRESHOLD, map_location="cpu")["threshold"]
    print(f"Gate threshold: {gate_threshold:.2f}")
else:
    print("\nCh1 checkpoint not found — running generator without binary gate.")

all_pred_text = []
gated_count   = 0

with torch.no_grad():
    for batch in test_loader:
        images   = batch["image"].to(DEVICE)
        q_tokens = batch["q_tokens"].to(DEVICE)
        gen_ids  = model.greedy_decode(images, q_tokens)

        if gate_model is not None:
            bin_token_list = []
            for img_name in batch["image_name"]:
                ann = next(a for a in test_annotations if a["image"] == img_name)
                bin_token_list.append(encode_text(ann["question"], binary_q_vocab, 20))
            bin_tokens   = torch.stack(bin_token_list).to(DEVICE)
            p_answerable = gate_model(images, bin_tokens).sigmoid().squeeze(1)

        for i in range(images.size(0)):
            if gate_model is not None and p_answerable[i].item() < gate_threshold:
                pred_text = "unanswerable"
                gated_count += 1
            else:
                tokens = [inv_ans_vocab.get(t.item(), "")
                          for t in gen_ids[i]
                          if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                pred_text = " ".join(tokens).strip() or "unanswerable"
            all_pred_text.append(pred_text)

assert len(all_pred_text) == 100

pred_tensors = [encode_answer(t, ans_vocab, 12) for t in all_pred_text]
pred_tensor  = torch.stack(pred_tensors)   # [100, 12]

out_pkl = Path(f"/content/{STUDENT_NAME}_challenge2.pkl")
torch.save(pred_tensor, out_pkl)

out_txt = Path(f"/content/{STUDENT_NAME}_challenge2_answers.txt")
with open(out_txt, "w") as f:
    f.write(f"Gated as unanswerable: {gated_count}/100\n\n")
    for i, (ann, pred) in enumerate(zip(test_annotations, all_pred_text)):
        f.write(f"[{100+i}] Q: {ann['question']}\n       Pred: {pred}\n\n")


# ──────────────────────────────────────────────────────────
# Final verbose summary
# ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  CHALLENGE 2 — TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best val VizWiz accuracy            : {best_val_vizwiz:.4f}")
print(f"  Gate used                           : {'Yes (best_ch1.pt)' if gate_model else 'No'}")
if gate_model:
    print(f"  Gated as unanswerable (test)        : {gated_count}/100")
print(f"  Checkpoint saved to                 : {CHECKPOINT_PATH}")
print(f"  Submission file                     : {out_pkl}")
print(f"  Human-readable answers              : {out_txt}")
print(f"{'='*60}\n")
