"""
eval_gated.py — Measure gated pipeline VizWiz accuracy on the val set.

Runs the full binary-classifier gate + generator pipeline on val
and reports the VizWiz score so you can compare against the
generator-only baseline.

COLAB USAGE:
    !python src/eval_gated.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from src.dataset import (
    VizWizAnswerDataset, build_vocab, build_answer_vocab,
    encode_text, vizwiz_accuracy,
)
from answer_generator import VizWizAnswerGenerator
from binary_classifier import VizWizBinaryClassifier


def find_image_dir(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


DATA_ROOT      = Path("/content/data")
VAL_IMAGE_DIR  = find_image_dir(DATA_ROOT / "val")
ann_base       = DATA_ROOT / "Annotations"
if not (ann_base / "val.json").exists():
    ann_base = next(ann_base.iterdir())

GENERATOR_CHECKPOINT = "/content/best_generator.pt"
BINARY_CHECKPOINT    = "/content/best_model.pt"
THRESHOLD_PATH       = "/content/best_threshold.pt"
MAX_VAL_SAMPLES      = 500   # use first 500 val samples for speed; set None for all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load val annotations ----
with open(ann_base / "val.json") as f:
    all_val = json.load(f)
val_available = {p.name for p in VAL_IMAGE_DIR.glob("*.jpg")}
val_annotations = [a for a in all_val if a["image"] in val_available]
if MAX_VAL_SAMPLES:
    val_annotations = val_annotations[:MAX_VAL_SAMPLES]
print(f"Val samples: {len(val_annotations)}")

# ---- Load generator checkpoint ----
gen_ckpt      = torch.load(GENERATOR_CHECKPOINT, map_location=device)
q_vocab       = gen_ckpt["q_vocab"]
ans_vocab     = gen_ckpt["ans_vocab"]
inv_ans_vocab = {v: k for k, v in ans_vocab.items()}
SOS_IDX = ans_vocab["<sos>"]
EOS_IDX = ans_vocab["<eos>"]
PAD_IDX = ans_vocab["<pad>"]
num_layers = gen_ckpt.get("num_layers", 2)
embed_dim  = gen_ckpt.get("embed_dim", 256)

gate_threshold = torch.load(THRESHOLD_PATH, map_location="cpu")["threshold"]
print(f"Gate threshold: {gate_threshold:.2f}")

generator = VizWizAnswerGenerator(
    q_vocab_size=len(q_vocab), ans_vocab_size=len(ans_vocab),
    embed_dim=embed_dim, num_heads=4, num_layers=num_layers,
    q_max_len=20, ans_max_len=12, dropout=0.3,
    sos_idx=SOS_IDX, eos_idx=EOS_IDX,
).to(device)
generator.load_state_dict(gen_ckpt["model_state"])
generator.eval()

# ---- Load binary classifier ----
with open(ann_base / "train.json") as f:
    all_train = json.load(f)
train_available = {p.name for p in find_image_dir(DATA_ROOT / "train").glob("*.jpg")}
train_annotations = [a for a in all_train if a["image"] in train_available][:10_000]
binary_q_vocab = build_vocab([a["question"] for a in train_annotations], min_freq=1)

binary = VizWizBinaryClassifier(
    vocab_size=len(binary_q_vocab), embed_dim=256, num_heads=4,
    num_layers=2, max_len=20, dropout=0.3,
).to(device)
binary.load_state_dict(torch.load(BINARY_CHECKPOINT, map_location=device))
binary.eval()

# ---- Dataset ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    from torch.utils.data import default_collate
    answers = [item.pop("answers") for item in batch]
    collated = default_collate(batch)
    collated["answers"] = answers
    return collated

dataset = VizWizAnswerDataset(
    val_annotations, VAL_IMAGE_DIR, q_vocab, ans_vocab,
    q_max_len=20, ans_max_len=12, transform=transform,
)
loader = DataLoader(dataset, batch_size=64, shuffle=False,
                    num_workers=2, collate_fn=collate_fn)

# ---- Evaluate ----
total = gated = 0
sum_vizwiz_gated  = 0.0
sum_vizwiz_nogated = 0.0

with torch.no_grad():
    for batch in loader:
        images      = batch["image"].to(device)
        q_tokens    = batch["q_tokens"].to(device)
        all_answers = batch["answers"]

        # Binary gate tokens
        bin_token_list = []
        for img_name in batch["image_name"]:
            ann = next(a for a in val_annotations if a["image"] == img_name)
            bin_token_list.append(encode_text(ann["question"], binary_q_vocab, 20))
        bin_tokens    = torch.stack(bin_token_list).to(device)
        p_answerable  = binary(images, bin_tokens).sigmoid().squeeze(1)

        gen_ids = generator.greedy_decode(images, q_tokens)

        for i in range(images.size(0)):
            answers = all_answers[i]
            is_gated = p_answerable[i].item() < gate_threshold

            if is_gated:
                pred_text = "unanswerable"
                gated += 1
            else:
                tokens = [inv_ans_vocab.get(t.item(), "")
                          for t in gen_ids[i]
                          if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                pred_text = " ".join(tokens).strip() or "unanswerable"

            vz = vizwiz_accuracy(pred_text, answers)
            sum_vizwiz_gated += vz
            total += 1

            # Also measure generator-only for comparison
            tokens = [inv_ans_vocab.get(t.item(), "")
                      for t in gen_ids[i]
                      if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
            gen_text = " ".join(tokens).strip() or "unanswerable"
            sum_vizwiz_nogated += vizwiz_accuracy(gen_text, answers)

print(f"\n{'='*50}")
print(f"Val samples evaluated:    {total}")
print(f"Gated as unanswerable:    {gated} ({100*gated/total:.1f}%)")
print(f"\nGenerator only VizWiz:    {sum_vizwiz_nogated/total:.4f}")
print(f"Gated pipeline VizWiz:    {sum_vizwiz_gated/total:.4f}")
print(f"Improvement from gating:  {(sum_vizwiz_gated-sum_vizwiz_nogated)/total:+.4f}")
print(f"{'='*50}")
