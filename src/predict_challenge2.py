"""
predict_challenge2.py — Generate submission for Challenge 2/4 (answer generation)

Uses a two-stage pipeline:
  1. Binary classifier GATE: if p(answerable) < gate_threshold → output "unanswerable"
  2. Answer generator: run only for images the binary classifier deems answerable

This leverages the strong binary classifier (75%+ val accuracy) to correctly
route the ~49% unanswerable questions, significantly boosting the generator's score.

COLAB USAGE:
    !python src/predict_challenge2.py --name "moritz_knodler" --challenge 2
"""

import argparse
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

from src.dataset import VizWizAnswerDataset, build_vocab, build_answer_vocab, encode_answer, encode_text
from answer_generator import VizWizAnswerGenerator
from binary_classifier import VizWizBinaryClassifier


def find_image_dir(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="firstname_lastname format")
    parser.add_argument("--challenge", default="2", choices=["2", "4"])
    parser.add_argument("--generator_checkpoint", default="/content/best_generator.pt")
    parser.add_argument("--binary_checkpoint",    default="/content/best_model.pt")
    parser.add_argument("--threshold_path",       default="/content/best_threshold.pt",
                        help="Saved threshold from binary classifier training")
    parser.add_argument("--data_root", default="/content/data")
    args = parser.parse_args()

    DATA_ROOT      = Path(args.data_root)
    TEST_IMAGE_DIR = find_image_dir(DATA_ROOT / "test")

    ann_base = DATA_ROOT / "Annotations"
    if not (ann_base / "train.json").exists():
        ann_base = next(ann_base.iterdir())
    test_ann_path = ann_base / "test.json"

    # ---- Load test annotations (indices 100–199) ----
    with open(test_ann_path) as f:
        all_test = json.load(f)
    test_annotations = all_test[100:200]
    # Add dummy field so VizWizAnswerDataset doesn't error
    for ann in test_annotations:
        ann.setdefault("answerable", 0)
    print(f"Test samples (indices 100–199): {len(test_annotations)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load generator checkpoint (contains saved vocabs) ----
    gen_ckpt      = torch.load(args.generator_checkpoint, map_location=device)
    q_vocab       = gen_ckpt["q_vocab"]
    ans_vocab     = gen_ckpt["ans_vocab"]
    inv_ans_vocab = {v: k for k, v in ans_vocab.items()}
    SOS_IDX = ans_vocab["<sos>"]
    EOS_IDX = ans_vocab["<eos>"]
    PAD_IDX = ans_vocab["<pad>"]
    print(f"Q vocab: {len(q_vocab)} | Ans vocab: {len(ans_vocab)}")

    # ---- Load gate threshold from binary classifier training ----
    gate_threshold = torch.load(args.threshold_path, map_location="cpu")["threshold"]
    print(f"Gate threshold (p_answerable): {gate_threshold:.2f} — images below this → 'unanswerable'")

    # ---- Generator model — architecture read from checkpoint ----
    num_layers = gen_ckpt.get("num_layers", 2)   # default 2 for old checkpoints
    embed_dim  = gen_ckpt.get("embed_dim",  256)
    print(f"Generator architecture: embed_dim={embed_dim}, num_layers={num_layers}")
    generator = VizWizAnswerGenerator(
        q_vocab_size=len(q_vocab), ans_vocab_size=len(ans_vocab),
        embed_dim=embed_dim, num_heads=4, num_layers=num_layers,
        q_max_len=20, ans_max_len=12, dropout=0.3,
        sos_idx=SOS_IDX, eos_idx=EOS_IDX,
    ).to(device)
    generator.load_state_dict(gen_ckpt["model_state"])
    generator.eval()
    print(f"Generator:         {sum(p.numel() for p in generator.parameters()):,} params")

    # ---- Binary classifier gate model ----
    # The binary classifier was trained on 10k samples → different vocab size.
    # Rebuild its vocab from the train set the same way predict_challenge1.py does.
    train_ann_path = ann_base / "train.json"
    with open(train_ann_path) as f:
        all_train = json.load(f)
    train_available = {p.name for p in find_image_dir(DATA_ROOT / "train").glob("*.jpg")}
    train_annotations_bin = [a for a in all_train if a["image"] in train_available][:10_000]
    binary_q_vocab = build_vocab([a["question"] for a in train_annotations_bin], min_freq=1)
    print(f"Binary classifier q_vocab: {len(binary_q_vocab)}")

    binary = VizWizBinaryClassifier(
        vocab_size=len(binary_q_vocab), embed_dim=256, num_heads=4,
        num_layers=2, max_len=20, dropout=0.3,
    ).to(device)
    binary.load_state_dict(torch.load(args.binary_checkpoint, map_location=device))
    binary.eval()
    print(f"Binary classifier: {sum(p.numel() for p in binary.parameters()):,} params")

    # ---- Dataset & DataLoader ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VizWizAnswerDataset(
        test_annotations, TEST_IMAGE_DIR, q_vocab, ans_vocab,
        q_max_len=20, ans_max_len=12, transform=transform,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # ---- Gated inference ----
    all_pred_text = []
    gated_count   = 0

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)
            q_tokens = batch["q_tokens"].to(device)

            # Step 1: binary classifier gate
            # Binary classifier has its own (larger) vocab — encode separately
            bin_token_list = []
            for img_name in batch["image_name"]:
                ann = next(a for a in test_annotations if a["image"] == img_name)
                bin_token_list.append(encode_text(ann["question"], binary_q_vocab, 20))
            bin_tokens = torch.stack(bin_token_list).to(device)
            p_answerable = binary(images, bin_tokens).sigmoid().squeeze(1)  # [B]

            # Step 2: generator output for all (we'll override gated ones)
            gen_ids = generator.greedy_decode(images, q_tokens)  # [B, ANS_MAX_LEN]

            for i in range(images.size(0)):
                if p_answerable[i].item() < gate_threshold:
                    # Binary classifier says unanswerable → skip generator
                    pred_text = "unanswerable"
                    gated_count += 1
                else:
                    tokens = [inv_ans_vocab.get(t.item(), "")
                              for t in gen_ids[i]
                              if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                    pred_text = " ".join(tokens).strip() or "unanswerable"
                all_pred_text.append(pred_text)

    assert len(all_pred_text) == 100

    # ---- Encode predictions as token-id tensor for .pkl ----
    pred_tensors = [encode_answer(t, ans_vocab, 12) for t in all_pred_text]
    pred_tensor  = torch.stack(pred_tensors)  # [100, 12]

    # ---- Save .pkl ----
    out_pkl = Path(f"/content/{args.name}_challenge{args.challenge}.pkl")
    torch.save(pred_tensor, out_pkl)

    # ---- Human-readable answers ----
    out_txt = Path(f"/content/{args.name}_challenge{args.challenge}_answers.txt")
    with open(out_txt, "w") as f:
        f.write(f"Gated as unanswerable by binary classifier: {gated_count}/100\n\n")
        for i, (ann, pred) in enumerate(zip(test_annotations, all_pred_text)):
            f.write(f"[{100+i}] Q: {ann['question']}\n")
            f.write(f"       Pred: {pred}\n\n")

    print(f"\nGated as unanswerable: {gated_count}/100")
    print(f"Saved predictions to:  {out_pkl}")
    print(f"Saved text answers to: {out_txt}")
    print("\nSample predictions:")
    for i in range(min(5, len(all_pred_text))):
        print(f"  [{100+i}] {test_annotations[i]['question']} → {all_pred_text[i]}")


if __name__ == "__main__":
    main()
