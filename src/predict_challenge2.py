"""
predict_challenge2.py — Generate submission for Challenge 2/4 (answer generation)

Runs on test samples at indices 100–199 and saves predictions as a .pkl file.
For text generation, the tensor contains encoded answer token ids; a companion
text file with decoded answers is also written for human inspection.

COLAB USAGE:
    !python src/predict_challenge2.py --name "moritz_knodler"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import VizWizAnswerDataset, build_vocab, build_answer_vocab
from src.models.answer_generator import VizWizAnswerGenerator


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
                        help="Your name in firstname_lastname format")
    parser.add_argument("--challenge", default="2", choices=["2", "4"])
    parser.add_argument("--checkpoint", default="/content/best_generator.pt")
    parser.add_argument("--data_root", default="/content/data")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    TEST_IMAGE_DIR = find_image_dir(DATA_ROOT / "test")

    ann_base = DATA_ROOT / "Annotations"
    if not (ann_base / "train.json").exists():
        ann_base = next(ann_base.iterdir())
    test_ann_path  = ann_base / "test.json"
    train_ann_path = ann_base / "train.json"

    # ---- Load test annotations (indices 100–199) ----
    with open(test_ann_path) as f:
        all_test = json.load(f)
    test_annotations = all_test[100:200]
    print(f"Test samples (indices 100–199): {len(test_annotations)}")

    # ---- Load checkpoint (contains saved vocabs) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    q_vocab   = ckpt["q_vocab"]
    ans_vocab = ckpt["ans_vocab"]
    inv_ans_vocab = {v: k for k, v in ans_vocab.items()}
    print(f"Q vocab: {len(q_vocab)} | Ans vocab: {len(ans_vocab)}")

    SOS_IDX = ans_vocab["<sos>"]
    EOS_IDX = ans_vocab["<eos>"]
    PAD_IDX = ans_vocab["<pad>"]

    # ---- Model ----
    model = VizWizAnswerGenerator(
        q_vocab_size=len(q_vocab),
        ans_vocab_size=len(ans_vocab),
        embed_dim=256, num_heads=4, num_layers=2,
        q_max_len=20, ans_max_len=12, dropout=0.3,
        sos_idx=SOS_IDX, eos_idx=EOS_IDX,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Dataset ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VizWizAnswerDataset(
        test_annotations, TEST_IMAGE_DIR, q_vocab, ans_vocab,
        q_max_len=20, ans_max_len=12, transform=transform,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # ---- Inference ----
    all_pred_ids = []
    all_pred_text = []

    with torch.no_grad():
        for batch in loader:
            images   = batch["image"].to(device)
            q_tokens = batch["q_tokens"].to(device)
            pred_ids = model.greedy_decode(images, q_tokens)  # [B, ANS_MAX_LEN]
            all_pred_ids.append(pred_ids.cpu())

            for i in range(pred_ids.size(0)):
                tokens = [inv_ans_vocab.get(t.item(), "")
                          for t in pred_ids[i]
                          if t.item() not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                all_pred_text.append(" ".join(tokens).strip())

    pred_tensor = torch.cat(all_pred_ids, dim=0)  # [100, ANS_MAX_LEN]
    assert len(all_pred_text) == 100

    # ---- Save .pkl (token id tensor) ----
    out_pkl = Path(f"/content/{args.name}_challenge{args.challenge}.pkl")
    torch.save(pred_tensor, out_pkl)

    # ---- Save human-readable answers ----
    out_txt = Path(f"/content/{args.name}_challenge{args.challenge}_answers.txt")
    with open(out_txt, "w") as f:
        for i, (ann, pred) in enumerate(zip(test_annotations, all_pred_text)):
            f.write(f"[{100+i}] Q: {ann['question']}\n")
            f.write(f"       Pred: {pred}\n\n")

    print(f"\nSaved predictions to: {out_pkl}")
    print(f"Saved decoded answers to: {out_txt}")
    print("\nSample predictions:")
    for i in range(min(5, len(all_pred_text))):
        print(f"  [{100+i}] {test_annotations[i]['question']} → {all_pred_text[i]}")


if __name__ == "__main__":
    main()
