"""
predict_challenge1.py — Generate submission for Challenge 1/3 (binary answerability)

Runs on test samples at indices 100–199 (inclusive) and saves a 1D tensor
of 0/1 predictions as a .pkl file using torch.save().

COLAB USAGE:
    !python src/predict_challenge1.py --name "moritz_knodler"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from src.dataset import VizWizBinaryDataset, build_vocab
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
                        help="Your name in firstname_lastname format, e.g. moritz_knodler")
    parser.add_argument("--challenge", default="1", choices=["1", "3"],
                        help="Challenge number (1 or 3)")
    parser.add_argument("--checkpoint", default="/content/best_model.pt")
    parser.add_argument("--threshold", default="/content/best_threshold.pt")
    parser.add_argument("--data_root", default="/content/data")
    parser.add_argument("--train_ann", default=None,
                        help="Path to train.json (needed to rebuild vocab)")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    TEST_IMAGE_DIR = find_image_dir(DATA_ROOT / "test")

    # Locate annotation files
    ann_base = DATA_ROOT / "Annotations"
    if not (ann_base / "train.json").exists():
        ann_base = next(ann_base.iterdir())

    test_ann_path  = ann_base / "test.json"
    train_ann_path = Path(args.train_ann) if args.train_ann else ann_base / "train.json"

    # ---- Load test annotations (indices 100–199) ----
    with open(test_ann_path) as f:
        all_test = json.load(f)

    test_annotations = all_test[100:200]   # second 100 samples (inclusive)
    print(f"Test samples (indices 100–199): {len(test_annotations)}")

    # ---- Rebuild vocab from train set ----
    with open(train_ann_path) as f:
        all_train = json.load(f)
    train_available = {p.name for p in find_image_dir(DATA_ROOT / "train").glob("*.jpg")}
    train_annotations = [a for a in all_train if a["image"] in train_available][:10_000]
    vocab = build_vocab([a["question"] for a in train_annotations], min_freq=1)
    print(f"Vocab size: {len(vocab)}")

    # ---- Load checkpoint & threshold ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    threshold_data = torch.load(args.threshold, map_location="cpu")
    threshold = threshold_data["threshold"]
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Using threshold: {threshold:.2f}")

    model = VizWizBinaryClassifier(
        vocab_size=len(vocab),
        embed_dim=256, num_heads=4, num_layers=2, max_len=20, dropout=0.3,
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Dataset & DataLoader ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VizWizBinaryDataset(
        annotations=test_annotations,
        image_dir=TEST_IMAGE_DIR,
        vocab=vocab,
        max_len=20,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # ---- Inference ----
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            probs  = model(images, tokens).sigmoid()
            preds  = (probs >= threshold).float().squeeze(1)
            all_preds.append(preds.cpu())

    predictions = torch.cat(all_preds).long()  # [100] — 0 or 1
    assert len(predictions) == 100, f"Expected 100 predictions, got {len(predictions)}"

    # ---- Save as .pkl ----
    out_path = Path(f"/content/{args.name}_challenge{args.challenge}.pkl")
    torch.save(predictions, out_path)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Answerable (1): {predictions.sum().item()} / 100")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
