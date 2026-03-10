"""
predict_challenge3.py — Generate submission for Challenge 3 (CLIP binary).

COLAB USAGE:
    !python src/predict_challenge3.py --name "moritz_knodler"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

from clip_models import CLIPBinaryClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--challenge", default="3", choices=["1", "3"])
    parser.add_argument("--checkpoint",  default="/content/best_clip_binary.pt")
    parser.add_argument("--threshold",   default="/content/best_clip_binary_threshold.pt")
    parser.add_argument("--feat_root",   default="/content/drive/MyDrive/NN Lab3/Data")
    parser.add_argument("--data_root",   default="/content/data")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    FEAT_ROOT = Path(args.feat_root)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test features (indices 100-199)
    vis_test = torch.load(FEAT_ROOT / "test_image_features.pkl", map_location="cpu")
    txt_test = torch.load(FEAT_ROOT / "test_text_features.pkl",  map_location="cpu")
    vis_test = vis_test[100:200].to(device)
    txt_test = txt_test[100:200].to(device)
    print(f"Test features loaded: {vis_test.shape}")

    # Load threshold
    threshold = torch.load(args.threshold, map_location="cpu")["threshold"]
    print(f"Using threshold: {threshold:.2f}")

    # Load model
    model = CLIPBinaryClassifier(feat_dim=512, hidden_dim=256, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        probs  = model(vis_test, txt_test).sigmoid().squeeze(1)
        preds  = (probs >= threshold).long()

    assert len(preds) == 100
    out_path = Path(f"/content/{args.name}_challenge{args.challenge}.pkl")
    torch.save(preds, out_path)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Answerable (1): {preds.sum().item()} / 100")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
