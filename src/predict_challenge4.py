"""
predict_challenge4.py — Generate submission for Challenge 4 (CLIP answer classification).

COLAB USAGE:
    !python src/predict_challenge4.py --name "moritz_knodler"
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

from clip_models import CLIPAnswerClassifier
from clip_dataset import vizwiz_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",       required=True)
    parser.add_argument("--challenge",  default="4", choices=["2", "4"])
    parser.add_argument("--checkpoint", default="/content/best_clip_answer.pt")
    parser.add_argument("--feat_root",  default="/content/clip_features")
    parser.add_argument("--data_root",  default="/content/data")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    FEAT_ROOT = Path(args.feat_root)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (contains vocab + architecture params)
    ckpt        = torch.load(args.checkpoint, map_location=device)
    answer_vocab = ckpt["answer_vocab"]
    top_answers  = ckpt["top_answers"]
    hidden_dim   = ckpt.get("hidden_dim", 512)
    print(f"Answer vocab: {len(top_answers)} classes")

    # Load test features (indices 100-199)
    vis_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Image.pkl", map_location="cpu")
    txt_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Text.pkl",  map_location="cpu")
    vis_test = vis_test[100:200].to(device)
    txt_test = txt_test[100:200].to(device)
    print(f"Test features: {vis_test.shape}")

    # Load model
    model = CLIPAnswerClassifier(
        feat_dim=512, hidden_dim=hidden_dim,
        num_answers=len(top_answers), dropout=0.3,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Predict
    all_pred_text = []
    with torch.no_grad():
        logits    = model(vis_test, txt_test)   # [100, K]
        pred_idxs = logits.argmax(dim=-1)       # [100]
        for idx in pred_idxs:
            all_pred_text.append(top_answers[idx.item()])

    assert len(all_pred_text) == 100

    # Encode predictions as tensor for submission
    # Each answer encoded as its class index (1D tensor of length 100)
    pred_tensor = pred_idxs.cpu()
    out_path = Path(f"/content/{args.name}_challenge{args.challenge}.pkl")
    torch.save(pred_tensor, out_path)

    # Human-readable answers
    out_txt = Path(f"/content/{args.name}_challenge{args.challenge}_answers.txt")

    # Load test annotations for printing questions
    ann_base = DATA_ROOT / "Annotations"
    if not (ann_base / "test.json").exists():
        ann_base = next(ann_base.iterdir())
    with open(ann_base / "test.json") as f:
        all_test = json.load(f)
    test_anns = all_test[100:200]

    with open(out_txt, "w") as f:
        for i, (ann, pred) in enumerate(zip(test_anns, all_pred_text)):
            f.write(f"[{100+i}] Q: {ann['question']}\n")
            f.write(f"       Pred: {pred}\n\n")

    print(f"\nSaved predictions to: {out_path}")
    print(f"Saved text answers to: {out_txt}")
    print("\nSample predictions:")
    for i in range(min(5, len(all_pred_text))):
        print(f"  [{100+i}] {test_anns[i]['question']} → {all_pred_text[i]}")


if __name__ == "__main__":
    main()
