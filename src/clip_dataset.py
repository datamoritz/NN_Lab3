"""
Dataset and vocabulary utilities for CLIP-based models (Challenges 3 & 4).
Features are pre-extracted 512-dim vectors — no image loading required.
"""
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


def get_majority_answer(answers):
    if not answers:
        return ""
    counts = Counter(a["answer"].strip().lower() for a in answers if a.get("answer"))
    return counts.most_common(1)[0][0] if counts else ""


def build_clip_answer_vocab(annotations, top_k=1000):
    """
    Build a closed answer vocabulary from training annotations.
    'unanswerable' is always included and treated as a valid class.
    Returns: {answer_str: class_idx}, list of answer strings indexed by class
    """
    # Count majority-vote answers
    counter = Counter()
    for ann in annotations:
        ans = get_majority_answer(ann.get("answers", []))
        if ans:
            counter[ans] += 1

    # Take top-K (includes "unanswerable" naturally since it's the most common)
    top_answers = [ans for ans, _ in counter.most_common(top_k)]

    # Build vocab dict
    vocab = {ans: i for i, ans in enumerate(top_answers)}
    return vocab, top_answers


def vizwiz_accuracy(pred_text, answers):
    if not answers or not pred_text.strip():
        return 0.0
    pred = pred_text.strip().lower()
    count = sum(1 for a in answers if a.get("answer", "").strip().lower() == pred)
    return min(count / 3, 1.0)


class CLIPBinaryDataset(Dataset):
    """
    Dataset for Challenge 3 (binary classification) using CLIP features.
    Each sample returns:
        vis_feat: Tensor [512]
        txt_feat: Tensor [512]
        label:    Tensor [1]  (1=answerable, 0=unanswerable)
    """
    def __init__(self, annotations, orig_indices, vis_feats, txt_feats):
        self.annotations  = annotations
        self.orig_indices = orig_indices
        self.vis_feats    = vis_feats   # (N_total, 512)
        self.txt_feats    = txt_feats   # (N_total, 512)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        orig_idx = self.orig_indices[idx]
        ann      = self.annotations[idx]
        vis = self.vis_feats[orig_idx].float()
        txt = self.txt_feats[orig_idx].float()
        label = torch.tensor([float(ann["answerable"])], dtype=torch.float32)
        return {"vis": vis, "txt": txt, "label": label,
                "answers": ann.get("answers", []),
                "image_name": ann.get("image", "")}


class CLIPAnswerDataset(Dataset):
    """
    Dataset for Challenge 4 (answer classification) using CLIP features.
    Each sample returns:
        vis_feat:     Tensor [512]
        txt_feat:     Tensor [512]
        answer_idx:   int    (class index of majority-vote answer)
        answer_text:  str    (for evaluation)
        answers:      list   (all 10 annotator answers, for VizWiz scoring)
    """
    def __init__(self, annotations, orig_indices, vis_feats, txt_feats,
                 answer_vocab, top_answers):
        self.annotations  = annotations
        self.orig_indices = orig_indices
        self.vis_feats    = vis_feats
        self.txt_feats    = txt_feats
        self.answer_vocab = answer_vocab
        self.top_answers  = top_answers
        self.unk_idx      = len(top_answers)   # index for answers outside vocab

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        orig_idx     = self.orig_indices[idx]
        ann          = self.annotations[idx]
        vis = self.vis_feats[orig_idx].float()
        txt = self.txt_feats[orig_idx].float()

        answer_text  = get_majority_answer(ann.get("answers", []))
        answer_idx   = self.answer_vocab.get(answer_text, self.unk_idx)

        return {
            "vis":         vis,
            "txt":         txt,
            "answer_idx":  answer_idx,
            "answer_text": answer_text,
            "answers":     ann.get("answers", []),
            "image_name":  ann.get("image", ""),
        }
