import json
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image


# Basic tokenizer: lowercase + whitespace split
def simple_tokenize(text):
    return text.lower().strip().split()


# Build a vocabulary mapping words -> integer ids
def build_vocab(questions, min_freq=1, specials=("<pad>", "<unk>")):
    counter = Counter()
    for q in questions:
        counter.update(simple_tokenize(q))
    vocab = {}
    idx = 0
    for token in specials:
        vocab[token] = idx
        idx += 1
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1
    return vocab


# Build an answer vocabulary with <pad>, <unk>, <sos>, <eos> specials
def build_answer_vocab(annotations, min_freq=1):
    specials = ("<pad>", "<unk>", "<sos>", "<eos>")
    counter = Counter()
    for ann in annotations:
        for ans_obj in ann.get("answers", []):
            answer = ans_obj.get("answer", "").strip()
            if answer:
                counter.update(simple_tokenize(answer))
    vocab = {}
    for i, tok in enumerate(specials):
        vocab[tok] = i
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


# VizWiz official accuracy: min(# annotators who match / 3, 1)
def vizwiz_accuracy(pred_text, answers):
    if not answers or not pred_text.strip():
        return 0.0
    pred = pred_text.strip().lower()
    count = sum(1 for a in answers if a.get("answer", "").strip().lower() == pred)
    return min(count / 3, 1.0)


# Pick the most common answer from the annotator list (majority vote)
def get_majority_answer(answers):
    if not answers:
        return ""
    counts = Counter(a["answer"].strip().lower() for a in answers if a.get("answer"))
    return counts.most_common(1)[0][0] if counts else ""


# Encode an answer string into a fixed-length tensor with <sos>/<eos> framing
def encode_answer(text, vocab, max_len):
    tokens = simple_tokenize(text)
    ids = [vocab["<sos>"]]
    ids += [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    ids += [vocab["<eos>"]]
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [vocab["<eos>"]]
    else:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# Convert a question string into a fixed-length tensor of token ids
def encode_text(text, vocab, max_len):
    tokens = simple_tokenize(text)

    # map tokens to vocab ids (unknown words -> <unk>)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    # truncate or pad to fixed length
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [vocab["<pad>"]] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


class VizWizBinaryDataset(Dataset):
    """
    PyTorch Dataset for VizWiz answerability task.

    Each sample returns:
        image  : Tensor [3, H, W]
        tokens : Tensor [T] (encoded question)
        label  : Tensor [1] (0 = not answerable, 1 = answerable)
    """

    def __init__(
        self,
        annotations,
        image_dir,
        vocab,
        max_len=20,
        transform=None,
    ):
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        # total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]

        # image filename from annotation
        image_path = self.image_dir / sample["image"]

        # question text
        question = sample["question"]

        # binary label (answerable / not answerable)
        label = float(sample["answerable"])

        # load image
        image = Image.open(image_path).convert("RGB")

        # apply preprocessing (resize, tensor conversion, etc.)
        if self.transform is not None:
            image = self.transform(image)

        # convert question -> token ids
        tokens = encode_text(question, self.vocab, self.max_len)

        # BCEWithLogitsLoss expects float labels
        label = torch.tensor([label], dtype=torch.float32)

        return {
            "image": image,
            "tokens": tokens,
            "label": label,
            "question": question,
            "image_name": sample["image"],
        }


class VizWizAnswerDataset(Dataset):
    """
    PyTorch Dataset for VizWiz answer generation task (Challenge 2/4).

    Each sample returns:
        image       : Tensor [3, H, W]
        q_tokens    : Tensor [Q] (encoded question)
        ans_tokens  : Tensor [A] (encoded answer with <sos>/<eos>)
        answer_text : str (ground truth majority answer, for evaluation)
    """

    def __init__(
        self,
        annotations,
        image_dir,
        q_vocab,
        ans_vocab,
        q_max_len=20,
        ans_max_len=12,
        transform=None,
    ):
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.q_vocab = q_vocab
        self.ans_vocab = ans_vocab
        self.q_max_len = q_max_len
        self.ans_max_len = ans_max_len
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image = Image.open(self.image_dir / sample["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        q_tokens = encode_text(sample["question"], self.q_vocab, self.q_max_len)

        # Use majority vote answer as training target
        answer_text = get_majority_answer(sample.get("answers", []))
        ans_tokens = encode_answer(answer_text, self.ans_vocab, self.ans_max_len)

        return {
            "image":       image,
            "q_tokens":    q_tokens,
            "ans_tokens":  ans_tokens,
            "answer_text": answer_text,
            "answers":     sample.get("answers", []),   # raw list for VizWiz accuracy
            "image_name":  sample["image"],
        }