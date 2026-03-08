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

    # count word frequencies across all questions
    for q in questions:
        counter.update(simple_tokenize(q))

    vocab = {}
    idx = 0

    # add special tokens first
    for token in specials:
        vocab[token] = idx
        idx += 1

    # add words that appear at least min_freq times
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1

    return vocab


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
            "question": question,      # optional, useful for debugging
            "image_name": sample["image"],
        }