# =============================================================================
# knoedler_moritz_code_bundle.py
# Lab 3 — VizWiz Visual Question Answering
# Author: Moritz Knödler
# =============================================================================
# This file bundles all source files in one place for PDF submission.
# Files included (in order):
#   1.  src/utils.py
#   2.  src/dataset.py
#   3.  src/model.py                  (binary classifier model — CNN+Transformer)
#   4.  src/binary_classifier.py      (same model, standalone copy)
#   5.  src/models/binary_classifier.py
#   6.  src/answer_generator.py       (seq2seq answer generator)
#   7.  src/models/answer_generator.py
#   8.  src/train.py                  (Challenge 1/3 — 50% training data)
#   9.  src/train_75.py               (Challenge 1/3 — 75% training data)
#   10. src/train_100.py              (Challenge 1/3 — 100% training data)
#   11. src/train_generator.py        (Challenge 2/4 answer generator training)
#   12. src/clip_models.py            (CLIP-based models Ch 3/4)
#   13. src/clip_dataset.py           (CLIP dataset utilities)
#   14. src/train_clip_binary.py      (Challenge 3 — 50% training data)
#   15. src/train_clip_binary_75.py   (Challenge 3 — 75% training data)
#   16. src/train_clip_binary_100.py  (Challenge 3 — 100% training data)
#   17. src/train_clip_answer.py      (Challenge 4 CLIP answer classifier)
#   18. src/predict_challenge1.py
#   19. src/predict_challenge2.py
#   20. src/predict_challenge3.py
#   21. src/predict_challenge4.py
#   22. src/eval_gated.py
# =============================================================================


# =============================================================================
# FILE: src/utils.py
# =============================================================================
"""Utility functions for Lab 3."""


def helper_function():
    """Placeholder utility function."""
    pass


# =============================================================================
# FILE: src/dataset.py
# =============================================================================
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


# =============================================================================
# FILE: src/model.py
# =============================================================================

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    CNN image encoder with four convolutional blocks + spatial patch output.

    Instead of collapsing the feature map to a single vector with (1,1) pooling,
    we use AdaptiveAvgPool2d((7,7)) to produce a 7x7 = 49 spatial patch grid.
    Each patch is projected to embed_dim, giving a sequence of 49 visual tokens
    that CrossAttentionFusion can use for fine-grained spatial attention.
    """

    def __init__(self, out_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            # --- Block 1: 3 -> 32 channels ---
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # spatial: H/2, W/2

            # --- Block 2: 32 -> 64 channels ---
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # spatial: H/4, W/4

            # --- Block 3: 64 -> 128 channels ---
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),            # spatial: H/8, W/8

            # --- Block 4: 128 -> 256 channels ---
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),            # spatial: H/16, W/16

            # Reduce to fixed 7x7 grid regardless of input resolution.
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # Project each of the 16 patches from 256 -> out_dim.
        # Dropout regularises patch representations before cross-attention.
        self.patch_proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

        # Learnable positional embeddings for each of the 49 spatial patches.
        # Without these, the cross-attention has no way to distinguish
        # top-left from bottom-right — spatial layout is completely lost.
        self.patch_pos = nn.Embedding(49, out_dim)

    def forward(self, x):
        x = self.cnn(x)                          # [B, 256, 4, 4]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)               # [B, 4, 4, 256]
        x = x.reshape(B, H * W, C)              # [B, 16, 256]
        x = self.patch_proj(x)                   # [B, 16, out_dim]

        # Add spatial positional signal to each patch token.
        positions = torch.arange(H * W, device=x.device)   # [16]
        x = x + self.patch_pos(positions)                   # [B, 16, out_dim]
        return x                                            # sequence of 16 patch tokens


class TextEncoder(nn.Module):
    """
    Text encoder using token embeddings + Transformer self-attention.

    Unlike simple mean pooling, the Transformer lets each token attend to
    every other token in the sequence, capturing contextual relationships
    (e.g. "is this readable?" vs "what color is this?").

    Returns both the full token sequence (used for cross-attention in fusion)
    and a mean-pooled summary vector over non-padding positions.
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=4,
                 num_layers=2, max_len=20, dropout=0.1):
        super().__init__()

        # Token embeddings; padding_idx=0 means <pad> tokens never
        # contribute gradients and stay as zero vectors.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Learnable positional embeddings encode word order into the sequence.
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # Stack of Transformer encoder layers (self-attention + FFN).
        # dim_feedforward = 4 * embed_dim is the standard ratio from "Attention is All You Need".
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,           # expect [B, T, E] not [T, B, E]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, tokens):
        """
        Args:
            tokens: [B, T]  integer token ids (0 = <pad>)
        Returns:
            seq:    [B, T, E]  contextual token representations
            pooled: [B, E]     mean-pooled over non-padding positions
        """
        B, T = tokens.shape

        # Position indices [0, 1, ..., T-1] broadcast over the batch.
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # [1, T]

        # Combine token and positional embeddings.
        x = self.embedding(tokens) + self.pos_embedding(positions)      # [B, T, E]
        x = self.dropout(x)

        # Padding mask: True where token == <pad>; Transformer ignores these positions.
        padding_mask = (tokens == 0)  # [B, T]

        # Self-attention: every token attends to every other (non-padding) token.
        seq = self.transformer(x, src_key_padding_mask=padding_mask)    # [B, T, E]

        # Mean pool only over real (non-padding) positions to get a fixed-size summary.
        mask = (~padding_mask).unsqueeze(-1).float()                     # [B, T, 1]
        pooled = (seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # [B, E]

        return seq, pooled


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: text tokens query the spatial image patch sequence.

    The question (text) acts as the intent: each text token attends over the
    16 image patches to find visually relevant regions.
    — e.g. "expired?" looks for sharp label text; "color?" looks for clear hue regions.

    The attended text representation (image-aware) is mean-pooled and concatenated
    with the pooled text summary, then projected to the shared embedding dimension.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        # Multi-head attention: text tokens query the image patch sequence.
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Project the concatenated [attended_txt || pooled_txt] back to embed_dim.
        # LayerNorm stabilizes the fused representation before the classifier.
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, img_patches, txt_feat, txt_seq):
        """
        Args:
            img_patches: [B, P, E]  spatial image patch tokens (keys / values)
            txt_feat:    [B, E]     pooled text feature
            txt_seq:     [B, T, E]  full text token sequence (queries)
        Returns:
            fused: [B, E]
        """
        # Text tokens query the image patch sequence.
        # Each text token learns which image regions are relevant to it.
        txt_ctx, _ = self.text_to_img_attn(
            query=txt_seq,       # [B, T, E]
            key=img_patches,     # [B, P, E]
            value=img_patches,   # [B, P, E]
        )                        # [B, T, E]

        # Mean-pool the image-aware text tokens -> single summary vector.
        txt_ctx_pooled = txt_ctx.mean(dim=1)    # [B, E]

        # Concatenate image-informed text rep with the original pooled text.
        fused = torch.cat([txt_ctx_pooled, txt_feat], dim=1)   # [B, 2E]
        return self.proj(fused)                                  # [B, E]


class VizWizBinaryClassifier(nn.Module):
    """
    Multi-modal binary classifier for VizWiz answerability prediction.

    Pipeline:
        1. ImageEncoder        — CNN extracts 16 spatial patch tokens [B, 16, E]
        2. TextEncoder         — Transformer extracts contextual text features (self-attention)
        3. CrossAttentionFusion — text tokens query image patches (text intent probes image)
        4. MLP classifier head  — maps fused feature to a single logit

    Use nn.BCEWithLogitsLoss for training (logit output, no sigmoid here).
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        max_len=20,
        dropout=0.3,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(out_dim=embed_dim)

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Two-layer MLP with Dropout between layers for regularization.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, images, tokens):
        """
        Args:
            images: [B, 3, H, W]
            tokens: [B, T]
        Returns:
            logits: [B, 1]  (raw logits; apply .sigmoid() for probabilities)
        """
        # Encode each modality independently.
        img_patches = self.image_encoder(images)         # [B, 16, E]
        txt_seq, txt_feat = self.text_encoder(tokens)    # [B, T, E], [B, E]

        # Fuse: text tokens query image patches.
        fused = self.fusion(img_patches, txt_feat, txt_seq)  # [B, E]

        # Produce a single classification logit.
        return self.classifier(fused)                         # [B, 1]


# =============================================================================
# FILE: src/answer_generator.py  (and src/models/answer_generator.py)
# =============================================================================
"""
answer_generator.py — VizWiz answer generation model (Challenge 2/4)

Architecture:
    1. ImageEncoder     — 4-block CNN → 7×7 = 49 spatial patch tokens
    2. QuestionEncoder  — Transformer encoder with self-attention
    3. AnswerDecoder    — Transformer decoder
                          - self-attention over previously generated tokens
                          - cross-attention over image patches (visual grounding)
                          - cross-attention over question encoding (intent)
    4. Output head      — linear projection to answer vocabulary

Training: teacher forcing with nn.CrossEntropyLoss (ignore <pad> tokens).
Inference: greedy decoding token-by-token until <eos> or max_len.
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# Shared encoder components (identical to binary_classifier.py)
# -----------------------------------------------------------------------

class ImageEncoder_Gen(nn.Module):
    """4-block CNN producing 49 spatial patch tokens [B, 49, E]."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.patch_proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )
        self.patch_pos = nn.Embedding(49, out_dim)

    def forward(self, x):
        x = self.cnn(x)                     # [B, 256, 7, 7]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 49, 256]
        x = self.patch_proj(x)
        x = x + self.patch_pos(torch.arange(H * W, device=x.device))
        return x                            # [B, 49, E]


class QuestionEncoder(nn.Module):
    """Transformer encoder over question tokens."""

    def __init__(self, vocab_size, embed_dim=256, num_heads=4,
                 num_layers=2, max_len=20, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.dropout(self.embedding(tokens) + self.pos_embedding(positions))
        padding_mask = (tokens == 0)
        return self.transformer(x, src_key_padding_mask=padding_mask)  # [B, T, E]


# -----------------------------------------------------------------------
# Answer decoder
# -----------------------------------------------------------------------

class AnswerDecoder(nn.Module):
    """
    Transformer decoder that generates answer tokens autoregressively.

    Each decoder layer performs:
        1. Masked self-attention  — attend to previously generated answer tokens
        2. Cross-attention        — attend over image patches (visual grounding)
        3. Cross-attention        — attend over question encoding (intent)
        4. Feed-forward network

    We implement this by stacking standard TransformerDecoderLayers and
    feeding them the image+question context as memory via a simple
    concatenation of both sequences.
    """

    def __init__(self, ans_vocab_size, embed_dim=256, num_heads=4,
                 num_layers=2, ans_max_len=12, dropout=0.1):
        super().__init__()
        self.ans_max_len = ans_max_len

        # Answer token embeddings + positional
        self.embedding = nn.Embedding(ans_vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(ans_max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Decoder: attends over concatenated [image_patches || question_tokens] as memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project decoder output to answer vocabulary
        self.output_proj = nn.Linear(embed_dim, ans_vocab_size)

    def forward(self, ans_tokens, img_patches, q_enc):
        """
        Args:
            ans_tokens:  [B, A]    answer token ids (teacher-forced during training)
            img_patches: [B, 49, E] image patch tokens
            q_enc:       [B, T, E]  question encoder output
        Returns:
            logits: [B, A, vocab_size]
        """
        B, A = ans_tokens.shape
        positions = torch.arange(A, device=ans_tokens.device).unsqueeze(0)
        x = self.dropout(self.embedding(ans_tokens) + self.pos_embedding(positions))

        # Causal mask: each position can only attend to previous tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(A, device=ans_tokens.device)

        # Concatenate image patches + question encoding as a single memory sequence
        memory = torch.cat([img_patches, q_enc], dim=1)  # [B, 49+T, E]

        out = self.transformer(x, memory, tgt_mask=causal_mask)  # [B, A, E]
        return self.output_proj(out)                              # [B, A, vocab_size]


# -----------------------------------------------------------------------
# Top-level model
# -----------------------------------------------------------------------

class VizWizAnswerGenerator(nn.Module):
    """
    Multi-modal answer generator for VizWiz (Challenge 2/4).

    Training:  teacher forcing — pass ground-truth answer tokens shifted right.
    Inference: greedy_decode() generates token-by-token until <eos> or max_len.
    """

    def __init__(
        self,
        q_vocab_size,
        ans_vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        q_max_len=20,
        ans_max_len=12,
        dropout=0.3,
        sos_idx=2,
        eos_idx=3,
    ):
        super().__init__()
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.ans_max_len = ans_max_len

        self.image_encoder = ImageEncoder_Gen(out_dim=embed_dim)
        self.question_encoder = QuestionEncoder(
            vocab_size=q_vocab_size, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_layers,
            max_len=q_max_len, dropout=dropout,
        )
        self.answer_decoder = AnswerDecoder(
            ans_vocab_size=ans_vocab_size, embed_dim=embed_dim,
            num_heads=num_heads, num_layers=num_layers,
            ans_max_len=ans_max_len, dropout=dropout,
        )

    def forward(self, images, q_tokens, ans_tokens):
        """
        Teacher-forced forward pass for training.

        Args:
            images:     [B, 3, H, W]
            q_tokens:   [B, Q]  question token ids
            ans_tokens: [B, A]  answer token ids (including <sos>, excluding last token)
        Returns:
            logits: [B, A, ans_vocab_size]
        """
        img_patches = self.image_encoder(images)      # [B, 49, E]
        q_enc = self.question_encoder(q_tokens)        # [B, Q, E]
        return self.answer_decoder(ans_tokens, img_patches, q_enc)

    @torch.no_grad()
    def greedy_decode(self, images, q_tokens):
        """
        Greedy autoregressive decoding for inference.

        Returns:
            preds: [B, ans_max_len]  predicted token ids (0-padded after <eos>)
        """
        self.eval()
        B = images.size(0)
        img_patches = self.image_encoder(images)
        q_enc = self.question_encoder(q_tokens)

        # Start with <sos> token for every sample in the batch
        generated = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)

        for _ in range(self.ans_max_len - 1):
            logits = self.answer_decoder(generated, img_patches, q_enc)  # [B, t, V]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            finished |= (next_token.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        # Pad everything after the first <eos> with <pad> (id=0)
        output = torch.zeros(B, self.ans_max_len, dtype=torch.long, device=images.device)
        for b in range(B):
            seq = generated[b, 1:]  # strip <sos>
            eos_positions = (seq == self.eos_idx).nonzero(as_tuple=True)[0]
            end = eos_positions[0].item() if len(eos_positions) > 0 else len(seq)
            output[b, :end] = seq[:end]

        return output  # [B, ans_max_len]


# =============================================================================
# FILE: src/clip_models.py
# =============================================================================
"""
CLIP-based models for Challenges 3 (binary) and 4 (answer classification).
Input: pre-extracted CLIP features (512-dim image + 512-dim text).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPBinaryClassifier(nn.Module):
    """
    Challenge 3: binary answerability classifier on CLIP features.
    Input: L2-normalized [vis_feat || txt_feat] (1024-dim)
    Output: single logit (apply sigmoid for probability)
    """
    def __init__(self, feat_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, vis_feat, txt_feat):
        vis = F.normalize(vis_feat, dim=-1)
        txt = F.normalize(txt_feat, dim=-1)
        x = torch.cat([vis, txt], dim=-1)
        return self.net(x)


class CLIPAnswerClassifier(nn.Module):
    """
    Challenge 4: multi-class answer classifier on CLIP features.
    Treats answer generation as closed-set classification over top-K answers.
    Input: L2-normalized [vis_feat || txt_feat] (1024-dim)
    Output: logits over K answer classes
    """
    def __init__(self, feat_dim=512, hidden_dim=512, num_answers=1000, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_answers),
        )

    def forward(self, vis_feat, txt_feat):
        vis = F.normalize(vis_feat, dim=-1)
        txt = F.normalize(txt_feat, dim=-1)
        x = torch.cat([vis, txt], dim=-1)
        return self.net(x)


# =============================================================================
# FILE: src/clip_dataset.py
# =============================================================================
"""
Dataset and vocabulary utilities for CLIP-based models (Challenges 3 & 4).
Features are pre-extracted 512-dim vectors — no image loading required.
"""
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


def get_majority_answer_clip(answers):
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
        ans = get_majority_answer_clip(ann.get("answers", []))
        if ans:
            counter[ans] += 1

    # Take top-K (includes "unanswerable" naturally since it's the most common)
    top_answers = [ans for ans, _ in counter.most_common(top_k)]

    # Build vocab dict
    vocab = {ans: i for i, ans in enumerate(top_answers)}
    return vocab, top_answers


def vizwiz_accuracy_clip(pred_text, answers):
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

        answer_text  = get_majority_answer_clip(ann.get("answers", []))
        answer_idx   = self.answer_vocab.get(answer_text, self.unk_idx)

        return {
            "vis":         vis,
            "txt":         txt,
            "answer_idx":  answer_idx,
            "answer_text": answer_text,
            "answers":     ann.get("answers", []),
            "image_name":  ann.get("image", ""),
        }


# =============================================================================
# FILE: src/train.py  — Challenge 1/3, 50% training data (10k samples)
# =============================================================================
"""
train.py — VizWiz binary answerability classifier
Designed to run on Google Colab with data stored as zip files on Google Drive.

─────────────────────────────────────────────────────────────
COLAB SETUP  (run these in a notebook cell before importing)
─────────────────────────────────────────────────────────────
# 1. Mount Drive and clone the repo
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo (skip if already cloned)
!git clone https://github.com/datamoritz/NN_Lab3.git /content/NN_Lab3
%cd /content/NN_Lab3

# 3. Unzip data from Drive into /content/data/
import zipfile, pathlib
DATA_DIR = pathlib.Path("/content/data")
DATA_DIR.mkdir(exist_ok=True)

DRIVE_DATA = pathlib.Path("/content/drive/MyDrive/NN Lab3/Data")
for zip_name, dest in [
    ("train.zip",       DATA_DIR / "train"),
    ("val.zip",         DATA_DIR / "val"),
    ("test.zip",        DATA_DIR / "test"),
    ("Annotations.zip", DATA_DIR / "Annotations"),
]:
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(DRIVE_DATA / zip_name) as zf:
        zf.extractall(dest)
    print(f"Extracted {zip_name}")

# 4. Run training
!python src/train.py
─────────────────────────────────────────────────────────────
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Allow `python src/train.py` from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from src.dataset import VizWizBinaryDataset, build_vocab
# from src.model import VizWizBinaryClassifier


def find_image_dir(base: Path) -> Path:
    """Return the directory that actually contains .jpg files.
    Handles zips that extract into a subdirectory (e.g. train/train/*.jpg)."""
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")

# -------------------------------------------------------
# Config
# -------------------------------------------------------

# Paths — point to wherever data was extracted on Colab (or locally)
DATA_ROOT      = Path("/content/data")
TRAIN_IMAGE_DIR = DATA_ROOT / "train"
VAL_IMAGE_DIR   = DATA_ROOT / "val"
_ann_base       = DATA_ROOT / "Annotations"
# Handle zips that extract into a subdirectory
# if not (_ann_base / "train.json").exists():
#     _ann_base = next(_ann_base.iterdir())
TRAIN_ANN_PATH  = _ann_base / "train.json"
VAL_ANN_PATH    = _ann_base / "val.json"

# Where to save the best checkpoint
CHECKPOINT_PATH = Path("/content/best_model.pt")

# -------------------------------------------------------
# Fast mode — set True for quick hyperparameter search:
#   - 128x128 images  (~4x fewer pixels to process)
#   - 5000 train / 2000 val samples
#   - 10 epochs
# Set False for the final full-quality run.
# -------------------------------------------------------
FAST_MODE = False

IMG_SIZE          = 128        if FAST_MODE else 224
MAX_TRAIN_SAMPLES = 5_000      if FAST_MODE else 10_000
MAX_VAL_SAMPLES   = 2_000      if FAST_MODE else None
NUM_EPOCHS        = 10         if FAST_MODE else 20
MAX_LEN           = 20

# DataLoader
BATCH_SIZE  = 256
NUM_WORKERS = 4

# Model
EMBED_DIM  = 256
NUM_HEADS  = 4
NUM_LAYERS = 2
DROPOUT    = 0.3

# Training
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
LABEL_SMOOTH  = 0.05
# Override the auto-computed pos_weight (neg/pos ratio ~0.37).
# Higher value penalises false positives more → improves TNR.
# Set to None to use the automatic class-balance ratio.
POS_WEIGHT_OVERRIDE = None  # use auto class-balance ratio (neg/pos)

# -------------------------------------------------------
# Device
# -------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# [Training loop identical to train_75.py / train_100.py — see below]
# train_transform, val_transform, vocab, datasets, dataloaders,
# model, criterion, optimizer, scheduler, training loop, threshold scan
# — all identical except CHECKPOINT_PATH and MAX_TRAIN_SAMPLES.


# =============================================================================
# FILE: src/train_75.py  — Challenge 1/3, 75% training data (15,392 samples)
# =============================================================================
"""train_75.py — same as train.py but MAX_TRAIN_SAMPLES=15_392, saves best_model_75.pt"""

# Key differences vs train.py:
# MAX_TRAIN_SAMPLES = 15_392
# CHECKPOINT_PATH   = Path("/content/best_model_75.pt")
# threshold saved to /content/best_threshold_75.pt


# =============================================================================
# FILE: src/train_100.py  — Challenge 1/3, 100% training data (all samples)
# =============================================================================
"""train_100.py — same as train.py but MAX_TRAIN_SAMPLES=None (all), saves best_model_100.pt"""

# Key differences vs train.py:
# MAX_TRAIN_SAMPLES = None   (use all available training images)
# CHECKPOINT_PATH   = Path("/content/best_model_100.pt")
# threshold saved to /content/best_threshold_100.pt


# =============================================================================
# FILE: src/train_generator.py  — Answer Generator training (Challenge 2/4)
# =============================================================================
"""
train_generator.py — Training script for VizWiz answer generation (Challenge 2/4)

COLAB SETUP (same data extraction as train_binary.py — skip if already done):
    %cd /content/NN_Lab3
    !git pull
    !python src/train_generator.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

_root = Path(__file__).resolve().parent.parent  # NN_Lab3/
_src  = Path(__file__).resolve().parent          # NN_Lab3/src/
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

# from src.dataset import (
#     VizWizAnswerDataset, build_vocab, build_answer_vocab,
#     get_majority_answer, encode_text, vizwiz_accuracy,
# )
# from answer_generator import VizWizAnswerGenerator


def find_image_dir_gen(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT_GEN       = Path("/content/data")
TRAIN_IMAGE_DIR_GEN = DATA_ROOT_GEN / "train"
VAL_IMAGE_DIR_GEN   = DATA_ROOT_GEN / "val"
_ann_base_gen       = DATA_ROOT_GEN / "Annotations"
if not (_ann_base_gen / "train.json").exists():
    _ann_base_gen = next(_ann_base_gen.iterdir())
TRAIN_ANN_PATH_GEN  = _ann_base_gen / "train.json"
VAL_ANN_PATH_GEN    = _ann_base_gen / "val.json"

CHECKPOINT_PATH_GEN  = Path("/content/best_generator.pt")
THRESHOLD_PATH_GEN   = Path("/content/best_threshold.pt")   # from binary training

FAST_MODE_GEN = False

IMG_SIZE_GEN          = 128   if FAST_MODE_GEN else 224
MAX_TRAIN_SAMPLES_GEN = 5_000 if FAST_MODE_GEN else 10_000
MAX_VAL_SAMPLES_GEN   = 2_000 if FAST_MODE_GEN else None
NUM_EPOCHS_GEN        = 10    if FAST_MODE_GEN else 20

Q_MAX_LEN   = 20
ANS_MAX_LEN = 12    # most VizWiz answers are 1–5 words; 12 tokens is safe

BATCH_SIZE_GEN   = 128
NUM_WORKERS_GEN  = 4
EMBED_DIM_GEN    = 256
NUM_HEADS_GEN    = 4
NUM_LAYERS_GEN   = 2
DROPOUT_GEN      = 0.3
LR_GEN           = 1e-3
WEIGHT_DECAY_GEN = 1e-4
WARMUP_EPOCHS_GEN = 2

# -------------------------------------------------------
# Device
# -------------------------------------------------------
if torch.cuda.is_available():
    DEVICE_GEN = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE_GEN = torch.device("mps")
else:
    DEVICE_GEN = torch.device("cpu")

# [Full training loop — see train_generator.py in repo for complete code]
# Highlights:
#   - Teacher forcing: decoder_input = ans_tokens[:, :-1], target = ans_tokens[:, 1:]
#   - Down-weights "unanswerable" token (class_weights[unanswerable_idx] = 0.3)
#   - CrossEntropyLoss with ignore_index=PAD_IDX and label_smoothing=0.1
#   - Separate LRs: image_encoder/decoder at LR, question_encoder at LR*0.3
#   - Mixed precision + gradient clipping (max_norm=1.0)
#   - Saves best checkpoint by VizWiz accuracy (not loss)


# =============================================================================
# FILE: src/train_clip_binary.py  — CLIP binary, 50% data (10k samples)
# =============================================================================
"""
train_clip_binary.py — Train CLIP-based binary classifier for Challenge 3.

COLAB USAGE:
    !python src/train_clip_binary.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

# from clip_models import CLIPBinaryClassifier
# from clip_dataset import CLIPBinaryDataset, vizwiz_accuracy

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT_CB    = Path("/content/data")
FEAT_ROOT_CB    = Path("/content/clip_features")

ANN_PATH_CB     = DATA_ROOT_CB / "Annotations"
if not (ANN_PATH_CB / "train.json").exists():
    ANN_PATH_CB = next(ANN_PATH_CB.iterdir())

CHECKPOINT_PATH_CB   = Path("/content/best_clip_binary.pt")
THRESHOLD_PATH_CB    = Path("/content/best_clip_binary_threshold.pt")

MAX_TRAIN_SAMPLES_CB = 10_000
MAX_VAL_SAMPLES_CB   = None

HIDDEN_DIM_CB  = 256
DROPOUT_CB     = 0.3
BATCH_SIZE_CB  = 512
NUM_EPOCHS_CB  = 30
LR_CB          = 1e-3
WEIGHT_DECAY_CB = 1e-4
LABEL_SMOOTH_CB = 0.05

# -------------------------------------------------------
# Device
# -------------------------------------------------------
device_cb = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Load features and annotations
# -------------------------------------------------------
# vis_train = torch.load(FEAT_ROOT_CB / "VizWiz_train_CLIP_Image.pkl", map_location="cpu")
# txt_train = torch.load(FEAT_ROOT_CB / "VizWiz_train_CLIP_Text.pkl",  map_location="cpu")
# vis_val   = torch.load(FEAT_ROOT_CB / "VizWiz_val_CLIP_Image.pkl",   map_location="cpu")
# txt_val   = torch.load(FEAT_ROOT_CB / "VizWiz_val_CLIP_Text.pkl",    map_location="cpu")

# [Full training loop: BCEWithLogitsLoss + pos_weight + CosineAnnealingLR
#  + threshold scan on val set. See train_clip_binary.py in repo.]


# =============================================================================
# FILE: src/train_clip_binary_75.py  — CLIP binary, 75% data (15,392 samples)
# =============================================================================
"""train_clip_binary_75.py — same as train_clip_binary.py but MAX_TRAIN_SAMPLES=15_392"""

# Key differences vs train_clip_binary.py:
# MAX_TRAIN_SAMPLES = 15_392
# CHECKPOINT_PATH   = Path("/content/best_clip_binary_75.pt")
# THRESHOLD_PATH    = Path("/content/best_clip_binary_threshold_75.pt")


# =============================================================================
# FILE: src/train_clip_binary_100.py  — CLIP binary, 100% data (20,523 samples)
# =============================================================================
"""train_clip_binary_100.py — same as train_clip_binary.py but MAX_TRAIN_SAMPLES=20_523"""

# Key differences vs train_clip_binary.py:
# MAX_TRAIN_SAMPLES = 20_523
# CHECKPOINT_PATH   = Path("/content/best_clip_binary_100.pt")
# THRESHOLD_PATH    = Path("/content/best_clip_binary_threshold_100.pt")


# =============================================================================
# FILE: src/train_clip_answer.py  — CLIP answer classifier (Challenge 4)
# =============================================================================
"""
train_clip_answer.py — Train CLIP-based answer classifier for Challenge 4.

COLAB USAGE:
    !python src/train_clip_answer.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parent.parent
_src  = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_src))

# from clip_models import CLIPAnswerClassifier
# from clip_dataset import CLIPAnswerDataset, build_clip_answer_vocab, vizwiz_accuracy

# -------------------------------------------------------
# Config
# -------------------------------------------------------
DATA_ROOT_CA  = Path("/content/data")
FEAT_ROOT_CA  = Path("/content/clip_features")

ANN_PATH_CA   = DATA_ROOT_CA / "Annotations"
if not (ANN_PATH_CA / "train.json").exists():
    ANN_PATH_CA = next(ANN_PATH_CA.iterdir())

CHECKPOINT_PATH_CA = Path("/content/best_clip_answer.pt")

MAX_TRAIN_SAMPLES_CA = 10_000
MAX_VAL_SAMPLES_CA   = None
TOP_K_ANSWERS     = 1000   # closed answer vocabulary size

HIDDEN_DIM_CA   = 512
DROPOUT_CA      = 0.3
BATCH_SIZE_CA   = 512
NUM_EPOCHS_CA   = 30
LR_CA           = 1e-3
WEIGHT_DECAY_CA = 1e-4
LABEL_SMOOTH_CA = 0.1

# -------------------------------------------------------
# Device
# -------------------------------------------------------
device_ca = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [Full training loop: CrossEntropyLoss with ignore_index=num_classes (for unknown answers),
#  label_smoothing=0.1, CosineAnnealingLR, saves best by VizWiz accuracy.
#  Checkpoint stores model_state + answer_vocab + top_answers + hidden_dim.
#  See train_clip_answer.py in repo for complete code.]


# =============================================================================
# FILE: src/predict_challenge1.py
# =============================================================================
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

# from src.dataset import VizWizBinaryDataset, build_vocab
# from binary_classifier import VizWizBinaryClassifier


def find_image_dir_p1(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


def main_challenge1():
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
    TEST_IMAGE_DIR = find_image_dir_p1(DATA_ROOT / "test")

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
    # Test annotations don't have 'answerable' — add a dummy value so
    # VizWizBinaryDataset doesn't KeyError (label is not used for prediction)
    for ann in test_annotations:
        ann.setdefault("answerable", 0)
    print(f"Test samples (indices 100–199): {len(test_annotations)}")

    # ---- Rebuild vocab from train set ----
    with open(train_ann_path) as f:
        all_train = json.load(f)
    train_available = {p.name for p in find_image_dir_p1(DATA_ROOT / "train").glob("*.jpg")}
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
    main_challenge1()


# =============================================================================
# FILE: src/predict_challenge2.py
# =============================================================================
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

# from src.dataset import VizWizAnswerDataset, build_vocab, build_answer_vocab, encode_answer, encode_text
# from answer_generator import VizWizAnswerGenerator
# from binary_classifier import VizWizBinaryClassifier


def find_image_dir_p2(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


def main_challenge2():
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
    TEST_IMAGE_DIR = find_image_dir_p2(DATA_ROOT / "test")

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
    train_available = {p.name for p in find_image_dir_p2(DATA_ROOT / "train").glob("*.jpg")}
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
    main_challenge2()


# =============================================================================
# FILE: src/predict_challenge3.py
# =============================================================================
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

# from clip_models import CLIPBinaryClassifier


def main_challenge3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--challenge", default="3", choices=["1", "3"])
    parser.add_argument("--checkpoint",  default="/content/best_clip_binary.pt")
    parser.add_argument("--threshold",   default="/content/best_clip_binary_threshold.pt")
    parser.add_argument("--feat_root",   default="/content/clip_features")
    parser.add_argument("--data_root",   default="/content/data")
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_root)
    FEAT_ROOT = Path(args.feat_root)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test features (indices 100-199)
    vis_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Image.pkl", map_location="cpu")
    txt_test = torch.load(FEAT_ROOT / "VizWiz_test_CLIP_Text.pkl",  map_location="cpu")
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
    main_challenge3()


# =============================================================================
# FILE: src/predict_challenge4.py
# =============================================================================
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

# from clip_models import CLIPAnswerClassifier
# from clip_dataset import vizwiz_accuracy


def main_challenge4():
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
    main_challenge4()


# =============================================================================
# FILE: src/eval_gated.py
# =============================================================================
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

# from src.dataset import (
#     VizWizAnswerDataset, build_vocab, build_answer_vocab,
#     encode_text, vizwiz_accuracy,
# )
# from answer_generator import VizWizAnswerGenerator
# from binary_classifier import VizWizBinaryClassifier


def find_image_dir_eg(base: Path) -> Path:
    if any(base.glob("*.jpg")):
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and any(sub.glob("*.jpg")):
            return sub
    raise FileNotFoundError(f"No .jpg files found under {base}")


DATA_ROOT_EG      = Path("/content/data")
# VAL_IMAGE_DIR_EG  = find_image_dir_eg(DATA_ROOT_EG / "val")
ann_base_eg       = DATA_ROOT_EG / "Annotations"
# if not (ann_base_eg / "val.json").exists():
#     ann_base_eg = next(ann_base_eg.iterdir())

GENERATOR_CHECKPOINT_EG = "/content/best_generator.pt"
BINARY_CHECKPOINT_EG    = "/content/best_model.pt"
THRESHOLD_PATH_EG       = "/content/best_threshold.pt"
MAX_VAL_SAMPLES_EG      = 500   # use first 500 val samples for speed; set None for all

device_eg = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- [Full eval loop] ----
# Loads val annotations, generator checkpoint (with saved vocabs),
# binary classifier (rebuilt vocab from train set), and runs the gated pipeline.
# Reports:
#   - Generator-only VizWiz accuracy
#   - Gated pipeline VizWiz accuracy
#   - Improvement from gating
#
# Key collate_fn: keeps 'answers' (list of dicts) as plain Python list
# to avoid torch default_collate errors.
#
# def collate_fn(batch):
#     from torch.utils.data import default_collate
#     answers = [item.pop("answers") for item in batch]
#     collated = default_collate(batch)
#     collated["answers"] = answers
#     return collated
