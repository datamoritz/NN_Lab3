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

class ImageEncoder(nn.Module):
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

        self.image_encoder = ImageEncoder(out_dim=embed_dim)
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
