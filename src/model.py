
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    CNN image encoder with three convolutional blocks.

    Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
      - BatchNorm stabilizes training by normalizing activations per batch.
      - MaxPool progressively reduces spatial resolution, increasing receptive field.
    Ends with AdaptiveAvgPool to collapse spatial dims to a single vector,
    followed by a Dropout + Linear projection to the shared embedding space.
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

            # Collapse all remaining spatial positions -> [B, 128, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project from CNN output dim to shared embedding dim.
        # Dropout here regularizes the image representation before fusion.
        self.proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)      # [B, 128, 1, 1]
        x = x.flatten(1)     # [B, 128]
        x = self.proj(x)     # [B, out_dim]
        return x


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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
    Cross-attention fusion: image attends over the full text token sequence.

    Instead of simply concatenating fixed-size image and text vectors,
    the image feature (as a single query token) attends over all text positions.
    This lets the model weight each word differently depending on the image context
    — e.g. focus on "readable?" for blurry images vs "color?" for clear ones.

    The resulting text-aware image representation is then concatenated with the
    pooled text feature and projected to the shared embedding dimension.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        # Multi-head attention: image feature queries the full text sequence.
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Project the concatenated [img_ctx || txt_feat] back to embed_dim.
        # LayerNorm stabilizes the fused representation before the classifier.
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, img_feat, txt_feat, txt_seq):
        """
        Args:
            img_feat: [B, E]    pooled image feature (query)
            txt_feat: [B, E]    pooled text feature
            txt_seq:  [B, T, E] full text token sequence (keys / values)
        Returns:
            fused: [B, E]
        """
        # Treat the image feature as a single query token.
        img_query = img_feat.unsqueeze(1)   # [B, 1, E]

        # Image query attends over every text token position.
        img_ctx, _ = self.img_to_text_attn(
            query=img_query,
            key=txt_seq,
            value=txt_seq,
        )                                   # [B, 1, E]
        img_ctx = img_ctx.squeeze(1)        # [B, E]

        # Concatenate the text-informed image rep with the pooled text summary.
        fused = torch.cat([img_ctx, txt_feat], dim=1)   # [B, 2E]
        return self.proj(fused)                          # [B, E]


class VizWizBinaryClassifier(nn.Module):
    """
    Multi-modal binary classifier for VizWiz answerability prediction.

    Pipeline:
        1. ImageEncoder        — deep CNN extracts visual features
        2. TextEncoder         — Transformer extracts contextual text features (self-attention)
        3. CrossAttentionFusion — image attends over text to produce a fused representation
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
        img_feat = self.image_encoder(images)           # [B, E]
        txt_seq, txt_feat = self.text_encoder(tokens)   # [B, T, E], [B, E]

        # Fuse via cross-attention (image queries text sequence).
        fused = self.fusion(img_feat, txt_feat, txt_seq)  # [B, E]

        # Produce a single classification logit.
        return self.classifier(fused)                     # [B, 1]
