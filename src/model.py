
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    CNN image encoder with four convolutional blocks + spatial patch output.

    Instead of collapsing the feature map to a single vector with (1,1) pooling,
    we use AdaptiveAvgPool2d((4,4)) to produce a 4x4 = 16 spatial patch grid.
    Each patch is projected to embed_dim, giving a sequence of 16 visual tokens
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

            # Reduce to fixed 4x4 grid regardless of input resolution.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Project each of the 16 patches from 256 -> out_dim.
        # Dropout regularises patch representations before cross-attention.
        self.patch_proj = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)                          # [B, 256, 4, 4]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)               # [B, 4, 4, 256]
        x = x.reshape(B, H * W, C)              # [B, 16, 256]
        x = self.patch_proj(x)                   # [B, 16, out_dim]
        return x                                 # sequence of 16 patch tokens


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
