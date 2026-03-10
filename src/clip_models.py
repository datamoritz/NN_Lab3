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
