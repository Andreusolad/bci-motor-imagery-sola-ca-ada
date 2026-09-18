"""Network models: EEG-Conformer and EEGNet, with 3 outputs by default (LH / RH / IDLE).

Both models expose .features(x), the penultimate representation before the linear head,
which the Mahalanobis OOD detector (ood.py) uses to model each class in feature space.

load_pretrained_backbone() initializes a model from a checkpoint trained with a different
number of classes: tensors with matching name and shape are copied and the rest (the final
head) keeps its random initialization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EEGNet
# ============================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels=8, n_samples=500, n_classes=3,
                 F1=8, D=2, F2=16, dropout=0.5, kern1=125, pool1=4, pool2=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern1), padding=(0, kern1 // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, pool1)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(dropout))
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            x = self.block3(self.block2(self.block1(x)))
            self.flat_dim = x.numel()
        self.head = nn.Linear(self.flat_dim, n_classes)

    def features(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return x.flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


# ============================================================
# EEG-Conformer
# ============================================================
class ConvFront(nn.Module):
    def __init__(self, n_channels=8, F1=40, kern1=25, pool=75, pool_stride=15, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, kern1), padding=(0, kern1 // 2), bias=False)
        self.spatial = nn.Conv2d(F1, F1, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(F1)
        self.pool = nn.AvgPool2d((1, pool), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(F1, F1, 1, bias=False)

    def forward(self, x):
        x = self.temporal(x); x = self.spatial(x); x = self.bn(x)
        x = F.elu(x); x = self.pool(x); x = self.drop(x)
        return self.proj(x).squeeze(2)


class TFBlock(nn.Module):
    def __init__(self, dim=40, heads=10, ff=160, dropout=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class EEGConformer(nn.Module):
    def __init__(self, n_channels=8, n_samples=500, n_classes=3,
                 F1=40, depth=6, heads=10, dropout=0.5):
        super().__init__()
        self.conv = ConvFront(n_channels, F1=F1, dropout=dropout)
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_samples)
            self.n_tokens = self.conv(x).shape[-1]
        self.tf_blocks = nn.Sequential(*[TFBlock(F1, heads, ff=4 * F1, dropout=dropout)
                                         for _ in range(depth)])
        self.feat_dim = F1 * self.n_tokens
        self.head = nn.Linear(self.feat_dim, n_classes)

    def features(self, x):
        """Penultimate vector (before the head); used by the Mahalanobis OOD detector."""
        x = self.conv(x).transpose(1, 2)
        x = self.tf_blocks(x)
        return x.flatten(1)

    def forward(self, x):
        return self.head(self.features(x))


# ============================================================
# Utilities
# ============================================================
def build_model(name: str = 'conformer', n_channels=8, n_samples=500, n_classes=3, **kw):
    name = name.lower()
    if name in ('conformer', 'eegconformer'):
        return EEGConformer(n_channels, n_samples, n_classes, **kw)
    if name in ('eegnet', 'net'):
        return EEGNet(n_channels, n_samples, n_classes, **kw)
    raise ValueError(f'unknown model: {name!r}')


def load_pretrained_backbone(model: nn.Module, ckpt_path, log=print):
    """Load checkpoint weights into `model`, skipping tensors whose name or shape differ.

    Lets a backbone trained with 2 outputs initialize a 3-output model: the final head does
    not match and keeps its random initialization.
    """
    import torch
    sd = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    log(f'  backbone: loaded {len(loaded)} tensors, skipped {len(skipped)} '
        f'(name/shape mismatch): {skipped[:4]}{"..." if len(skipped) > 4 else ""}')
    return model
