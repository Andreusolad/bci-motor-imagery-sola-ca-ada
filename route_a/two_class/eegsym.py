#!/usr/bin/env python
r"""PyTorch port of EEGSym (Perez-Velasco et al., IEEE TNSRE 2022).

Translated layer by layer from the official Keras/TensorFlow implementation
(github.com/Serpeve/EEGSym, EEGSym_architecture.py) with residual=True and symmetric=True,
the published configuration. The Conv3D layers carry a hemisphere axis (division = 2). Tensor
layout: (B, feat, division, time, channel), the Keras layout (B, division, time, channel, feat)
permuted.

Check: with filters_per_branch = 24 the output feature vector has 36 elements, as stated in
the paper. `python eegsym.py` runs this check and a forward pass with 750 and 125 samples.

The class defaults are the published hyperparameters: filters_per_branch = 24, dropout = 0.4,
ch_lateral = 3, scales_time = (500, 250, 125) ms. The input is raw EEG (no CWT). The montage
must be symmetric and ordered [3 left lateral, 2 midline, 3 right lateral]; for the 8-channel
montage used here that is [FC3, C3, CP3, FCz, Cz, FC4, C4, CP4] (see EEGSYM_REORDER).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralModule(nn.Module):
    """General inception/residual module (general_module of the Keras code).
    Written for residual=True. Operates on (B, feat_in, div, T, C)."""
    def __init__(self, feat_in, scales_samples, filters_per_branch, ncha, dropout, average):
        super().__init__()
        self.nscales = len(scales_samples)
        self.ncha = ncha
        self.average = average
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        Fb = filters_per_branch
        self.conv_t = nn.ModuleList(
            [nn.Conv3d(feat_in, Fb, (1, s, 1), padding='same', bias=True) for s in scales_samples])
        self.bn_t = nn.ModuleList([nn.BatchNorm3d(Fb) for _ in scales_samples])
        feat_cat = Fb * self.nscales           # after concatenating the temporal branches
        # temporal residual: conv_1 is used only when nscales == 1
        if self.nscales == 1:
            self.conv_1 = nn.Conv3d(feat_in, Fb, (1, 1, 1), bias=False)
            self.bn_1 = nn.BatchNorm3d(Fb)
        # spatial analysis across the channels of each hemisphere
        if ncha != 1:
            if self.nscales != 1:
                # depthwise: feat_cat -> feat_cat, groups=feat_cat, kernel (1,1,ncha), no padding
                self.sconv = nn.Conv3d(feat_cat, feat_cat, (1, 1, ncha), groups=feat_cat, bias=False)
                self.sbn = nn.BatchNorm3d(feat_cat)
            else:
                self.sconv = nn.Conv3d(feat_cat, feat_cat, (1, 1, ncha), bias=False)
                self.sbn = nn.BatchNorm3d(feat_cat)
        self.feat_out = feat_cat

    def forward(self, x):
        outs = [self.drop(self.act(bn(ct(x)))) for ct, bn in zip(self.conv_t, self.bn_t)]
        out = torch.cat(outs, dim=1) if len(outs) != 1 else outs[0]
        # temporal residual (broadcast over feat when x has a single feature map)
        if self.nscales != 1:
            out = out + x
        else:
            out = out + self.drop(self.act(self.bn_1(self.conv_1(x))))
        if self.average != 1:
            out = F.avg_pool3d(out, (1, self.average, 1))
        # spatial residual (broadcast over the channel axis C: 1 -> C)
        if self.ncha != 1:
            temp = self.drop(self.act(self.sbn(self.sconv(out))))
            out = out + temp
        return out


class EEGSym(nn.Module):
    def __init__(self, ncha=8, input_samples=750, fs=250, filters_per_branch=24,
                 scales_time=(500, 250, 125), dropout=0.4, ch_lateral=3):
        super().__init__()
        self.ncha_in = ncha
        self.ch_lateral = ch_lateral
        self.nc_hemi = ncha - ch_lateral                 # channels per hemisphere (3 + 2 = 5)
        Fb, nsc = filters_per_branch, len(scales_time)
        scales = [max(1, int(s * fs / 1000)) for s in scales_time]
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

        # --- tempospatial: 2 inception modules ---
        self.b1 = GeneralModule(1, scales, Fb, self.nc_hemi, dropout, average=2)
        self.b2 = GeneralModule(self.b1.feat_out, [max(1, x // 4) for x in scales], Fb, self.nc_hemi, dropout, average=2)
        # --- 3 residual modules ---
        f_half = int(Fb * nsc / 2)                       # 36 for N = 24
        f_qtr = int(Fb * nsc / 4)                        # 18 for N = 24
        self.b3a = GeneralModule(self.b2.feat_out, [16], f_half, self.nc_hemi, dropout, average=2)
        self.b3b = GeneralModule(self.b3a.feat_out, [8], f_half, self.nc_hemi, dropout, average=2)
        self.b3c = GeneralModule(self.b3b.feat_out, [4], f_qtr, self.nc_hemi, dropout, average=2)

        # --- temporal reduction ---
        self.tred_conv = nn.Conv3d(f_qtr, f_qtr, (1, 4, 1), padding='same', bias=False)
        self.tred_bn = nn.BatchNorm3d(f_qtr)

        # --- channel merging (2 residual + 1 grouped final) ---
        self.chm_conv = nn.ModuleList([nn.Conv3d(f_qtr, f_qtr, (2, 1, self.nc_hemi), bias=False) for _ in range(2)])
        self.chm_bn = nn.ModuleList([nn.BatchNorm3d(f_qtr) for _ in range(2)])
        self.chm_final = nn.Conv3d(f_qtr, f_qtr, (2, 1, self.nc_hemi), groups=int(Fb * nsc / 8), bias=False)
        self.chm_final_bn = nn.BatchNorm3d(f_qtr)

        # --- temporal merging ---
        tm_k = max(1, input_samples // 64)
        self.tm_k = tm_k
        self.tm_conv = nn.Conv3d(f_qtr, f_qtr, (1, tm_k, 1), bias=False)
        self.tm_bn = nn.BatchNorm3d(f_qtr)
        self.tm_final = nn.Conv3d(f_qtr, f_qtr * 2, (1, tm_k, 1), groups=f_qtr, bias=False)
        self.tm_final_bn = nn.BatchNorm3d(f_qtr * 2)

        # --- output module (4 residual 1x1x1 convs) ---
        f_out = f_qtr * 2                                # 36 for N = 24
        self.out_convs = nn.ModuleList([nn.Conv3d(f_out, f_half, (1, 1, 1), bias=False) for _ in range(4)])
        self.out_bns = nn.ModuleList([nn.BatchNorm3d(f_half) for _ in range(4)])
        # the output residual needs f_out == f_half
        assert f_out == f_half, f"output residual needs f_out == f_half ({f_out} != {f_half})"
        self.fc = nn.Linear(f_out, 2)
        self.feat_out_dim = f_out

    def _symmetric_input(self, x):
        """(B, C_in, T) -> (B, 1, div=2, T, C_hemi). Input order [Llat x3, mid x2, Rlat x3]."""
        cl = self.ch_lateral
        nc_h = self.nc_hemi
        left = x[:, list(range(cl)) + list(range(cl, nc_h)), :]          # left lateral + midline
        right = x[:, list(range(nc_h, nc_h + cl)) + list(range(cl, nc_h)), :]  # right lat + mid
        h = torch.stack([left, right], dim=1)          # (B, 2, C_hemi, T)
        h = h.permute(0, 1, 3, 2).unsqueeze(1)         # (B, 1, 2, T, C_hemi)
        return h

    def forward(self, x):                              # x: (B, C_in, T)
        h = self._symmetric_input(x)
        h = self.b1(h); h = self.b2(h)
        h = self.b3a(h); h = self.b3b(h); h = self.b3c(h)
        # temporal reduction
        h = h + self.drop(self.act(self.tred_bn(self.tred_conv(h))))
        h = F.avg_pool3d(h, (1, 2, 1))
        # channel merging
        for cv, bn in zip(self.chm_conv, self.chm_bn):
            h = h + self.drop(self.act(bn(cv(h))))     # broadcast over div and channel
        h = self.drop(self.act(self.chm_final_bn(self.chm_final(h))))   # div, channel -> 1
        # temporal merging
        h = h + self.drop(self.act(self.tm_bn(self.tm_conv(h))))
        h = self.drop(self.act(self.tm_final_bn(self.tm_final(h))))     # time -> 1
        # output module
        for cv, bn in zip(self.out_convs, self.out_bns):
            h = h + self.drop(self.act(bn(cv(h))))
        h = h.flatten(1)
        return self.fc(h)


# EEGSym channel order [FC3,C3,CP3, FCz,Cz, FC4,C4,CP4] as indices into the cache order
# [FC3,FCz,FC4,C3,Cz,C4,CP3,CP4] (0..7):
EEGSYM_REORDER = [0, 3, 6, 1, 4, 2, 5, 7]


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for T in (750, 125):
        m = EEGSym(ncha=8, input_samples=T, fs=250).to(dev)
        x = torch.randn(4, 8, T, device=dev)
        y = m(x)
        nparams = sum(p.numel() for p in m.parameters())
        print(f"T={T}: out {tuple(y.shape)} | feat_out_dim={m.feat_out_dim} | params={nparams:,}")
    print("check '36 features': feat_out_dim == 36 for N=24 ->",
          EEGSym(input_samples=750).feat_out_dim == 36)
