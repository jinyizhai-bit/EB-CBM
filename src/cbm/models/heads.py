"""Heads for the evidence-bound CBM (EBC) family.

Mask plane convention everywhere: planes 0..K-1 = concepts (registry order), plane K = lesion.

* MaskHead            - light decoder F -> K+1 mask logits at 128x128 (+ ``mask_loss``: BCE + soft Dice).
* RRConceptHead (I2)  - concept logit from evidence pooled *only inside* the concept's predicted mask:
                        s_c = MLP_c([masked_avg(G, M_c), masked_max(G, M_c), log area(M_c)]); empty mask ->
                        learned null tokens.  Also returns the k-dim appearance descriptor P_c(masked_avg).
* GlobalConceptHead   - Koh-style image-level concept predictor from globally pooled features.
* CEMHead             - CEM (Espinosa Zarlenga 2022) +/- embeddings from globally pooled features.
* compute_report (I3) - clinically named report descriptors from masks (+ image for colours).
* ReportHead (I3)     - standardisation + sparse linear (L1) or 2-layer MLP over the report.
* Verifier (I4)       - small CNN on the 128x128 crop of x masked by a region M plus the mask channel ->
                        K "region contains structure c" logits; ``sample_verifier_regions`` builds its
                        training set (expert masks, other-structure masks, random lesion blobs, empty
                        masks, predicted masks) with overlap-derived targets.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ mask head
class MaskHead(nn.Module):
    def __init__(self, in_dim: int, n_out: int, mid: int = 256, out_size: int = 128):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Sequential(nn.Conv2d(in_dim, mid, 1), nn.GroupNorm(32, mid), nn.GELU(),
                                  nn.Conv2d(mid, mid, 3, padding=1), nn.GroupNorm(32, mid), nn.GELU())
        self.up1 = nn.Sequential(nn.Conv2d(mid, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.out = nn.Conv2d(64, n_out, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        h = self.proj(f)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.up1(h)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        h = self.up2(h)
        return self.out(h)


def mask_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, dice_w: float = 1.0,
              bce_w: float = 1.0) -> tuple[torch.Tensor, dict]:
    """BCE + batch-aggregated soft Dice per plane. logits/target (N,P,H,W); valid (N,P) bool.
    Extra logit planes beyond the target planes (discovered parts) are ignored."""
    logits, target = logits[:, :target.shape[1]].float(), target.float()
    v = valid.float()
    if v.sum() == 0:
        z = logits.sum() * 0
        return z, {"mask_bce": 0.0, "mask_dice_loss": 0.0}
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((2, 3))
    bce = (bce * v).sum() / v.sum()
    p = torch.sigmoid(logits)
    vv = v[:, :, None, None]
    inter = (p * target * vv).sum((0, 2, 3))
    den = (p * vv).sum((0, 2, 3)) + (target * vv).sum((0, 2, 3))
    has = v.sum(0) > 0
    dice = 1 - (2 * inter + 1.0) / (den + 1.0)
    dice = dice[has].mean()
    return bce_w * bce + dice_w * dice, {"mask_bce": float(bce.detach()), "mask_dice_loss": float(dice.detach())}


@torch.no_grad()
def dice_score(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """Per-plane dataset-level Dice (hard, thr 0.5) accumulators: returns (P,2) [2*inter, |p|+|t|]."""
    p = (prob[:, :target.shape[1]] > thr).float()
    t = (target > 0.5).float()
    vv = valid.float()[:, :, None, None]
    inter = (p * t * vv).sum((0, 2, 3))
    den = (p * vv).sum((0, 2, 3)) + (t * vv).sum((0, 2, 3))
    return torch.stack([2 * inter, den], 1)


# ============================================================================ concept heads
class _PerConceptMLP(nn.Module):
    """K independent 2-layer MLPs evaluated in one einsum. in (N,K,D) -> (N,K,out)."""

    def __init__(self, K: int, d_in: int, hidden: int, d_out: int = 1):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(K, d_in, hidden) / math.sqrt(d_in))
        self.b1 = nn.Parameter(torch.zeros(K, hidden))
        self.w2 = nn.Parameter(torch.randn(K, hidden, d_out) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(K, d_out))

    def forward(self, x):
        h = F.gelu(torch.einsum("nkd,kdh->nkh", x, self.w1) + self.b1)
        return torch.einsum("nkh,kho->nko", h, self.w2) + self.b2


class RRConceptHead(nn.Module):
    """Region-restricted concept head (I2)."""

    def __init__(self, in_dim: int, K: int, proj_dim: int = 256, hidden: int = 128, app_dim: int = 4,
                 pool_tau: float = 0.05, null_thresh: float = 0.5):
        super().__init__()
        self.K, self.tau, self.null_thresh = K, pool_tau, null_thresh
        self.proj = nn.Sequential(nn.Conv2d(in_dim, proj_dim, 1), nn.GELU())
        self.null_avg = nn.Parameter(torch.zeros(K, proj_dim))
        self.null_max = nn.Parameter(torch.zeros(K, proj_dim))
        nn.init.normal_(self.null_avg, std=0.02)
        nn.init.normal_(self.null_max, std=0.02)
        self.norm = nn.LayerNorm(2 * proj_dim)
        self.mlp = _PerConceptMLP(K, 2 * proj_dim + 1, hidden, 1)
        self.app_dim = app_dim
        if app_dim > 0:
            self.app_w = nn.Parameter(torch.randn(K, proj_dim, app_dim) / math.sqrt(proj_dim))
            self.app_b = nn.Parameter(torch.zeros(K, app_dim))

    def appearance(self, avg: torch.Tensor) -> torch.Tensor:
        if self.app_dim == 0:
            return avg.new_zeros(avg.shape[0], self.K, 0)
        return torch.tanh(torch.einsum("nkp,kpa->nka", avg, self.app_w) + self.app_b)

    def null_appearance(self) -> torch.Tensor:
        return self.appearance(self.null_avg[None])[0]       # (K, app_dim)

    def forward(self, f: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """f (N,d,h,w) features; m (N,K,H,W) soft masks in [0,1]."""
        g = self.proj(f).float()                                         # (N,p,h,w)
        n, p, h, w = g.shape
        m = m.float()
        mf = F.adaptive_avg_pool2d(m, (h, w))                             # area-interpolated (N,K,h,w)
        gate = mf > self.tau
        wgt = (mf * gate).flatten(2)                                      # (N,K,hw)
        gf = g.flatten(2)                                                 # (N,p,hw)
        wsum = wgt.sum(-1, keepdim=True)
        avg = torch.einsum("nkl,npl->nkp", wgt, gf) / wsum.clamp_min(1e-6)
        neg = torch.finfo(gf.dtype).min
        mx = torch.where(gate.flatten(2)[:, :, None, :], gf[:, None], torch.full_like(gf[:, None], neg)).amax(-1)
        present = (m.flatten(2).amax(-1) > self.null_thresh) & gate.flatten(2).any(-1)   # (N,K)
        e = present[..., None]
        avg = torch.where(e, avg, self.null_avg[None].float())
        mx = torch.where(e, mx, self.null_max[None].float())
        area = m.mean((2, 3))                                             # fraction of image
        log_area = torch.where(present, torch.log(area + 1e-4), torch.full_like(area, math.log(1e-4))) / 5.0
        inp = torch.cat([self.norm(torch.cat([avg, mx], -1)), log_area[..., None]], -1)
        s = self.mlp(inp)[..., 0]
        return s, {"avg": avg, "area": area, "present": present, "app": self.appearance(avg)}


class LesionEmbed(nn.Module):
    """Per-image appearance descriptor pooled only inside the (predicted) lesion mask.
    f (N,d,h,w), L (N,H,W) soft lesion mask -> (N,dim) in [-1,1]; empty lesion mask -> learned null token."""

    def __init__(self, in_dim: int, dim: int = 8, proj_dim: int = 64, pool_tau: float = 0.05,
                 null_thresh: float = 0.5):
        super().__init__()
        self.tau, self.null_thresh = pool_tau, null_thresh
        self.proj = nn.Sequential(nn.Conv2d(in_dim, proj_dim, 1), nn.GELU())
        self.null = nn.Parameter(torch.zeros(2 * proj_dim))
        nn.init.normal_(self.null, std=0.02)
        self.out = nn.Sequential(nn.LayerNorm(2 * proj_dim), nn.Linear(2 * proj_dim, dim), nn.Tanh())

    def forward(self, f: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        g = self.proj(f).float()
        n, p, h, w = g.shape
        L = L.float()
        mf = F.adaptive_avg_pool2d(L[:, None], (h, w))[:, 0]
        gate = (mf > self.tau).flatten(1)                                   # (N,hw)
        wgt = mf.flatten(1) * gate
        gf = g.flatten(2)                                                   # (N,p,hw)
        avg = torch.einsum("nl,npl->np", wgt, gf) / wgt.sum(-1, keepdim=True).clamp_min(1e-6)
        neg = torch.finfo(gf.dtype).min
        mx = torch.where(gate[:, None], gf, torch.full_like(gf, neg)).amax(-1)
        present = (L.flatten(1).amax(-1) > self.null_thresh) & gate.any(-1)
        z = torch.where(present[:, None], torch.cat([avg, mx], -1), self.null[None].float())
        return self.out(z)


class LesionCropEmbed(nn.Module):
    """Strictly pixel-local lesion appearance descriptor: crop the bounding box (+10 %) of the binarised predicted
    lesion mask, zero every pixel outside the mask, small CNN on [x * m, m] at crop x crop -> (N,dim) in [-1,1].
    Outside-lesion pixels can only act through the predicted mask itself (which pixels count as lesion)."""

    def __init__(self, dim: int = 8, width: int = 32, crop: int = 128):
        super().__init__()
        self.crop = crop

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        w = width
        self.net = nn.Sequential(blk(4, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8 * w, dim), nn.Tanh())

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        from torchvision.ops import roi_align
        H, W = x.shape[-2:]
        Lb = (L.detach().float() > 0.5).float()                          # (N,h,w) binarised lesion mask
        bx = region_boxes(Lb, (H, W), margin=0.10, min_box=32.0)
        idx = torch.arange(len(x), device=x.device, dtype=torch.float32)[:, None]
        Lf = F.interpolate(Lb[:, None], size=(H, W), mode="nearest")      # zero non-lesion pixels *before* resampling
        xc = roi_align(x.float() * Lf, torch.cat([idx, bx], 1), (self.crop, self.crop), 1.0, 2, True)
        mc = roi_align(Lf, torch.cat([idx, bx], 1), (self.crop, self.crop), 1.0, 2, True)
        return self.net(torch.cat([xc, mc], 1))


class _RGBMConv(nn.Module):
    """First conv of a pretrained RGB encoder extended by a zero-initialised mask-channel branch (4-channel input)."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.rgb = conv
        self.m = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        nn.init.zeros_(self.m.weight)

    def forward(self, x):
        return self.rgb(x[:, :3]) + self.m(x[:, 3:4])


class LocalEncoder(nn.Module):
    """timm encoder (default resnet18, ImageNet-pretrained, trainable) on masked crops -> (M, D) pooled features."""

    def __init__(self, name: str = "resnet18.a1_in1k", pretrained: bool = True, mask_channel: bool = True,
                 grad_checkpointing: bool = True, chunk: int = 128):
        super().__init__()
        import timm
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if mask_channel:
            if not isinstance(getattr(self.net, "conv1", None), nn.Conv2d):
                raise ValueError(f"mask_channel needs a ResNet-style encoder with .conv1, got {name}")
            self.net.conv1 = _RGBMConv(self.net.conv1)
        self.out_dim = self.net.num_features
        self.ckpt, self.chunk = grad_checkpointing, chunk

    def forward(self, x):
        outs = []
        for i in range(0, len(x), self.chunk):
            xi = x[i:i + self.chunk]
            if self.ckpt and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                outs.append(checkpoint(self.net, xi, use_reentrant=False))
            else:
                outs.append(self.net(xi))
        return torch.cat(outs) if outs else x.new_zeros(0, self.out_dim)


def square_boxes(Rb: torch.Tensor, img_hw: tuple[int, int], margin: float = 0.10, min_box: float = 16.0):
    """Bounding boxes of binary regions (M,h,w), enlarged by `margin` per side, made square, clamped (shifted) into
    the image. Returns (M,4) x0,y0,x1,y1 in image pixels."""
    H, W = img_hw
    b = region_boxes(Rb, img_hw, margin=margin, min_box=min_box)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    side = torch.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]).clamp(max=float(min(H, W)))
    x0 = (cx - side / 2).clamp(min=0.0)
    y0 = (cy - side / 2).clamp(min=0.0)
    x0 = torch.minimum(x0, W - side)
    y0 = torch.minimum(y0, H - side)
    return torch.stack([x0, y0, x0 + side, y0 + side], 1)


class CropEvidence(nn.Module):
    """Strictly local concept evidence (concept_evidence.source = crop).

    For every concept plane c (and the lesion plane when lesion_dim > 0): binarise the detached predicted mask at
    `mask_thresh`; if its area < `min_area_frac` of the image use learned null tokens; otherwise crop the square
    bounding box (+margin) of the region from the input image, keep only pixels whose every sampled source pixel lies
    inside the nearest-upsampled region (everything else = 0), optionally append the region as a 4th channel, resize
    to crop_size and encode all crops of the batch with one shared local encoder. Concept-specific linear heads give
    s_c (from [features, log-area]) and the appearance descriptor a_c = tanh(W_c features); a separate head gives the
    lesion descriptor. Nothing here sees the global backbone features: the masks are the only global input."""

    def __init__(self, K: int, app_dim: int = 4, lesion_dim: int = 0, encoder: dict | None = None,
                 crop_size: int = 224, margin: float = 0.10, min_area_frac: float = 0.002, mask_thresh: float = 0.5,
                 mask_channel: bool = True, n_parts: int = 0, part_app_dim: int = 4, part_support_tau: float = 0.1):
        super().__init__()
        enc = dict(encoder or {})
        self.enc = LocalEncoder(enc.get("name", "resnet18.a1_in1k"), enc.get("pretrained", True), mask_channel,
                                enc.get("grad_checkpointing", True), int(enc.get("chunk", 128)))
        D = self.enc.out_dim
        self.K, self.app_dim, self.lesion_dim = K, app_dim, lesion_dim
        self.crop, self.margin, self.min_area, self.thr, self.mask_channel = crop_size, margin, min_area_frac, \
            mask_thresh, mask_channel
        self.w_s = nn.Parameter(torch.randn(K, D) / math.sqrt(D))
        self.w_area = nn.Parameter(torch.zeros(K))
        self.b_s = nn.Parameter(torch.zeros(K))
        self.null_s = nn.Parameter(torch.full((K,), -2.0))
        if app_dim > 0:
            self.w_a = nn.Parameter(torch.randn(K, D, app_dim) / math.sqrt(D))
            self.b_a = nn.Parameter(torch.zeros(K, app_dim))
            self.null_a = nn.Parameter(torch.zeros(K, app_dim))
        if lesion_dim > 0:
            self.lesion_head = nn.Linear(D, lesion_dim)
            self.null_l = nn.Parameter(torch.zeros(lesion_dim))
        # discovered evidence regions (no expert labels): soft-mask-weighted crops, appearance only (no presence)
        self.n_parts, self.part_app_dim, self.part_tau = n_parts, part_app_dim, part_support_tau
        if n_parts > 0:
            self.w_p = nn.Parameter(torch.randn(n_parts, D, part_app_dim) / math.sqrt(D))
            self.b_p = nn.Parameter(torch.zeros(n_parts, part_app_dim))
            self.null_p = nn.Parameter(torch.zeros(n_parts, part_app_dim))
        self.last_n_crops = 0
        self.last_n_part_crops = 0

    def null_appearance(self) -> torch.Tensor:
        return torch.tanh(self.null_a) if self.app_dim > 0 else self.null_s.new_zeros(self.K, 0)

    def crops(self, x: torch.Tensor, Mb: torch.Tensor):
        """x (N,3,H,W); Mb (N,P,h,w) binary -> crops (n_sel, C, S, S) for the non-empty planes, sel (n_sel,) flat idx."""
        from torchvision.ops import roi_align
        N, P, h, w = Mb.shape
        H, W = x.shape[-2:]
        empty = Mb.flatten(2).mean(-1) < self.min_area                      # (N,P)
        flat = Mb.reshape(N * P, h, w)
        sel = torch.nonzero(~empty.flatten(), as_tuple=True)[0]
        if sel.numel() == 0:
            return None, sel, empty
        bx = square_boxes(flat[sel], (H, W), self.margin)
        img_idx = (sel // P).float()[:, None]
        xc = roi_align(x.float(), torch.cat([img_idx, bx], 1), (self.crop, self.crop), 1.0, 0, True)
        own = torch.arange(len(sel), device=x.device, dtype=torch.float32)[:, None]
        mc = roi_align(flat[sel][:, None].float(), torch.cat([own, bx], 1), (self.crop, self.crop), w / W, 0, True)
        keep = (mc >= 0.999).float()            # all bilinear source samples inside the region
        inp = xc * keep
        if self.mask_channel:
            inp = torch.cat([inp, keep], 1)
        return inp, sel, empty

    def part_support(self, Mp_up: torch.Tensor) -> torch.Tensor:
        """Weights of a part crop: the bilinearly upsampled SOFT part mask, exactly 0 where it is <= part_tau
        (pixels there never influence the descriptor; the gradient flows into the mask inside the support)."""
        return Mp_up * (Mp_up.detach() > self.part_tau).float()

    def part_crops(self, x: torch.Tensor, Mp: torch.Tensor):
        """x (N,3,H,W); Mp (N,P,h,w) soft part masks (with grad) -> crops (n_sel,C,S,S), sel, empty (N,P)."""
        from torchvision.ops import roi_align
        N, P, h, w = Mp.shape
        H, W = x.shape[-2:]
        hard = (Mp.detach().float() > self.thr).float()
        empty = hard.flatten(2).mean(-1) < self.min_area
        sel = torch.nonzero(~empty.flatten(), as_tuple=True)[0]
        if sel.numel() == 0:
            return None, sel, empty
        bx = square_boxes(hard.reshape(N * P, h, w)[sel], (H, W), self.margin)
        up = F.interpolate(Mp.reshape(N * P, 1, h, w)[sel].float(), size=(H, W), mode="bilinear", align_corners=False)
        wgt = self.part_support(up)                                          # (n_sel,1,H,W)
        img = (sel // P)
        xw = x.float()[img] * wgt
        own = torch.arange(len(sel), device=x.device, dtype=torch.float32)[:, None]
        rois = torch.cat([own, bx], 1)
        xc = roi_align(xw, rois, (self.crop, self.crop), 1.0, 0, True)
        if self.mask_channel:
            xc = torch.cat([xc, roi_align(wgt, rois, (self.crop, self.crop), 1.0, 0, True)], 1)
        return xc, sel, empty

    def forward(self, x: torch.Tensor, M: torch.Tensor, Mp: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """x (N,3,H,W) normalised image; M (N,P,h,w) soft masks, P = K (+1 lesion plane if lesion_dim);
        Mp (N,n_parts,h,w) soft discovered-part masks (gradient kept) or None."""
        N, P = M.shape[:2]
        K = self.K
        Mb = (M.detach().float() > self.thr).float()
        inp, sel, empty = self.crops(x, Mb)
        pin, psel, pempty = (None, None, None)
        if self.n_parts > 0 and Mp is not None:
            pin, psel, pempty = self.part_crops(x, Mp)
        D = self.enc.out_dim
        Hf = x.new_zeros(N * P, D, dtype=torch.float32)
        # concept crops and part crops are encoded in SEPARATE calls: the batch composition of one never changes the
        # (bf16 / cuDNN) numerics of the other, so fixed-mask locality tests give exactly zero drift
        if inp is not None:
            Hf = Hf.index_copy(0, sel, self.enc(inp).float())
        self.last_n_crops = int(sel.numel())
        Hf = Hf.view(N, P, D)
        hard_area = Mb.flatten(2).mean(-1)                                  # (N,P)
        la = torch.log(hard_area[:, :K].clamp_min(1e-4)) / 5.0
        s = torch.einsum("nkd,kd->nk", Hf[:, :K], self.w_s) + self.w_area * la + self.b_s
        ek = empty[:, :K]
        s = torch.where(ek, self.null_s[None].expand_as(s), s)
        if self.app_dim > 0:
            app = torch.tanh(torch.einsum("nkd,kda->nka", Hf[:, :K], self.w_a) + self.b_a)
            app = torch.where(ek[..., None], self.null_appearance()[None].expand_as(app), app)
        else:
            app = s.new_zeros(N, K, 0)
        info = {"app": app, "area": M[:, :K].float().mean((2, 3)), "present": ~ek, "n_crops": self.last_n_crops}
        if self.lesion_dim > 0:
            le = torch.tanh(self.lesion_head(Hf[:, K]))
            info["lesion_emb"] = torch.where(empty[:, K:K + 1], torch.tanh(self.null_l)[None].expand_as(le), le)
        if self.n_parts > 0 and Mp is not None:
            NP = self.n_parts
            Hp = x.new_zeros(N * NP, D, dtype=torch.float32)
            if pin is not None:
                Hp = Hp.index_copy(0, psel, self.enc(pin).float())
            Hp = Hp.view(N, NP, D)
            pa = torch.tanh(torch.einsum("npd,pda->npa", Hp, self.w_p) + self.b_p)
            pa = torch.where(pempty[..., None], torch.tanh(self.null_p)[None].expand_as(pa), pa)
            info["part_app"] = pa
            info["part_logarea"] = torch.log(Mp.float().mean((2, 3)) + 1e-4) / 5.0      # differentiable
            info["part_present"] = ~pempty
            self.last_n_part_crops = 0 if psel is None else int(psel.numel())
            info["n_part_crops"] = self.last_n_part_crops
        return s, info


def part_regularizers(Mp: torch.Tensor, L: torch.Tensor, S: torch.Tensor | None, area_min: float = 0.01,
                      area_max: float = 0.30) -> dict:
    # (+ binary: mean 4 m (1 - m) inside the lesion, pushes part masks towards crisp 0/1 regions;
    #  + peak: hinge on the mean of each part's 16 highest values (>= 0.6), keeps a confident core per part)
    """Regularisers for discovered part masks Mp (N,P,h,w) given a lesion mask L (N,h,w) and expert structure masks
    S (N,K,h,w) (or None). Returns dict of scalar tensors: outside (mass outside the lesion), compact (mean distance of
    mass to its centroid / lesion radius), overlap (soft Dice part-part and part-structure), area (hinge outside
    [area_min, area_max] x lesion area)."""
    with torch.autocast("cuda", enabled=False):
        Mp, L = Mp.float(), L.float().detach()
        N, P, h, w = Mp.shape
        eps = 1e-6
        mass = Mp.sum((2, 3)) + eps                                          # (N,P)
        outside = ((Mp * (1 - L[:, None])).sum((2, 3)) / mass).mean()
        yy, xx = torch.meshgrid(torch.arange(h, device=Mp.device, dtype=torch.float32),
                                torch.arange(w, device=Mp.device, dtype=torch.float32), indexing="ij")
        cy = (Mp * yy).sum((2, 3)) / mass
        cx = (Mp * xx).sum((2, 3)) / mass
        dist = torch.sqrt((yy[None, None] - cy[..., None, None]) ** 2 + (xx[None, None] - cx[..., None, None]) ** 2 + eps)
        radius = torch.sqrt(L.sum((1, 2)) / math.pi).clamp_min(1.0)          # (N,)
        compact = (((Mp * dist).sum((2, 3)) / mass) / radius[:, None]).mean()

        def sdice(a, b):                                                     # (N,A,h,w),(N,B,h,w) -> (N,A,B)
            inter = torch.einsum("nahw,nbhw->nab", a, b)
            return 2 * inter / (a.sum((2, 3))[:, :, None] + b.sum((2, 3))[:, None, :] + eps)
        ov = []
        if P > 1:
            d = sdice(Mp, Mp)
            iu = torch.triu_indices(P, P, 1, device=Mp.device)
            ov.append(d[:, iu[0], iu[1]].mean())
        if S is not None and S.shape[1] > 0:
            ov.append(sdice(Mp, S.float().detach()).mean())
        overlap = torch.stack(ov).mean() if ov else Mp.sum() * 0
        a = Mp.sum((2, 3)) / (L.sum((1, 2))[:, None] + eps)                  # fraction of the lesion
        area = (F.relu(area_min - a) + F.relu(a - area_max)).mean()
        binary = ((4 * Mp * (1 - Mp)) * L[:, None]).sum((2, 3)).div(L.sum((1, 2))[:, None] + eps).mean()
        # peak: every part keeps a confident core (mean of its 16 highest values >= 0.6), so that diffuse low-valued
        # masks (which satisfy the soft area bound but give no hard region and hence no crop) are penalised
        top = Mp.flatten(2).topk(min(16, h * w), dim=-1).values.mean(-1)
        peak = F.relu(0.6 - top).mean()
        return {"outside": outside, "compact": compact, "overlap": overlap, "area": area, "binary": binary,
                "peak": peak}


class GlobalConceptHead(nn.Module):
    def __init__(self, in_dim: int, K: int, hidden: int = 0):
        super().__init__()
        self.net = nn.Linear(in_dim, K) if hidden == 0 else nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, K))

    def forward(self, g):
        return self.net(g)


class CEMHead(nn.Module):
    """Concept Embedding Model head: per concept c+ / c- embeddings from global features, shared scorer."""

    def __init__(self, in_dim: int, K: int, emb: int = 16):
        super().__init__()
        self.K, self.emb = K, emb
        self.gen = nn.Linear(in_dim, K * 2 * emb)
        self.score = nn.Linear(2 * emb, 1)

    def forward(self, g):
        e = F.leaky_relu(self.gen(g)).view(g.shape[0], self.K, 2, self.emb)
        cpos, cneg = e[:, :, 0], e[:, :, 1]
        s = self.score(torch.cat([cpos, cneg], -1))[..., 0]
        return s, cpos, cneg


# ============================================================================ I3 report descriptors
ABCD_COLOURS = ["white", "red", "light_brown", "dark_brown", "blue_gray", "black"]
# Reference sRGB colours from the automatic-ABCD literature (Kasmi & Mokrani, IET IP 2016); a lesion pixel is
# assigned to its nearest reference in CIELAB if Delta-E < COLOUR_DE; colour present if >= COLOUR_FRAC of lesion.
_REF_RGB = {"white": (197, 188, 217), "red": (118, 21, 17), "light_brown": (163, 82, 16),
            "dark_brown": (135, 44, 5), "blue_gray": (113, 108, 139), "black": (41, 31, 30)}
COLOUR_DE = 30.0
COLOUR_FRAC = 0.05
STRUCT_DESC = ["presence", "area_frac_lesion", "n_components", "peripheral_index", "angular_asymmetry"]
LESION_DESC = ["lesion_asymmetry", "border_irregularity", "colour_count", "lesion_area"]


def report_names(concepts: list[str], colour_detail: bool = True) -> list[str]:
    names = [f"{c}:{d}" for c in concepts for d in STRUCT_DESC] + [f"lesion:{d}" for d in LESION_DESC]
    if colour_detail:
        names += [f"colour:{c}" for c in ABCD_COLOURS]
    return names


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """x (N,3,H,W) sRGB in [0,1] -> CIELAB (D65)."""
    x = x.float()
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=x.device)
    xyz = torch.einsum("ij,njhw->nihw", M, lin)
    wp = torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)
    t = xyz / wp
    d = 6 / 29
    ft = torch.where(t > d ** 3, t.clamp_min(1e-8) ** (1 / 3), t / (3 * d * d) + 4 / 29)
    L = 116 * ft[:, 1] - 16
    a = 500 * (ft[:, 0] - ft[:, 1])
    b = 200 * (ft[:, 1] - ft[:, 2])
    return torch.stack([L, a, b], 1)


def _ref_lab(device) -> torch.Tensor:
    ref = torch.tensor([_REF_RGB[c] for c in ABCD_COLOURS], dtype=torch.float32, device=device) / 255.0
    return rgb_to_lab(ref.view(-1, 3, 1, 1))[:, :, 0, 0]          # (6,3)


@torch.no_grad()
def _component_counts(m: torch.Tensor, thr: float = 0.5, min_px: int = 3) -> torch.Tensor:
    """Connected components (8-conn) of each detached plane. m (N,K,H,W) -> (N,K) float counts."""
    import cv2
    a = (m.detach().float().cpu().numpy() > thr).astype(np.uint8)
    out = np.zeros(a.shape[:2], np.float32)
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            if a[i, k].any():
                n, _, st, _ = cv2.connectedComponentsWithStats(a[i, k], connectivity=8)
                out[i, k] = float((st[1:, cv2.CC_STAT_AREA] >= min_px).sum())
    return torch.from_numpy(out).to(m.device)


def compute_report(masks: torch.Tensor, presence: torch.Tensor, x01: torch.Tensor | None,
                   colour_detail: bool = True, hard_colour: bool = False) -> torch.Tensor:
    """Report descriptors (N, D) from soft masks (N,K+1,H,W) [plane K = lesion], presence probs (N,K) and the
    un-normalised image x01 (N,3,H',W') in [0,1] (colours).  Differentiable w.r.t. the masks except the
    component counts (computed on the detached, thresholded mask)."""
    with torch.autocast("cuda", enabled=False):
        m = masks.float()
        n, P, H, W = m.shape
        K = P - 1
        S, L = m[:, :K], m[:, K]
        eps = 1e-6
        yy, xx = torch.meshgrid(torch.arange(H, device=m.device, dtype=torch.float32),
                                torch.arange(W, device=m.device, dtype=torch.float32), indexing="ij")
        AL = L.sum((1, 2)) + eps                                              # lesion pixels
        cx = (L * xx).sum((1, 2)) / AL
        cy = (L * yy).sum((1, 2)) / AL
        dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
        Ld = L.detach()
        ALd = Ld.sum((1, 2)) + eps
        sxx = (Ld * dx.detach() ** 2).sum((1, 2)) / ALd
        syy = (Ld * dy.detach() ** 2).sum((1, 2)) / ALd
        sxy = (Ld * dx.detach() * dy.detach()).sum((1, 2)) / ALd
        theta = 0.5 * torch.atan2(2 * sxy, sxx - syy)                         # major axis angle
        e1 = torch.stack([torch.cos(theta), torch.sin(theta)], -1)            # (N,2)
        e2 = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
        u = dx * e1[:, 0, None, None] + dy * e1[:, 1, None, None]             # (N,H,W)
        v = dx * e2[:, 0, None, None] + dy * e2[:, 1, None, None]
        radius = torch.sqrt(AL / math.pi)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + eps)
        # ---- per structure
        AS = S.sum((2, 3))                                                    # (N,K)
        area_frac = (AS / AL[:, None]).clamp(max=1.0)
        periph = (S * dist[:, None]).sum((2, 3)) / (AS * radius[:, None] + eps)
        periph = torch.where(AS > 0.5, periph, torch.zeros_like(periph))
        tau = 1.5
        su_p = (S * torch.sigmoid(u[:, None] / tau)).sum((2, 3))
        sv_p = (S * torch.sigmoid(v[:, None] / tau)).sum((2, 3))
        ang = 0.5 * ((2 * su_p - AS).abs() + (2 * sv_p - AS).abs()) / (AS + eps)
        ang = torch.where(AS > 0.5, ang, torch.zeros_like(ang))
        ncomp = torch.log1p(_component_counts(S))
        struct = torch.stack([presence.float(), area_frac, ncomp, periph, ang], -1)   # (N,K,5)
        # ---- lesion asymmetry: reflect the lesion mask about both principal axes through the centroid
        cxn = (cx + 0.5) / W * 2 - 1
        cyn = (cy + 0.5) / H * 2 - 1
        asym = []
        for e in (e1, e2):
            ex, ey = e[:, 0], e[:, 1]
            R = torch.stack([torch.stack([2 * ex * ex - 1, 2 * ex * ey], -1),
                             torch.stack([2 * ex * ey, 2 * ey * ey - 1], -1)], 1)    # (N,2,2)
            c = torch.stack([cxn, cyn], -1)
            t = c - torch.einsum("nij,nj->ni", R, c)
            theta_aff = torch.cat([R, t[..., None]], -1)
            grid = F.affine_grid(theta_aff, (n, 1, H, W), align_corners=False)
            Lr = F.grid_sample(L[:, None], grid, align_corners=False, padding_mode="zeros")[:, 0]
            asym.append((L - Lr).abs().sum((1, 2)) / (2 * AL))
        lesion_asym = asym[0] + asym[1]                                       # 0..2 ("number of asymmetric axes")
        # ---- border irregularity = 1 - compactness, soft perimeter via central differences
        Lp = F.pad(L[:, None], (1, 1, 1, 1), mode="replicate")
        gx = (Lp[:, :, 1:-1, 2:] - Lp[:, :, 1:-1, :-2]) / 2
        gy = (Lp[:, :, 2:, 1:-1] - Lp[:, :, :-2, 1:-1]) / 2
        per = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).sum((1, 2, 3))
        compact = (4 * math.pi * AL / (per ** 2 + eps)).clamp(0, 1)
        border = 1 - compact
        lesion_area = AL / (H * W)
        # ---- colours (non-differentiable w.r.t. the image; soft w.r.t. the lesion mask)
        if x01 is not None:
            xi = F.interpolate(x01.float(), size=(H, W), mode="area")
            lab = rgb_to_lab(xi)                                              # (N,3,H,W)
            ref = _ref_lab(m.device)                                          # (6,3)
            d = torch.cdist(lab.permute(0, 2, 3, 1).reshape(n, -1, 3), ref[None].expand(n, -1, -1))  # (N,HW,6)
            dmin, arg = d.min(-1)
            onehot = F.one_hot(arg, 6).float() * (dmin < COLOUR_DE).float()[..., None]   # (N,HW,6)
            Lc = (L > 0.5).float() if hard_colour else L      # hard: colours only from pixels inside the lesion mask
            frac = (onehot * Lc.reshape(n, -1, 1)).sum(1) / (Lc.sum((1, 2)) + eps)[:, None]   # (N,6)
            colour_present = torch.sigmoid((frac - COLOUR_FRAC) / 0.01)
        else:
            colour_present = torch.zeros(n, 6, device=m.device)
        ccount = colour_present.sum(-1)
        lesion = torch.stack([lesion_asym, border, ccount, lesion_area], -1)
        parts = [struct.flatten(1), lesion] + ([colour_present] if colour_detail else [])
        return torch.cat(parts, -1)


def absent_struct_values(rep: torch.Tensor, K: int, absent: torch.Tensor) -> torch.Tensor:
    """Set the 5 structure descriptors of concepts flagged absent (N,K bool) to their 'absent' values (0)."""
    r = rep.clone()
    s = r[:, :K * 5].view(-1, K, 5)
    s = torch.where(absent[..., None], torch.zeros_like(s), s)
    r[:, :K * 5] = s.reshape(-1, K * 5)
    return r


class ReportHead(nn.Module):
    """Standardise the report (BatchNorm without affine = running train statistics) then linear (L1) or MLP."""

    def __init__(self, d_in: int, n_classes: int, kind: str = "linear", hidden: int = 32):
        super().__init__()
        self.kind = kind
        self.bn = nn.BatchNorm1d(d_in, affine=False, momentum=0.05)
        self.net = nn.Linear(d_in, n_classes) if kind == "linear" else nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, r):
        return self.net(self.bn(r.float()))

    def l1(self) -> torch.Tensor:
        w = self.net.weight if self.kind == "linear" else self.net[0].weight
        return w.abs().sum()


# ============================================================================ I4 verifier
def region_boxes(R: torch.Tensor, img_hw: tuple[int, int], margin: float = 0.15, min_box: float = 48.0) -> torch.Tensor:
    """R (M,h,w) region masks -> (M,4) boxes in image pixel coords."""
    Mn, h, w = R.shape
    H, W = img_hw
    b = R > 0.5
    rows, cols = b.any(2), b.any(1)
    ar_h = torch.arange(h, device=R.device)
    ar_w = torch.arange(w, device=R.device)
    y0 = torch.where(rows, ar_h, h).amin(1).float()
    y1 = torch.where(rows, ar_h, -1).amax(1).float() + 1
    x0 = torch.where(cols, ar_w, w).amin(1).float()
    x1 = torch.where(cols, ar_w, -1).amax(1).float() + 1
    empty = ~b.flatten(1).any(1)
    sy, sx = H / h, W / w
    y0, y1, x0, x1 = y0 * sy, y1 * sy, x0 * sx, x1 * sx
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    bh = ((y1 - y0) * (1 + 2 * margin)).clamp_min(min_box)
    bw = ((x1 - x0) * (1 + 2 * margin)).clamp_min(min_box)
    y0, y1 = (cy - bh / 2).clamp(0, H), (cy + bh / 2).clamp(0, H)
    x0, x1 = (cx - bw / 2).clamp(0, W), (cx + bw / 2).clamp(0, W)
    full = torch.tensor([0.0, 0.0, float(W), float(H)], device=R.device)
    bx = torch.stack([x0, y0, x1, y1], 1)
    return torch.where(empty[:, None], full[None], bx)


class Verifier(nn.Module):
    """V(x, M): crops the region's bounding box (+margin) from x, resizes to crop x crop, input [x*M, M]."""

    def __init__(self, K: int, width: int = 32, crop: int = 128, margin: float = 0.15, min_box: float = 48.0,
                 strict: bool = False):
        super().__init__()
        self.K, self.crop, self.margin, self.min_box = K, crop, margin, min_box
        # strict: regions binarised at 0.5 and crop pixels kept only where every sampled source pixel lies inside the
        # (nearest-upsampled) region -> the score depends on no pixel outside the region (used with crop evidence)
        self.strict = strict

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        w = width
        self.net = nn.Sequential(blk(4, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(8 * w, K))

    def boxes(self, R: torch.Tensor, img_hw: tuple[int, int]) -> torch.Tensor:
        return region_boxes(R, img_hw, self.margin, self.min_box)

    def forward(self, x: torch.Tensor, R: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """x (N,3,H,W) normalised images; R (M,h,w) soft regions in [0,1]; idx (M,) image index -> logits (M,K)."""
        from torchvision.ops import roi_align
        H, W = x.shape[-2:]
        if self.strict:
            R = (R.detach().float() > 0.5).float()
        bx = self.boxes(R.detach(), (H, W))
        rois = torch.cat([idx.float()[:, None], bx], 1).to(x.dtype)
        xc = roi_align(x, rois, (self.crop, self.crop), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        sc = R.shape[-1] / W
        rr = torch.cat([torch.arange(len(R), device=R.device, dtype=torch.float32)[:, None], bx], 1)
        mc = roi_align(R[:, None].float(), rr, (self.crop, self.crop), spatial_scale=sc, sampling_ratio=2,
                       aligned=True).to(x.dtype)
        if self.strict:
            mc = (mc >= 0.999).to(x.dtype)
        return self.net(torch.cat([xc * mc, mc], 1))

    def score_masks(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """v_c logits for each concept's own predicted mask: M (N,K,h,w) -> (N,K)."""
        N, K = M.shape[:2]
        R = M.reshape(N * K, *M.shape[2:])
        idx = torch.arange(N, device=x.device).repeat_interleave(K)
        lg = self.forward(x, R, idx).view(N, K, K)
        return torch.diagonal(lg, dim1=1, dim2=2)


@torch.no_grad()
def random_lesion_blobs(lesion: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """One random ellipse (intersected with the lesion) per image. lesion (N,h,w) -> (N,h,w) {0,1}."""
    N, h, w = lesion.shape
    dev = lesion.device
    prob = (lesion > 0.5).float().flatten(1) + 1e-6
    ctr = torch.multinomial(prob, 1, generator=gen)[:, 0]
    cy, cx = (ctr // w).float(), (ctr % w).float()
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(N, device=dev, generator=gen)
    a, b, th = r(0.05 * h, 0.2 * h), r(0.05 * h, 0.2 * h), r(0, math.pi)
    yy, xx = torch.meshgrid(torch.arange(h, device=dev).float(), torch.arange(w, device=dev).float(), indexing="ij")
    dx, dy = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    u, v = dx * c + dy * s, -dx * s + dy * c
    ell = (u / a[:, None, None]) ** 2 + (v / b[:, None, None]) ** 2 <= 1
    return (ell & (lesion > 0.5)).float()


@torch.no_grad()
def sample_verifier_regions(gt: torch.Tensor, valid: torch.Tensor, pred: torch.Tensor | None = None,
                            max_per_image: int = 4, gen: torch.Generator | None = None, overlap_thr: float = 0.5):
    """Training regions for the verifier.

    gt (N,K+1,h,w) soft expert masks, valid (N,K+1) bool, pred (N,K+1,h,w) predicted masks (detached) or None.
    Candidate regions per image: every non-empty expert structure mask, one random lesion blob, one empty
    mask, every non-empty predicted structure mask.  Targets t_c = 1[|R & E_c| / |R| > overlap_thr].
    Returns R (M,h,w), idx (M,), targets (M,K)."""
    N, P, h, w = gt.shape
    K = P - 1
    dev = gt.device
    gtb = (gt > 0.5).float() * valid[:, :, None, None].float()
    cand_R, cand_i = [], []
    nonempty = gtb[:, :K].flatten(2).sum(-1) > 0                           # (N,K)
    ii, kk = torch.nonzero(nonempty, as_tuple=True)
    cand_R.append(gtb[ii, kk]); cand_i.append(ii)
    lesion = torch.where(valid[:, K, None, None], gtb[:, K], torch.ones_like(gtb[:, K]))
    cand_R.append(random_lesion_blobs(lesion, gen)); cand_i.append(torch.arange(N, device=dev))
    cand_R.append(torch.zeros(N, h, w, device=dev)); cand_i.append(torch.arange(N, device=dev))
    if pred is not None:
        pb = (pred[:, :K] > 0.5).float()
        ne = pb.flatten(2).sum(-1) > 4
        ii, kk = torch.nonzero(ne, as_tuple=True)
        cand_R.append(pb[ii, kk]); cand_i.append(ii)
    R = torch.cat(cand_R)
    idx = torch.cat(cand_i)
    # subsample to at most max_per_image regions per image
    if max_per_image > 0:
        idx_np = idx.cpu().numpy()
        perm = torch.randperm(len(R), generator=None).numpy()
        keep = np.zeros(len(R), bool)
        cnt = np.zeros(N, np.int64)
        for j in perm:
            if cnt[idx_np[j]] < max_per_image:
                keep[j] = True
                cnt[idx_np[j]] += 1
        keep_t = torch.from_numpy(keep).to(dev)
        R, idx = R[keep_t], idx[keep_t]
    area = R.flatten(1).sum(-1)                                           # (M,)
    inter = (R[:, None] * gtb[idx, :K]).flatten(2).sum(-1)                # (M,K)
    t = (inter / area.clamp_min(1)[:, None] > overlap_thr).float()
    known = valid[idx, :K]
    return R, idx, t, known
