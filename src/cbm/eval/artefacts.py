"""Synthetic dermoscopy artefacts pasted *outside* the lesion (shortcut-robustness test, DESIGN Section 7).

Each function takes uint8 RGB images (H,W,3) + a boolean lesion mask (H,W) and a seeded RandomState and returns
the modified image; pixels inside the (dilated) lesion are never changed.
  ruler      - semi-transparent scale bar with tick marks along the image side farthest from the lesion
  ink        - gentian-violet marker strokes / dots in a ring 15-60 px outside the lesion border
  vignette   - dark circular-aperture corners (dermatoscope vignetting)
  hair       - 6-15 dark curved hair strokes (quadratic Bezier, 1-3 px)
``apply_artefact(x01, lesion, kind, seeds)`` works on batched torch tensors (N,3,H,W) in [0,1].
"""
from __future__ import annotations

import numpy as np

ARTEFACTS = ["ruler", "ink", "vignette", "hair"]


def _dil(m: np.ndarray, r: int) -> np.ndarray:
    import cv2
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def _blend_outside(img, layer, alpha, keep):
    a = (alpha * (~keep))[..., None].astype(np.float32)
    return (img.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a).clip(0, 255).astype(np.uint8)


def ruler(img: np.ndarray, lesion: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    keep = _dil(lesion, 5)
    bw = int(0.08 * min(H, W))
    off = int(0.03 * min(H, W))
    bands = {"top": (slice(off, off + bw), slice(0, W)), "bottom": (slice(H - off - bw, H - off), slice(0, W)),
             "left": (slice(0, H), slice(off, off + bw)), "right": (slice(0, H), slice(W - off - bw, W - off))}
    side = min(bands, key=lambda s: keep[bands[s]].mean() + 1e-3 * rng.rand())
    layer = img.copy()
    alpha = np.zeros((H, W), np.float32)
    ys, xs = bands[side]
    base = np.array([225, 222, 200], np.float32) * rng.uniform(0.9, 1.05)
    layer[ys, xs] = base.clip(0, 255).astype(np.uint8)
    alpha[ys, xs] = 0.85
    step = int(rng.randint(9, 14))
    horiz = side in ("top", "bottom")
    L = W if horiz else H
    for i, p in enumerate(range(rng.randint(0, step), L, step)):
        ln = int(bw * (0.7 if i % 5 == 0 else 0.35))
        if horiz:
            y0 = ys.start if side == "top" else ys.stop - ln
            cv2.line(layer, (p, y0), (p, y0 + ln), (20, 20, 20), 1)
            alpha[y0:y0 + ln, max(p - 1, 0):p + 1] = 0.95
        else:
            x0 = xs.start if side == "left" else xs.stop - ln
            cv2.line(layer, (x0, p), (x0 + ln, p), (20, 20, 20), 1)
            alpha[max(p - 1, 0):p + 1, x0:x0 + ln] = 0.95
    return _blend_outside(img, layer, alpha, keep)


def ink(img: np.ndarray, lesion: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    keep = _dil(lesion, 5)
    ring = _dil(lesion, 60) & ~_dil(lesion, 15)
    if not ring.any():
        ring = ~keep
    if not ring.any():          # lesion covers the whole image: nothing can be pasted outside it
        return img.copy()
    ys, xs = np.nonzero(ring)
    layer = img.copy()
    alpha = np.zeros((H, W), np.uint8)
    col = tuple(int(c) for c in (np.array([95, 45, 150]) * rng.uniform(0.8, 1.1)).clip(0, 255))
    for _ in range(rng.randint(2, 5)):
        j = rng.randint(len(ys))
        cy, cx = int(ys[j]), int(xs[j])
        kind = rng.randint(3)
        th = int(rng.randint(4, 9))
        if kind == 0:      # dot
            cv2.circle(layer, (cx, cy), th + 2, col, -1); cv2.circle(alpha, (cx, cy), th + 2, 255, -1)
        elif kind == 1:    # arc
            ax = (int(rng.randint(15, 40)), int(rng.randint(10, 30)))
            a0 = int(rng.randint(0, 360))
            cv2.ellipse(layer, (cx, cy), ax, a0, 0, int(rng.randint(60, 180)), col, th)
            cv2.ellipse(alpha, (cx, cy), ax, a0, 0, int(rng.randint(60, 180)), 255, th)
        else:              # straight stroke
            d = rng.randn(2); d = d / (np.linalg.norm(d) + 1e-6) * rng.randint(15, 45)
            p2 = (int(cx + d[0]), int(cy + d[1]))
            cv2.line(layer, (cx, cy), p2, col, th); cv2.line(alpha, (cx, cy), p2, 255, th)
    a = cv2.GaussianBlur(alpha, (5, 5), 0).astype(np.float32) / 255.0 * 0.8
    return _blend_outside(img, layer, a, keep)


def vignette(img: np.ndarray, lesion: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    H, W = img.shape[:2]
    keep = _dil(lesion, 5)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
    r0 = rng.uniform(0.62, 0.75)
    t = np.clip((r - r0) / 0.12, 0, 1)
    fac = 1 - t * t * (3 - 2 * t) * 0.95
    dark = (img.astype(np.float32) * fac[..., None]).astype(np.uint8)
    return _blend_outside(img, dark, np.ones((H, W), np.float32), keep)


def hair(img: np.ndarray, lesion: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    import cv2
    H, W = img.shape[:2]
    keep = _dil(lesion, 5)
    layer = img.copy()
    alpha = np.zeros((H, W), np.uint8)
    for _ in range(rng.randint(6, 16)):
        p0 = rng.rand(2) * [W, H]
        p2 = p0 + rng.randn(2) * 0.35 * min(H, W)
        p1 = (p0 + p2) / 2 + rng.randn(2) * 0.12 * min(H, W)
        t = np.linspace(0, 1, 60)[:, None]
        pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
        th = int(rng.randint(1, 4))
        col = tuple(int(c) for c in (np.array([35, 25, 20]) * rng.uniform(0.6, 1.4)).clip(0, 255))
        cv2.polylines(layer, [pts], False, col, th, cv2.LINE_AA)
        cv2.polylines(alpha, [pts], False, 255, th, cv2.LINE_AA)
    return _blend_outside(img, layer, alpha.astype(np.float32) / 255.0 * 0.9, keep)


FUNCS = {"ruler": ruler, "ink": ink, "vignette": vignette, "hair": hair}


def apply_artefact(x01, lesion, kind: str, seeds):
    """x01 torch (N,3,H,W) in [0,1]; lesion torch (N,H,W) (>0.5 = lesion); seeds iterable of ints."""
    import torch
    imgs = (x01.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    les = (lesion.detach().float().cpu().numpy() > 0.5)
    out = np.stack([FUNCS[kind](im, l, np.random.RandomState(int(s) % (2 ** 31))) for im, l, s in
                    zip(imgs, les, seeds)])
    return torch.from_numpy(out).to(x01.device).permute(0, 3, 1, 2).float() / 255.0
