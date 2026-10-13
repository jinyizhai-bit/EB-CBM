"""I1 - counterfactually grounded concepts: LaMa preprocessing + training loss.

Preprocessing (run once, GPU):
    CUDA_VISIBLE_DEVICES=0 python -m cbm.train.counterfactual \
        --splits train val test [--limit N] [--batch 8]

For every ISIC-Derm image (label >= 0) and every present concept c (expert mask non-empty):
  * ``minus_<c>``: LaMa inpainting of the expert mask of c dilated by 3 px (structure removed);
  * ``ctrl_<c>`` : LaMa inpainting of a control region R_ctrl with the same pixel area as the dilated
    mask, inside the lesion and overlapping no structure (all 5 masks dilated by 3 px).  R_ctrl is,
    in order of preference: (1) the dilated structure mask itself translated / flipped to a free location
    ("shift", shape-matched); (2) a compact blob grown inside the free lesion area from a random seed
    ("blob"); (3) if the free lesion area is smaller than the target, the largest connected free region
    ("largest_free"); (4) if the lesion has no free pixels, free skin outside the lesion ("outside_lesion").
Outputs (512-px cache resolution, i.e. same size as data/processed/isic2018_task12/images_512):
    cf_512/<image_id>__<edit>.jpg   (JPEG q95; outside the region pixels are copied from the cache image)
    cf_512/regions/<image_id>__<edit>.png  (inpainted region, 0/255)
    cf_512/index.csv  columns: image_id, split, label, edit, kind (minus|ctrl), concept, path, region_path,
                      area_px, target_px, method, overlap_px, post_<concept> (presence after the edit)
Presence after a ``minus`` edit: c -> 0; other concept c' stays present iff >= 10% of its pixels survive
outside the inpainted region.  ``ctrl`` edits keep every presence label.

Training loss (``cf_loss``): removed edits -> concept c target 0, mask c target 0, other targets updated as
above; control edits -> original targets + KL consistency between predictions on x and x_ctrl.
The evaluation metrics CLS / XE / diagnosis shift are in cbm.eval.evaluate_run (``eval_counterfactual``).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from cbm.paths import CF_DIR as OUT_DIR, LAMA_PT as _LAMA_PT  # noqa: E402

LAMA_PT = str(_LAMA_PT)          # TorchScript big-lama (scripts/download_lama.sh or $CBM_LAMA_PT)
DILATE_PX = 3
KEEP_FRAC = 0.10


# ============================================================================ region construction
def _disk(r: int) -> np.ndarray:
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(m: np.ndarray, r: int = DILATE_PX) -> np.ndarray:
    import cv2
    return cv2.dilate(m.astype(np.uint8), _disk(r)) > 0


def _seed(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def control_region(target: np.ndarray, lesion: np.ndarray, union_struct: np.ndarray, rng: np.random.RandomState,
                   n_shift: int = 60) -> tuple[np.ndarray, str]:
    """Return (R_ctrl bool mask, method). target = dilated structure mask (defines area and shape)."""
    import cv2
    H, W = target.shape
    area = int(target.sum())
    blocked = dilate(union_struct, DILATE_PX)
    free = lesion & ~blocked
    # (1) shape-matched translation (+ flips) fully inside the free lesion area
    ys, xs = np.nonzero(target)
    if area > 0 and free.sum() >= area:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        shape = target[y0:y1 + 1, x0:x1 + 1]
        fy, fx = np.nonzero(free)
        for _ in range(n_shift):
            sh = shape
            if rng.rand() < 0.5:
                sh = sh[:, ::-1]
            if rng.rand() < 0.5:
                sh = sh[::-1, :]
            h, w = sh.shape
            if h > H or w > W:
                continue
            j = rng.randint(len(fy))
            ty, tx = fy[j] - h // 2, fx[j] - w // 2
            ty, tx = int(np.clip(ty, 0, H - h)), int(np.clip(tx, 0, W - w))
            sub = free[ty:ty + h, tx:tx + w]
            if not (sub | ~sh).all():
                continue
            R = np.zeros_like(target)
            R[ty:ty + h, tx:tx + w] = sh
            return R, "shift"
    # (2) compact blob grown inside the free lesion area (component with enough area)
    if area > 0 and free.sum() >= area:
        n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= area]
        if big:
            comp = lab == big[rng.randint(len(big))]
            cy, cx = np.nonzero(comp)
            j = rng.randint(len(cy))
            # geodesic-ish growth: distance transform from the seed restricted to the component
            seed = np.ones_like(comp, np.uint8)
            seed[cy[j], cx[j]] = 0
            dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
            dist[~comp] = np.inf
            order = np.argsort(dist.ravel(), kind="stable")[:area]
            R = np.zeros(H * W, bool)
            R[order] = True
            return R.reshape(H, W) & comp, "blob"
        # (3) union of free components is large enough but no single one: take the largest component(s)
    if free.sum() > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        comp = lab == i
        if comp.sum() > area > 0:  # (only when free total >= area but fragmented): crop the component to area
            cy, cx = np.nonzero(comp)
            j = rng.randint(len(cy))
            seed = np.ones_like(comp, np.uint8)
            seed[cy[j], cx[j]] = 0
            dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
            dist[~comp] = np.inf
            order = np.argsort(dist.ravel(), kind="stable")[:area]
            R = np.zeros(H * W, bool)
            R[order] = True
            comp = R.reshape(H, W) & comp
        return comp, "largest_free"
    # (4) no free lesion pixels: free skin outside the lesion
    free_out = ~lesion & ~blocked
    if free_out.sum() > 0 and area > 0:
        cy, cx = np.nonzero(free_out)
        j = rng.randint(len(cy))
        seed = np.ones_like(free_out, np.uint8)
        seed[cy[j], cx[j]] = 0
        dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
        dist[~free_out] = np.inf
        order = np.argsort(dist.ravel(), kind="stable")[:area]
        R = np.zeros(H * W, bool)
        R[order] = True
        return R.reshape(H, W) & free_out, "outside_lesion"
    return np.zeros_like(target), "none"


# ============================================================================ LaMa
class LaMa:
    """TorchScript big-lama (from simple-lama-inpainting): forward(image[0,1] (N,3,H,W), mask{0,1} (N,1,H,W))."""

    def __init__(self, path: str = LAMA_PT, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device
        # the profiling executor re-optimises the graph for every new input shape (minutes); disable it
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        if not os.path.exists(path):
            raise FileNotFoundError(f"LaMa checkpoint not found: {path} (run scripts/download_lama.sh or set "
                                    "CBM_LAMA_PT)")
        self.model = torch.jit.load(path, map_location=device).eval()

    def __call__(self, imgs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """imgs uint8 (N,H,W,3), masks bool (N,H,W) -> uint8 (N,H,W,3), composited outside the mask."""
        torch = self.torch
        N, H, W, _ = imgs.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        x = torch.from_numpy(imgs).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        m = torch.from_numpy(masks.astype(np.float32)).to(self.device)[:, None]
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
            m = torch.nn.functional.pad(m, (0, pw, 0, ph), mode="constant", value=0)
        with torch.no_grad():
            y = self.model(x, m)[:, :, :H, :W]
        y = y.clamp(0, 1)
        m0 = m[:, :, :H, :W]
        y = m0 * y + (1 - m0) * x[:, :, :H, :W]
        return (y * 255.0 + 0.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


# ============================================================================ preprocessing driver
def build_jobs(df, concepts: list[str], limit: int = 0):
    """Yield per-image job dicts with all edits (regions computed on CPU)."""
    from PIL import Image
    for n, (_, r) in enumerate(df.iterrows()):
        if limit and n >= limit:
            break
        img = np.asarray(Image.open(r["image_path"]).convert("RGB"))
        H, W = img.shape[:2]
        ms = {}
        for c in concepts + ["lesion"]:
            p = r.get(f"mask_{c}")
            ms[c] = (np.asarray(Image.open(p).convert("L")) > 127) if isinstance(p, str) and os.path.exists(p) \
                else np.zeros((H, W), bool)
        union = np.zeros((H, W), bool)
        for c in concepts:
            union |= ms[c]
        lesion = ms["lesion"] if ms["lesion"].any() else np.ones((H, W), bool)
        edits = []
        for c in concepts:
            if not ms[c].any():
                continue
            Rm = dilate(ms[c], DILATE_PX)
            post = {}
            for c2 in concepts:
                if c2 == c:
                    post[c2] = 0
                elif ms[c2].any():
                    post[c2] = int((ms[c2] & ~Rm).sum() >= KEEP_FRAC * ms[c2].sum())
                else:
                    post[c2] = 0
            edits.append(dict(edit=f"minus_{c}", kind="minus", concept=c, region=Rm, target_px=int(Rm.sum()),
                              method="dilate3", overlap_px=0, post=post))
            rng = np.random.RandomState(_seed(r["image_id"], c))
            Rc, meth = control_region(Rm, lesion, union, rng)
            post_c = {c2: int(ms[c2].any()) for c2 in concepts}
            edits.append(dict(edit=f"ctrl_{c}", kind="ctrl", concept=c, region=Rc, target_px=int(Rm.sum()),
                              method=meth, overlap_px=int((Rc & union).sum()), post=post_c))
        yield dict(image_id=str(r["image_id"]), split=r["split"], label=int(r["label"]), img=img, edits=edits)


def run_preprocess(splits: list[str], limit: int = 0, batch: int = 8, overwrite: bool = False,
                   device: str = "cuda") -> Path:
    import pandas as pd
    from PIL import Image
    from cbm.data.datasets import DATASETS, load_manifest
    concepts = list(DATASETS["isic2018_task12"]["concept_names"])
    df = load_manifest("isic2018_task12")
    df = df[df.split.isin(splits) & (df.label >= 0)].reset_index(drop=True)
    (OUT_DIR / "regions").mkdir(parents=True, exist_ok=True)
    lama = LaMa(device=device)
    rows, pend = [], []      # pend: (row_dict, img, region)
    t0, n_done = time.time(), 0

    def flush():
        nonlocal pend, n_done
        if not pend:
            return
        by_size: dict = {}
        for it in pend:
            by_size.setdefault(it[1].shape, []).append(it)
        for _, items in by_size.items():
            for k in range(0, len(items), batch):
                chunk = items[k:k + batch]
                out = lama(np.stack([c[1] for c in chunk]), np.stack([c[2] for c in chunk]))
                for (row, _, _), o in zip(chunk, out):
                    Image.fromarray(o).save(row["path"], quality=95)
                    n_done += 1
        pend = []

    for job in build_jobs(df, concepts, limit):
        for e in job["edits"]:
            name = f"{job['image_id']}__{e['edit']}"
            path, rpath = OUT_DIR / f"{name}.jpg", OUT_DIR / "regions" / f"{name}.png"
            row = dict(image_id=job["image_id"], split=job["split"], label=job["label"], edit=e["edit"],
                       kind=e["kind"], concept=e["concept"], path=str(path), region_path=str(rpath),
                       area_px=int(e["region"].sum()), target_px=e["target_px"], method=e["method"],
                       overlap_px=e["overlap_px"], **{f"post_{k}": v for k, v in e["post"].items()})
            rows.append(row)
            if overwrite or not rpath.exists():
                Image.fromarray(e["region"].astype(np.uint8) * 255).save(rpath)
            if overwrite or not path.exists():
                if e["region"].any():
                    pend.append((row, job["img"], e["region"]))
                else:
                    Image.fromarray(job["img"]).save(path, quality=95)
        if len(pend) >= 64:
            flush()
            el = time.time() - t0
            print(f"[cf] {len(rows)} edits indexed, {n_done} inpainted, {el:.0f}s", flush=True)
    flush()
    idx = pd.DataFrame(rows)
    out = OUT_DIR / "index.csv"
    if out.exists() and set(splits) != {"train", "val", "test"}:
        old = pd.read_csv(out)
        idx = pd.concat([old[~old.image_id.isin(idx.image_id)], idx], ignore_index=True)
    idx.to_csv(out, index=False)
    print(f"[cf] done: {len(idx)} edits in index ({n_done} inpainted now) in {time.time() - t0:.0f}s -> {out}")
    if len(idx):
        print(idx.groupby(["kind", "method"]).size())
    return out


# ============================================================================ training loss
def bern_kl(p_ref, p, eps: float = 1e-5):
    import torch
    p_ref, p = p_ref.clamp(eps, 1 - eps), p.clamp(eps, 1 - eps)
    return p_ref * torch.log(p_ref / p) + (1 - p_ref) * torch.log((1 - p_ref) / (1 - p))


def cf_loss(out_x: dict, out_minus: dict, out_ctrl: dict, c_true, c_minus, known, m_true, m_minus, m_valid,
            has_concepts: bool, use_mask: bool, kl_w: float = 1.0, pos_weight=None):
    """I1 loss on the CF part of a batch (rows already restricted to samples with an edit).

    out_*: model outputs for x / x_minus / x_ctrl (same rows); c_true / c_minus (B,K); known (B,K) bool;
    m_true / m_minus (B,K+1,h,w) soft targets; m_valid (B,K+1)."""
    import torch
    import torch.nn.functional as F
    from cbm.models.heads import mask_loss
    logs, total = {}, 0.0
    if has_concepts:
        kw = known.float()

        def bce(s, t):
            l = F.binary_cross_entropy_with_logits(s.float(), torch.nan_to_num(t, 0.0), reduction="none",
                                                   pos_weight=pos_weight)
            return (l * kw).sum() / kw.sum().clamp_min(1)
        l_minus = bce(out_minus["concept_logits"], c_minus)
        l_ctrl = bce(out_ctrl["concept_logits"], c_true)
        p_ref = torch.sigmoid(out_x["concept_logits"].float()).detach()
        kl_c = (bern_kl(p_ref, torch.sigmoid(out_ctrl["concept_logits"].float())) * kw).sum() / kw.sum().clamp_min(1)
        total = total + l_minus + l_ctrl + kl_w * kl_c
        logs.update(cf_minus=float(l_minus.detach()), cf_ctrl=float(l_ctrl.detach()), cf_kl_c=float(kl_c.detach()))
    if use_mask and out_minus.get("masks") is not None:
        lm, _ = mask_loss(out_minus["extra"]["mask_logits"], m_minus, m_valid)
        lc, _ = mask_loss(out_ctrl["extra"]["mask_logits"], m_true, m_valid)
        total = total + lm + lc
        logs.update(cf_mask=float((lm + lc).detach()))
    py = F.softmax(out_x["logits"].float(), -1).detach()
    kl_y = F.kl_div(F.log_softmax(out_ctrl["logits"].float(), -1), py, reduction="batchmean")
    total = total + kl_w * kl_y
    logs.update(cf_kl_y=float(kl_y.detach()))
    return total, logs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run_preprocess(a.splits, a.limit, a.batch, a.overwrite)
