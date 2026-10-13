"""Thin data adapter for the EBC core code (does not modify cbm.data).

Adds on top of ``cbm.data.datasets.MedConceptDataset``:
  * joint geometric augmentation applied identically to the image, its K+1 masks and (for I1) the
    counterfactual edits of the same image;
  * masks at ``mask_size`` (default 128) as *soft* area-averaged targets, ordered
    ``concept_names + ['lesion']`` (plane K = lesion), plus ``mask_valid`` (K+1,);
  * image-level concept labels re-derived after random cropping (a structure that is cropped out
    becomes absent; the manifest presence = "mask has any pixel");
  * ``area_true`` (K,) = structure area / lesion area (manifest ``<c>_area``);
  * optional counterfactual pair (x_minus_c, x_ctrl_c) sampled from the cf index (see
    cbm.train.counterfactual) with updated concept/mask targets for the removed edit;
  * ``subset`` for smoke tests (stratified random subset, deterministic).

Batch dict keys: image (3,S,S) normalised, image_id, idx, label, concepts (K,) NaN=unknown,
concept_known (K,), masks (K+1,m,m) float in [0,1], mask_valid (K+1,), area_true (K,)
[, cf_has, cf_minus, cf_ctrl, cf_concepts_minus, cf_masks_minus, cf_concept (int)].
"""
from __future__ import annotations

import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from cbm.data.datasets import DATASETS, IMAGENET_MEAN, IMAGENET_STD, MedConceptDataset, load_manifest
from cbm.paths import CF_DIR, PROCESSED, SPLITS  # noqa: F401

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
HOLDOUT_DIR = PROCESSED


def val_holdout_ids(name: str = "isic2018_task12", frac: float = 0.12, seed: int = 0) -> list[str]:
    """Image ids carved out of the official *train* split and merged into 'val' (model selection / early stopping).

    Stratified by label over in-task train rows (label >= 0), sklearn ``train_test_split(test_size=frac,
    random_state=seed)``.  The ids are written once to data/processed/<name>/val_holdout_seed<seed>.csv and reused
    afterwards (the file is canonical: every method must read it); if the file exists its content wins regardless
    of ``frac``.  The ids used in the paper ship with the repository (data/splits/) and are copied into place when
    no hold-out file exists yet."""
    path = HOLDOUT_DIR / name / f"val_holdout_seed{seed}.csv"
    shipped = SPLITS / f"{name}_val_holdout_seed{seed}.csv"
    if not path.exists() and shipped.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped, path)
    if path.exists():
        return pd.read_csv(path)["image_id"].astype(str).tolist()
    from sklearn.model_selection import train_test_split
    df = load_manifest(name)
    tr = df[(df["split"] == "train") & (df["label"] >= 0)].sort_values("image_id").reset_index(drop=True)
    _, ho = train_test_split(tr, test_size=frac, random_state=seed, stratify=tr["label"])
    ho = ho.sort_values("image_id")[["image_id", "label", "label_name"]].assign(orig_split="train", frac=frac,
                                                                                 seed=seed)
    tmp = path.with_suffix(".csv.tmp")
    ho.to_csv(tmp, index=False)
    if not path.exists():           # first writer wins (atomic rename)
        os.replace(tmp, path)
    else:
        tmp.unlink(missing_ok=True)
    return pd.read_csv(path)["image_id"].astype(str).tolist()


def manifest_with_holdout(name: str, frac: float, seed: int = 0) -> pd.DataFrame:
    """Manifest where the held-out train ids have split='val' (official val kept, official test untouched)."""
    df = load_manifest(name)
    if frac and frac > 0:
        ids = set(val_holdout_ids(name, frac, seed))
        m = df["image_id"].astype(str).isin(ids) & (df["split"] == "train")
        df.loc[m, "split"] = "val"
    return df


def normalize(x01: torch.Tensor) -> torch.Tensor:
    m, s = MEAN.to(x01.device, x01.dtype), STD.to(x01.device, x01.dtype)
    if x01.dim() == 4:
        m, s = m[None], s[None]
    return (x01 - m) / s


def denormalize(x: torch.Tensor) -> torch.Tensor:
    m, s = MEAN.to(x.device, x.dtype), STD.to(x.device, x.dtype)
    if x.dim() == 4:
        m, s = m[None], s[None]
    return (x * s + m).clamp(0, 1)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float().div_(255.0)


class AugParams:
    """Random augmentation parameters, sampled once and applied to several aligned images/masks."""

    def __init__(self, w: int, h: int, train: bool, scale_min: float = 0.7, rot90: bool = True,
                 color_jitter: float = 0.1, rng: random.Random | None = None):
        r = rng or random
        self.box = (0, 0, w, h)
        self.ops: list[int] = []
        self.jit = (1.0, 1.0, 1.0)
        if not train:
            return
        s = r.uniform(scale_min, 1.0)
        cw, ch = max(1, int(w * math.sqrt(s))), max(1, int(h * math.sqrt(s)))
        x0, y0 = r.randint(0, w - cw), r.randint(0, h - ch)
        self.box = (x0, y0, x0 + cw, y0 + ch)
        if r.random() < 0.5:
            self.ops.append(Image.FLIP_LEFT_RIGHT)
        if r.random() < 0.5:
            self.ops.append(Image.FLIP_TOP_BOTTOM)
        if rot90 and r.random() < 0.5:
            self.ops.append(Image.ROTATE_90)
        if color_jitter > 0:
            j = color_jitter
            self.jit = (r.uniform(1 - j, 1 + j), r.uniform(1 - j, 1 + j), r.uniform(1 - j, 1 + j))

    def image(self, img: Image.Image, size: int) -> torch.Tensor:
        img = img.crop(self.box).resize((size, size), Image.BICUBIC)
        for op in self.ops:
            img = img.transpose(op)
        b, c, s = self.jit
        if b != 1.0:
            img = ImageEnhance.Brightness(img).enhance(b)
        if c != 1.0:
            img = ImageEnhance.Contrast(img).enhance(c)
        if s != 1.0:
            img = ImageEnhance.Color(img).enhance(s)
        return _to_tensor(img)

    def mask(self, m: Image.Image, size: int) -> torch.Tensor:
        """Soft (area-averaged) mask in [0,1] at size x size."""
        m = m.crop(self.box).resize((size, size), Image.BOX)
        for op in self.ops:
            m = m.transpose(op)
        return torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)


class DermDataset(MedConceptDataset):
    """ISIC-Derm (or any registry dataset with masks) for the EBC models."""

    def __init__(self, name: str = "isic2018_task12", split="train", *, train: bool = False, image_size: int = 448,
                 mask_size: int = 128, aug: dict | None = None, subset: int = 0, seed: int = 0,
                 cf_index: str | None = None, cf_frac: float = 0.0, cf_splits: tuple = ("train",),
                 full_masks: bool = False, drop_unlabeled: bool = True, fold: int | None = None,
                 val_from_train_frac: float = 0.0, holdout_seed: int = 0):
        man = manifest_with_holdout(name, val_from_train_frac, holdout_seed) if val_from_train_frac else None
        super().__init__(name, split, transform=lambda im: im, return_masks=False, image_size=image_size,
                         drop_unlabeled=drop_unlabeled, fold=fold, manifest=man)
        if subset and subset < len(self.df):
            rs = np.random.RandomState(seed)
            # stratified by label, deterministic
            idx = []
            for lab in np.unique(self.labels):
                ii = np.flatnonzero(self.labels == lab)
                k = max(1, int(round(subset * len(ii) / len(self.df))))
                idx.extend(rs.choice(ii, min(k, len(ii)), replace=False).tolist())
            idx = sorted(idx)[:subset] if len(idx) > subset else sorted(idx)
            self._reindex(np.array(idx))
        self.train = train
        self.image_size, self.mask_size = image_size, mask_size
        self.aug = {"scale_min": 0.7, "rot90": True, "color_jitter": 0.1, **(aug or {})}
        self.K = len(self.concept_names)
        # plane order: concepts then lesion
        self.plane_names = list(self.concept_names) + ["lesion"]
        cols = [f"mask_{m}" for m in self.plane_names]
        for c in cols:
            if c not in self.df.columns:
                self.df[c] = np.nan
        self.plane_paths = self.df[cols].to_numpy(object)
        acols = [f"{c}_area" for c in self.concept_names]
        # structure area / lesion area, clipped at 1 (attribute masks may extend beyond the lesion mask)
        self.area_true = (np.minimum(self.df[acols].to_numpy(np.float32), 1.0) if all(a in self.df for a in acols)
                          else np.full((len(self.df), self.K), np.nan, np.float32))
        self.full_masks = full_masks
        self.seed = seed
        self.rng = random.Random(seed)
        # counterfactual edits
        self.cf_frac = cf_frac
        self.cf_by_image: dict[str, list[dict]] = {}
        if cf_frac > 0 and cf_index and os.path.exists(cf_index):
            cf = pd.read_csv(cf_index)
            cf = cf[cf.split.isin(cf_splits)] if "split" in cf else cf
            for iid, g in cf.groupby("image_id"):
                pairs = []
                for c, gg in g.groupby("concept"):
                    mi = gg[gg.kind == "minus"]
                    ct = gg[gg.kind == "ctrl"]
                    if len(mi) and len(ct):
                        pairs.append({"concept": c, "minus": mi.iloc[0].to_dict(), "ctrl": ct.iloc[0].to_dict()})
                if pairs:
                    self.cf_by_image[str(iid)] = pairs
        elif cf_frac > 0:
            raise FileNotFoundError(f"cf_frac>0 but cf index {cf_index} not found; run cbm.train.counterfactual first")

    def _reindex(self, idx: np.ndarray) -> None:
        self.df = self.df.iloc[idx].reset_index(drop=True)
        self.mask_paths = self.mask_paths[idx]
        self.labels = self.df["label"].to_numpy(np.int64)
        self.paths = self.df["image_path"].tolist()
        self.ids = self.df["image_id"].astype(str).tolist()
        self.concepts = self.df[self.concept_names].to_numpy(np.float32) if self.concept_names else \
            np.zeros((len(self.df), 0), np.float32)

    def worker_init(self, wid: int) -> None:
        self.rng = random.Random(self.seed * 1000 + wid + int(torch.initial_seed() % 100000))

    def _load_planes(self, i: int, size_wh) -> tuple[list[Image.Image], list[bool]]:
        ms, valid = [], []
        for p in self.plane_paths[i]:
            ok = isinstance(p, str) and len(p) > 0 and os.path.exists(p)
            valid.append(ok)
            ms.append(Image.open(p).convert("L") if ok else Image.new("L", size_wh, 0))
        return ms, valid

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB")
        w, h = img.size
        ap = AugParams(w, h, self.train, self.aug["scale_min"], self.aug["rot90"], self.aug["color_jitter"], self.rng)
        planes, valid = self._load_planes(i, img.size)
        x = ap.image(img, self.image_size)
        m = torch.stack([ap.mask(p, self.mask_size) for p in planes])         # (K+1, m, m)
        c = torch.from_numpy(self.concepts[i].copy())
        valid_t = torch.tensor(valid)
        if self.train and ap.box != (0, 0, w, h):
            # structure cropped away entirely -> absent
            gone = (m[:self.K].flatten(1).amax(1) <= 0) & valid_t[:self.K] & (c == 1)
            c = torch.where(gone, torch.zeros_like(c), c)
        out = {"image": normalize(x), "image_id": self.ids[i], "idx": i, "label": int(self.labels[i]),
               "concepts": c, "concept_known": ~torch.isnan(c), "masks": m, "mask_valid": valid_t,
               "area_true": torch.from_numpy(self.area_true[i].copy())}
        if self.full_masks:
            out["masks_full"] = torch.stack([ap.mask(p, self.image_size) for p in planes])
        if self.cf_frac > 0:
            out.update(self._cf_item(i, ap, c, m))
        return out

    def _cf_item(self, i, ap, c, m):
        S, Ms, K = self.image_size, self.mask_size, self.K
        pairs = self.cf_by_image.get(self.ids[i], [])
        # only pairs whose concept is still present after cropping
        pairs = [p for p in pairs if p["concept"] in self.concept_names and
                 c[self.concept_names.index(p["concept"])] == 1]
        if not pairs or self.rng.random() >= self.cf_frac:
            z = torch.zeros(3, S, S)
            return {"cf_has": False, "cf_minus": z, "cf_ctrl": z.clone(), "cf_concepts_minus": torch.zeros(K),
                    "cf_masks_minus": torch.zeros(K + 1, Ms, Ms), "cf_concept": -1}
        p = pairs[self.rng.randrange(len(pairs))]
        xm = ap.image(Image.open(p["minus"]["path"]).convert("RGB"), S)
        xc = ap.image(Image.open(p["ctrl"]["path"]).convert("RGB"), S)
        reg = ap.mask(Image.open(p["minus"]["region_path"]).convert("L"), Ms)    # removed region
        ci = self.concept_names.index(p["concept"])
        mm = m.clone()
        mm[:K] = mm[:K] * (1 - reg)                       # removed region: no structure left there
        mm[ci] = 0
        cm = c.clone()
        for k, cn in enumerate(self.concept_names):
            col = f"post_{cn}"
            if col in p["minus"] and not (isinstance(p["minus"][col], float) and math.isnan(p["minus"][col])):
                cm[k] = torch.minimum(cm[k], torch.tensor(float(p["minus"][col])))
        cm[ci] = 0
        return {"cf_has": True, "cf_minus": normalize(xm), "cf_ctrl": normalize(xc), "cf_concepts_minus": cm,
                "cf_masks_minus": mm, "cf_concept": ci}


def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.stack(v)
        elif isinstance(v[0], (bool, np.bool_)):
            out[k] = torch.tensor(v, dtype=torch.bool)
        elif isinstance(v[0], (int, np.integer)):
            out[k] = torch.tensor(v, dtype=torch.long)
        else:
            out[k] = v
    return out


def make_loader(ds: Dataset, batch_size: int, train: bool, num_workers: int = 8, sampler=None, seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    wi = getattr(ds, "worker_init", None)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=(train and sampler is None), sampler=sampler, num_workers=num_workers,
        collate_fn=collate, drop_last=train, pin_memory=True, persistent_workers=train and num_workers > 0,
        worker_init_fn=wi, generator=g)


# ------------------------------------------------------------------ PH2 (external test)
class PH2Dataset(Dataset):
    """PH2 for external testing: image at image_size (eval transform), lesion mask, PH2 label + concepts."""

    def __init__(self, image_size: int = 448, mask_size: int = 128):
        df = load_manifest("ph2")
        self.df = df[df["split"] != "excluded"].reset_index(drop=True) if "split" in df else df
        self.image_size, self.mask_size = image_size, mask_size
        self.info = DATASETS["ph2"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["image_path"]).convert("RGB")
        ap = AugParams(*img.size, train=False)
        out = {"image": normalize(ap.image(img, self.image_size)), "image_id": str(r["image_id"]), "idx": i,
               "label": int(r["label"])}
        p = r.get("mask_lesion")
        ok = isinstance(p, str) and os.path.exists(p)
        out["lesion"] = ap.mask(Image.open(p).convert("L"), self.mask_size) if ok else \
            torch.zeros(self.mask_size, self.mask_size)
        out["lesion_valid"] = bool(ok)
        cn = self.info["concept_names"]
        out["ph2_concepts"] = torch.tensor(r[cn].to_numpy(np.float32))
        return out


def dataset_kwargs(cfg: dict) -> dict:
    """DermDataset kwargs shared by trainer / evaluator for a run config."""
    dc = cfg.get("data", {}) or {}
    return dict(image_size=dc.get("image_size", 448), mask_size=dc.get("mask_size", 128), seed=cfg.get("seed", 0),
                val_from_train_frac=dc.get("val_from_train_frac", 0.0), holdout_seed=dc.get("holdout_seed", 0))
