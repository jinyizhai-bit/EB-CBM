"""One torch Dataset for every prepared CBM dataset.

    from cbm.data.datasets import MedConceptDataset, DATASETS
    ds = MedConceptDataset("isic2018_task12", "train", transform=my_tf, return_masks=False)
    item = ds[0]   # dict: image, label, concepts, concept_known, image_id [, masks, mask_valid]

* ``concepts`` is a float32 tensor in registry order (``DATASETS[name]['concept_names']``); unknown
  values are NaN and ``concept_known`` is the matching bool tensor (use it to mask the concept loss).
* ``transform`` acts on the PIL RGB image only.  Default: resize to ``image_size`` x ``image_size``,
  ToTensor, ImageNet normalisation.
* With ``return_masks=True`` the item also has ``masks`` (float {0,1}, [K, H, W], K = len(mask_names))
  and ``mask_valid`` ([K] bool, False where the dataset has no such mask for this image).  If you
  use geometric augmentation, pass ``joint_transform(img, list_of_mask_PIL) -> (img_tensor, mask_tensor)``
  (e.g. ``JointTransform``) so masks stay aligned; otherwise masks are resized (nearest) to the
  output image size, which is only correct for a pure-resize ``transform``.
* ``split`` may be a string, a list of strings, or ``'all'`` (every split except 'excluded').
* For ``ph2`` pass ``fold=k`` to use CV fold k as 'test' and the other folds as 'train'.
* Rows with ``label < 0`` (ISIC 2018 T1-2 'other' diagnoses) are dropped unless ``drop_unlabeled=False``.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .registry import DATASETS, PROCESSED  # noqa: F401  (DATASETS re-exported)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = ["MedConceptDataset", "DATASETS", "JointTransform", "default_transform", "load_manifest"]


def load_manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS[name]["manifest"], low_memory=False)


def default_transform(image_size: int = 224, train: bool = False):
    import torchvision.transforms as T
    tf = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        tf += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    tf += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(tf)


class JointTransform:
    """Resize (+ optional random flips / 90-degree rotations / scale-crop) applied identically to image and masks."""

    def __init__(self, image_size: int = 224, train: bool = False, rot90: bool = True,
                 scale: tuple[float, float] = (0.6, 1.0), normalize: bool = True):
        self.size, self.train, self.rot90, self.scale, self.normalize = image_size, train, rot90, scale, normalize

    def __call__(self, img: Image.Image, masks: Sequence[Image.Image]):
        import torchvision.transforms.functional as F
        if self.train:
            w, h = img.size
            s = random.uniform(*self.scale)
            cw, ch = max(1, int(w * math.sqrt(s))), max(1, int(h * math.sqrt(s)))
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
            masks = [m.crop((x0, y0, x0 + cw, y0 + ch)) for m in masks]
        img = img.resize((self.size, self.size), Image.BICUBIC)
        masks = [m.resize((self.size, self.size), Image.NEAREST) for m in masks]
        if self.train:
            ops = []
            if random.random() < 0.5:
                ops.append(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                ops.append(Image.FLIP_TOP_BOTTOM)
            if self.rot90 and random.random() < 0.5:
                ops.append(Image.ROTATE_90)
            for op in ops:
                img = img.transpose(op)
                masks = [m.transpose(op) for m in masks]
        x = F.to_tensor(img)
        if self.normalize:
            x = F.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        m = torch.from_numpy(np.stack([np.asarray(mm) > 127 for mm in masks])).float() if masks else \
            torch.zeros(0, self.size, self.size)
        return x, m


class MedConceptDataset(Dataset):
    def __init__(self, name: str, split: str | Sequence[str] = "train", transform: Callable | None = None,
                 return_masks: bool = False, *, joint_transform: Callable | None = None, image_size: int = 224,
                 drop_unlabeled: bool = True, fold: int | None = None, concepts: Sequence[str] | None = None,
                 manifest: pd.DataFrame | None = None):
        if name not in DATASETS:
            raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
        self.name, self.info = name, DATASETS[name]
        df = load_manifest(name) if manifest is None else manifest.copy()
        if fold is not None:
            if "fold" not in df:
                raise ValueError(f"{name} has no CV folds")
            df["split"] = np.where(df["fold"] == fold, "test", np.where(df["split"] == "excluded", "excluded",
                                                                          "train"))
        if isinstance(split, str):
            splits = [s for s in df["split"].unique() if s != "excluded"] if split == "all" else [split]
        else:
            splits = list(split)
        unknown = set(splits) - set(df["split"].unique())
        if unknown:
            raise ValueError(f"{name}: unknown split(s) {unknown}; available {sorted(df['split'].unique())}")
        df = df[df["split"].isin(splits)]
        if drop_unlabeled:
            df = df[df["label"] >= 0]
        self.df = df.reset_index(drop=True)
        self.split = split
        self.class_names = list(self.info["class_names"])
        self.concept_names = list(concepts) if concepts is not None else list(self.info["concept_names"])
        self.mask_names = list(self.info["mask_names"])
        self.transform = transform if transform is not None else default_transform(image_size)
        self.joint_transform = joint_transform
        self.return_masks = return_masks and len(self.mask_names) > 0
        self.image_size = image_size
        self.labels = self.df["label"].to_numpy(np.int64)
        self.paths = self.df["image_path"].tolist()
        self.ids = self.df["image_id"].astype(str).tolist()
        c = self.df[self.concept_names].to_numpy(np.float32) if self.concept_names else \
            np.zeros((len(self.df), 0), np.float32)
        self.concepts = c
        self.mask_paths = self.df[[f"mask_{m}" for m in self.mask_names]].to_numpy(object) if self.mask_names else \
            np.empty((len(self.df), 0), object)

    # ------------------------------------------------------------------ helpers
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_concepts(self) -> int:
        return len(self.concept_names)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_classes)

    def concept_pos_rate(self) -> np.ndarray:
        return np.nanmean(self.concepts, axis=0) if self.n_concepts else np.zeros(0)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB")
        c = torch.from_numpy(self.concepts[i].copy())
        out = {"image_id": self.ids[i], "label": int(self.labels[i]), "concepts": c, "concept_known": ~torch.isnan(c)}
        if self.return_masks:
            valid, ms = [], []
            for p in self.mask_paths[i]:
                ok = isinstance(p, str) and len(p) > 0
                valid.append(ok)
                ms.append(Image.open(p).convert("L") if ok else Image.new("L", img.size, 0))
            if self.joint_transform is not None:
                x, m = self.joint_transform(img, ms)
            else:
                x = self.transform(img)
                h, w = x.shape[-2:]
                m = torch.from_numpy(np.stack([np.asarray(mm.resize((w, h), Image.NEAREST)) > 127 for mm in ms]))
                m = m.float()
            out["image"], out["masks"], out["mask_valid"] = x, m, torch.tensor(valid)
        else:
            out["image"] = self.transform(img)
        return out

    def __repr__(self) -> str:
        return (f"MedConceptDataset({self.name!r}, split={self.split!r}, n={len(self)}, classes={self.n_classes}, "
                f"concepts={self.n_concepts}, masks={self.mask_names if self.return_masks else []})")
