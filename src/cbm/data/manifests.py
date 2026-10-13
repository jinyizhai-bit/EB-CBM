"""Build manifests and 512-px image/mask caches for the datasets used by EB-CBM.

    python -m cbm.data.manifests isic2018_task12 ph2 [--overwrite]
    python -m cbm.data.manifests all

Every ``build_<dataset>()`` writes ``data/processed/<dataset>/manifest.csv`` with the columns

    image_id, split, image_path (512 cache), orig_path, label (int), label_name,
    <concept columns> (0/1, NaN = unknown), mask_<name> (512 cache png, NaN if absent),
    [<concept>_area, <concept>_area_img, lesion_area_img]  (ISIC 2018 T1-2 only),
    orig_width, orig_height, width, height, <metadata columns>

plus ``data/processed/<dataset>/build_stats.json`` with data checks and ``per_image_stats.csv``.
Images are resized so that the longer side is 512 px (LANCZOS, JPEG q95); images whose longer side
is already <= 512 px are re-encoded at native size (never upscaled).  Masks are binarised (>127),
resized with nearest neighbour to exactly the cached image size and saved as 0/255 png.
Nothing under data/raw is modified.  Existing cache files are reused unless ``--overwrite`` is passed.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .registry import ISIC18_ATTRS, ISIC_DERM3, METADATA, PH2_CLASSES, PH2_COLOURS, PH2_CONCEPTS, PROCESSED, RAW

Image.MAX_IMAGE_PIXELS = None
LONG_SIDE = 512
JPEG_Q = 95
N_WORKERS = min(60, os.cpu_count() or 8)
SEED = 0


# ============================================================================ image / mask helpers
def target_size(w: int, h: int, long_side: int = LONG_SIDE) -> tuple[int, int]:
    s = long_side / max(w, h)
    if s >= 1.0:
        return w, h
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def _bin_mask(path: str, shape_hw: tuple[int, int] | None) -> tuple[np.ndarray, bool, set]:
    """Load a mask as bool; resize (nearest) to shape_hw if it does not match. Returns (mask, mismatched, values)."""
    m = Image.open(path)
    if m.mode not in ("L", "1"):
        m = m.convert("L")
    a = np.asarray(m)
    a8 = a if a.dtype == np.uint8 else a.astype(np.uint8)
    vals = set(np.flatnonzero(np.bincount(a8.ravel(), minlength=256)).tolist())
    mism = False
    if shape_hw is not None and a.shape[:2] != tuple(shape_hw):
        mism = True
        a = np.asarray(Image.fromarray(a).resize((shape_hw[1], shape_hw[0]), Image.NEAREST))
    if m.mode == "1":
        return a.astype(bool), mism, vals
    return a > 127, mism, vals


def _superpixel_index(path: str) -> np.ndarray:
    sp = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    return sp[..., 0] + 256 * sp[..., 1] + 65536 * sp[..., 2]


def process_item(task: dict) -> dict:
    """Resize one image (+ its masks) into the cache and return per-image statistics.

    task keys: image_id, src, dst, mask_dir, masks {name: path | [paths] (union)},
               superpixel (sp_png, features_json) | None, sp_features [names], lesion (mask name for areas) | None,
               overwrite (bool)
    """
    out = {"image_id": task["image_id"]}
    try:
        im = Image.open(task["src"])
        W, H = im.size
        out.update(orig_width=W, orig_height=H, orig_mode=im.mode)
        try:
            out["exif_orientation"] = int(im.getexif().get(274, 1))
        except Exception:
            out["exif_orientation"] = 1
        tw, th = target_size(W, H)
        out.update(width=tw, height=th)
        dst = Path(task["dst"])
        if task.get("overwrite") or not dst.exists():
            if im.format == "JPEG" and (tw, th) != (W, H):
                im.draft("RGB", (tw * 2, th * 2))
            im = im.convert("RGB")
            if im.size != (tw, th):
                im = im.resize((tw, th), Image.LANCZOS)
            tmp = dst.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=JPEG_Q)
            os.replace(tmp, dst)
        im.close()

        masks: dict[str, np.ndarray] = {}
        for name, p in (task.get("masks") or {}).items():
            paths = p if isinstance(p, (list, tuple)) else [p]
            acc = None
            for q in paths:
                a, mism, vals = _bin_mask(q, (H, W))
                if mism:
                    out[f"{name}_shape_mismatch"] = True
                if vals and not vals <= {0, 1, 255}:
                    out[f"{name}_nonbinary_values"] = len(vals)
                acc = a if acc is None else (acc | a)
            masks[name] = acc
        if task.get("superpixel"):
            sp_png, fjson = task["superpixel"]
            idx = _superpixel_index(sp_png)
            feats = json.load(open(fjson))
            n_sp = len(next(iter(feats.values())))
            out["sp_max_index"] = int(idx.max())
            out["sp_n_list"] = n_sp
            out["sp_index_ok"] = bool(idx.max() == n_sp - 1 and all(len(v) == n_sp for v in feats.values()))
            if idx.shape != (H, W):
                out["superpixel_shape_mismatch"] = True
                idx = np.asarray(Image.fromarray(idx).resize((W, H), Image.NEAREST))
            idx = np.clip(idx, 0, n_sp - 1)
            for f in task["sp_features"]:
                lut = np.asarray(feats.get(f, [0] * n_sp), dtype=bool)
                masks[f] = lut[idx]
                out[f"{f}_n_superpixels"] = int(lut.sum())

        n_img = float(W * H)
        lesion = masks.get(task.get("lesion")) if task.get("lesion") else None
        n_les = float(lesion.sum()) if lesion is not None else np.nan
        mdir = Path(task["mask_dir"]) if masks else None
        for name, a in masks.items():
            cnt = float(a.sum())
            out[f"{name}_px"] = cnt
            out[f"{name}_frac_img"] = cnt / n_img
            if lesion is not None and name != task.get("lesion"):
                out[f"{name}_frac_lesion"] = cnt / n_les if n_les > 0 else np.nan
                out[f"{name}_frac_outside_lesion"] = float((a & ~lesion).sum()) / cnt if cnt > 0 else 0.0
            mdst = mdir / f"{task['image_id']}_{name}.png"
            small = Image.fromarray((a.astype(np.uint8) * 255)).resize((tw, th), Image.NEAREST)
            out[f"{name}_cache_px"] = int((np.asarray(small) > 0).sum())
            if task.get("overwrite") or not mdst.exists():
                tmp = mdst.with_suffix(".tmp.png")
                small.save(tmp, "PNG", optimize=False)
                os.replace(tmp, mdst)
            out[f"mask_{name}"] = str(mdst)
    except Exception as e:  # pragma: no cover - reported in build_stats
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_pool(tasks: list[dict], desc: str) -> pd.DataFrame:
    t0 = time.time()
    res = []
    step = max(1, len(tasks) // 10)
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(process_item, tasks, chunksize=2), 1):
            res.append(r)
            if i % step == 0 or i == len(tasks):
                print(f"[{desc}] {i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(res)
    if "error" in df and df["error"].notna().any():
        bad = df[df["error"].notna()]
        print(f"[{desc}] {len(bad)} ERRORS, e.g. {bad.iloc[0].to_dict()}", flush=True)
    return df


def _prep_dirs(name: str) -> tuple[Path, Path, Path]:
    root = PROCESSED / name
    (root / "images_512").mkdir(parents=True, exist_ok=True)
    (root / "masks_512").mkdir(parents=True, exist_ok=True)
    return root, root / "images_512", root / "masks_512"


def _task(name, image_id, src, masks=None, lesion=None, superpixel=None, sp_features=None, overwrite=False):
    root = PROCESSED / name
    return dict(image_id=image_id, src=str(src), dst=str(root / "images_512" / f"{image_id}.jpg"),
                mask_dir=str(root / "masks_512"), masks=masks or {}, lesion=lesion, superpixel=superpixel,
                sp_features=sp_features or [], overwrite=overwrite)


def _finalize(name: str, df: pd.DataFrame, concept_cols: list[str], mask_names: list[str], stats_df: pd.DataFrame,
              extra_first: list[str] | None = None, stats: dict | None = None) -> pd.DataFrame:
    """Merge per-image stats, order columns, write manifest.csv and build_stats.json."""
    root = PROCESSED / name
    keep = ["image_id", "orig_width", "orig_height", "width", "height"] + [f"mask_{m}" for m in mask_names
                                                                          if f"mask_{m}" in stats_df]
    df = df.merge(stats_df[keep], on="image_id", how="left")
    df["image_path"] = [str(root / "images_512" / f"{i}.jpg") for i in df["image_id"]]
    for m in mask_names:
        if f"mask_{m}" not in df:
            df[f"mask_{m}"] = np.nan
    first = ["image_id", "split", "image_path", "orig_path", "label", "label_name"]
    area = [c for c in df.columns if c.endswith("_area") or c.endswith("_area_img")]
    mask_cols = [f"mask_{m}" for m in mask_names]
    size_cols = ["orig_width", "orig_height", "width", "height"]
    order = first + concept_cols + mask_cols + area + (extra_first or []) + size_cols
    rest = [c for c in df.columns if c not in order]
    df = df[order + rest]
    assert df["image_id"].is_unique, f"{name}: duplicate image_id"
    df.to_csv(root / "manifest.csv", index=False)

    st = dict(stats or {})
    st["n_rows"] = int(len(df))
    st["split_counts"] = df["split"].value_counts().to_dict()
    st["errors"] = int(stats_df["error"].notna().sum()) if "error" in stats_df else 0
    st["missing_cache"] = int((~df["image_path"].map(os.path.exists)).sum())
    if "exif_orientation" in stats_df:
        st["exif_orientation_not1"] = int((stats_df["exif_orientation"].fillna(1) != 1).sum())
    for c in stats_df.columns:
        if c.endswith("_shape_mismatch"):
            st[c] = int(stats_df[c].fillna(False).astype(bool).sum())
        if c.endswith("_nonbinary_values"):
            st[c] = int(stats_df[c].notna().sum())
    for m in mask_names:  # positives at full res that vanish after nearest-neighbour downsampling
        if f"{m}_px" in stats_df and f"{m}_cache_px" in stats_df:
            v = ((stats_df[f"{m}_px"] > 0) & (stats_df[f"{m}_cache_px"] == 0)).sum()
            st[f"{m}_vanished_in_cache"] = int(v)
    with open(root / "build_stats.json", "w") as f:
        json.dump(st, f, indent=1, default=str)
    stats_df.to_csv(root / "per_image_stats.csv", index=False)
    print(f"[{name}] manifest {df.shape} -> {root / 'manifest.csv'}; stats {json.dumps(st, default=str)[:800]}",
          flush=True)
    return df


# ============================================================================ ISIC 2018 Task 1-2
ISIC_D3_MAP = {"Nevus": "nevus", "Melanoma in situ": "melanoma", "Melanoma Invasive": "melanoma",
               "Melanoma, NOS": "melanoma", "Seborrheic keratosis": "seborrheic_keratosis"}


def isic_coarse_label(d2, d3) -> str:
    """Map ISIC Archive ``diagnosis_3`` to nevus / melanoma / seborrheic_keratosis; everything else (incl. NaN)
    -> 'other' (label -1, in_task=0).  ``d2`` is accepted for API symmetry but not used."""
    d3 = "" if pd.isna(d3) else str(d3)
    return ISIC_D3_MAP.get(d3, "other")


def _attach_area(df: pd.DataFrame, st: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    a = st[["image_id"]].copy()
    for c in concepts:
        a[c] = (st[f"{c}_px"] > 0).astype(float)
        a[f"{c}_area"] = st[f"{c}_frac_lesion"]
        a[f"{c}_area_img"] = st[f"{c}_frac_img"]
        a[f"{c}_frac_outside_lesion"] = st[f"{c}_frac_outside_lesion"]
    a["lesion_area_img"] = st["lesion_frac_img"]
    return df.merge(a, on="image_id", how="left")


def build_isic2018_task12(overwrite: bool = False) -> pd.DataFrame:
    name = "isic2018_task12"
    _prep_dirs(name)
    r = RAW / "isic2018_task12"
    split_dirs = {"train": ("Training", "Task2_Training_GroundTruth_v3", "training", "train"),
                  "val": ("Validation", "Task2_Validation_GroundTruth", "validation", "val"),
                  "test": ("Test", "Task2_Test_GroundTruth", "test", "test")}
    rows, tasks, stats = [], [], {}
    metas = []
    # ISIC Archive metadata snapshot (data/metadata/isic2018_task12/, see scripts/fetch_isic_metadata.py)
    for split, (S, t2, apis, _) in split_dirs.items():
        ids = sorted(p.stem for p in (r / f"ISIC2018_Task1-2_{S}_Input").glob("ISIC_*.jpg"))
        api = pd.read_csv(METADATA / name / f"isic_archive_api_metadata_task1-2_{apis}.csv")
        stats[f"{split}_n_images"] = len(ids)
        stats[f"{split}_api_missing"] = len(set(ids) - set(api["isic_id"]))
        metas.append(api.assign(split=split))
        for i in ids:
            masks = {"lesion": str(r / f"ISIC2018_Task1_{S}_GroundTruth" / f"{i}_segmentation.png")}
            for a in ISIC18_ATTRS:
                masks[a] = str(r / f"ISIC2018_{t2}" / f"{i}_attribute_{a}.png")
            miss = [k for k, v in masks.items() if not os.path.exists(v)]
            if miss:
                stats.setdefault("missing_masks", []).append((i, miss))
                masks = {k: v for k, v in masks.items() if k not in miss}
            rows.append(dict(image_id=i, split=split, orig_path=str(r / f"ISIC2018_Task1-2_{S}_Input" / f"{i}.jpg")))
            tasks.append(_task(name, i, rows[-1]["orig_path"], masks=masks, lesion="lesion", overwrite=overwrite))
    df = pd.DataFrame(rows)
    api = pd.concat(metas, ignore_index=True).drop(columns=["split"]).rename(columns={"isic_id": "image_id"})
    df = df.merge(api, on="image_id", how="left")
    df["label_name"] = [isic_coarse_label(a, b) for a, b in zip(df.diagnosis_2, df.diagnosis_3)]
    df["label"] = [ISIC_DERM3.index(n) if n in ISIC_DERM3 else -1 for n in df.label_name]
    df["in_task"] = (df.label >= 0).astype(int)
    df["is_melanoma"] = (df.label_name == "melanoma").astype(int)
    df["benign_malignant"] = df["diagnosis_1"]
    stats["other_label_by_split"] = df[df.label < 0].groupby("split").size().to_dict()

    st = run_pool(tasks, name)
    df = _attach_area(df, st, ISIC18_ATTRS)
    stats["attr_pixels_outside_lesion_mean_frac"] = {a: float(df[f"{a}_frac_outside_lesion"].mean())
                                                     for a in ISIC18_ATTRS}
    stats["attr_area_gt1"] = {a: int((df[f"{a}_area"] > 1).sum()) for a in ISIC18_ATTRS}
    meta_cols = ["in_task", "is_melanoma", "benign_malignant", "diagnosis_1", "diagnosis_2", "diagnosis_3",
                 "diagnosis_4", "diagnosis_5", "diagnosis_confirm_type", "melanocytic", "age_approx", "sex",
                 "anatom_site_general", "anatom_site_1", "dermoscopic_type", "license", "attribution"]
    df = df.drop(columns=["image_type", "full_url"] + [f"{a}_frac_outside_lesion" for a in ISIC18_ATTRS])
    return _finalize(name, df, ISIC18_ATTRS, ["lesion"] + ISIC18_ATTRS, st, extra_first=meta_cols, stats=stats)


# ============================================================================ PH2
def parse_ph2_table(path: Path) -> pd.DataFrame:
    rows = []
    for line in open(path):
        if "|| IMD" not in line:
            continue
        p = [x.strip() for x in line.strip().strip("|").split("||")]
        crit = [x.strip() for x in p[3].split("|")]
        rows.append(dict(image_id=p[0], histological_diagnosis=p[1] or np.nan, clinical_diagnosis=int(p[2]),
                         asymmetry=int(crit[0]), pigment_network_raw=crit[1], dots_globules_raw=crit[2],
                         streaks_raw=crit[3], regression_areas_raw=crit[4], blue_whitish_veil_raw=crit[5],
                         colours_raw=" ".join(p[4].split())))
    df = pd.DataFrame(rows)
    df["asymmetry_1axis"] = (df.asymmetry == 1).astype(float)
    df["asymmetry_2axes"] = (df.asymmetry == 2).astype(float)
    assert set(df.pigment_network_raw) <= {"T", "AT"}
    df["pigment_network_atypical"] = (df.pigment_network_raw == "AT").astype(float)
    df["dots_globules_typical"] = (df.dots_globules_raw == "T").astype(float)
    df["dots_globules_atypical"] = (df.dots_globules_raw == "AT").astype(float)
    for c in ["streaks", "regression_areas", "blue_whitish_veil"]:
        assert set(df[f"{c}_raw"]) <= {"A", "P"}, c
        df[c] = (df[f"{c}_raw"] == "P").astype(float)
    cols = [set(int(x) for x in s.split()) for s in df.colours_raw]
    for k, v in PH2_COLOURS.items():
        df[f"colour_{v}"] = [float(k in s) for s in cols]
    df["n_colours"] = [len(s) for s in cols]
    df["label"] = df.clinical_diagnosis
    df["label_name"] = [PH2_CLASSES[i] for i in df.label]
    return df


def build_ph2(overwrite: bool = False) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold
    name = "ph2"
    _prep_dirs(name)
    root = RAW / "ph2"
    tabs = sorted(root.rglob("PH2_dataset.txt"))
    if not tabs:
        raise FileNotFoundError(f"PH2_dataset.txt not found under {root} (see docs/data.md)")
    df = parse_ph2_table(tabs[0])
    # official release: "PH2 Dataset images/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp" (+ IMDxxx_lesion/*.bmp);
    # a flat images/IMDxxx.jpg layout (public mirrors, no masks) is accepted as well
    imgs = {p.stem: p for p in root.rglob("IMD*") if p.suffix.lower() in (".jpg", ".bmp", ".png")}
    img_files = {k: v for k, v in imgs.items() if not k.endswith(("_lesion", "_roi")) and "_R" not in k}
    les_files = {k[:-7]: v for k, v in imgs.items() if k.endswith("_lesion")}
    stats = {"n_table_rows": len(df), "n_images": len(img_files), "n_lesion_masks": len(les_files),
             "table_ids_without_image": sorted(set(df.image_id) - set(img_files)),
             "images_without_table_row": sorted(set(img_files) - set(df.image_id))}
    df = df[df.image_id.isin(set(img_files))].reset_index(drop=True)
    df["orig_path"] = [str(img_files[i]) for i in df.image_id]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, te) in enumerate(skf.split(df, df.label)):
        df.loc[te, "fold"] = k
    df["split"] = np.where(df.fold == 0, "test", "train")
    tasks = [_task(name, i, p, masks={"lesion": str(les_files[i])} if i in les_files else {},
                   overwrite=overwrite) for i, p in zip(df.image_id, df.orig_path)]
    st = run_pool(tasks, name)
    meta_cols = ["fold", "clinical_diagnosis", "histological_diagnosis", "asymmetry", "pigment_network_raw",
                 "dots_globules_raw", "streaks_raw", "regression_areas_raw", "blue_whitish_veil_raw", "colours_raw",
                 "n_colours"]
    return _finalize(name, df, PH2_CONCEPTS, ["lesion"] if les_files else [], st, extra_first=meta_cols,
                     stats=stats)

BUILDERS = {
    "isic2018_task12": build_isic2018_task12,
    "ph2": build_ph2,
}


def main(argv: list[str]) -> None:
    names = list(BUILDERS) if (not argv or argv == ["all"]) else argv
    overwrite = "--overwrite" in names
    names = [n for n in names if not n.startswith("--")]
    for n in names:
        t0 = time.time()
        BUILDERS[n](overwrite=overwrite)
        print(f"== {n} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
