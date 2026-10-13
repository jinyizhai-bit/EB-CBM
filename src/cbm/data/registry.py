"""Static registry of the datasets prepared by ``cbm.data.manifests``.

Each entry gives the class names (index == integer ``label`` in the manifest), the concept column names (order ==
order of the ``concepts`` tensor returned by ``MedConceptDataset``), concept groups, the mask names (manifest column
``mask_<name>``) and the available splits.  ``datasets.py`` re-exports this as ``DATASETS``.
"""
from __future__ import annotations

import re

from cbm.paths import DATA_ROOT, METADATA, PROCESSED, RAW  # noqa: F401  (re-exported for the builders)


def slug(s: str) -> str:
    """'Purpura/Petechiae' -> 'purpura_petechiae', 'light blue' -> 'light_blue'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip().lower())
    return s.strip("_")


# ----------------------------------------------------------------------------- ISIC 2018 Task 1-2
ISIC18_ATTRS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
ISIC_DERM3 = ["nevus", "melanoma", "seborrheic_keratosis"]

# ----------------------------------------------------------------------------- PH2
PH2_CLASSES = ["common_nevus", "atypical_nevus", "melanoma"]
PH2_COLOURS = {1: "white", 2: "red", 3: "light_brown", 4: "dark_brown", 5: "blue_gray", 6: "black"}
PH2_GROUPS = {
    "asymmetry": ["asymmetry_1axis", "asymmetry_2axes"],
    "pigment_network": ["pigment_network_atypical"],
    "dots_globules": ["dots_globules_typical", "dots_globules_atypical"],
    "streaks": ["streaks"],
    "regression_areas": ["regression_areas"],
    "blue_whitish_veil": ["blue_whitish_veil"],
    "colours": [f"colour_{v}" for v in PH2_COLOURS.values()],
}
PH2_CONCEPTS = [c for g in PH2_GROUPS.values() for c in g]


def _entry(**kw):
    kw.setdefault("concept_groups", {c: [c] for c in kw.get("concept_names", [])})
    kw.setdefault("mask_names", [])
    kw["has_masks"] = len(kw["mask_names"]) > 0
    kw["manifest"] = str(PROCESSED / kw["name"] / "manifest.csv")
    kw["n_classes"] = len(kw["class_names"])
    kw["n_concepts"] = len(kw["concept_names"])
    return kw


DATASETS: dict[str, dict] = {
    "isic2018_task12": _entry(
        name="isic2018_task12", modality="dermoscopy",
        class_names=ISIC_DERM3,
        concept_names=ISIC18_ATTRS,
        concept_groups={"network": ["pigment_network", "negative_network"], "streaks": ["streaks"],
                        "globular": ["globules", "milia_like_cyst"]},
        mask_names=["lesion"] + ISIC18_ATTRS,
        area_columns=[f"{c}_area" for c in ISIC18_ATTRS],
        splits=["train", "val", "test"],
        notes="Concepts = image-level presence recomputed from the Task 2 masks. Label from the ISIC Archive "
              "diagnosis_3; label=-1 ('other', in_task=0) for anything not nevus/melanoma/SK; dropped by default in "
              "MedConceptDataset.",
    ),
    "ph2": _entry(
        name="ph2", modality="dermoscopy",
        class_names=PH2_CLASSES,
        concept_names=PH2_CONCEPTS, concept_groups=PH2_GROUPS,
        mask_names=[],  # lesion masks (official release) are cached as mask_lesion but not used for grounding
        splits=["train", "test"],
        notes="External test set (all 200 images are used by the evaluator). 5-fold CV column 'fold' (stratified, "
              "seed 0); default split: fold 0 = test, folds 1-4 = train; MedConceptDataset(fold=k) uses fold k as test.",
    ),
}
