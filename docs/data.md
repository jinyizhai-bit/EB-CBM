# Data notes

This file describes the raw-data layout, the manifest-building rules, the split protocol and the known data issues of
the two datasets used by EB-CBM. All data live under `data/` (override with `CBM_DATA_ROOT`):

```
data/
├── raw/                 # downloaded raw data (read-only; archives are unpacked in place)
│   ├── isic2018_task12/
│   └── ph2/
├── metadata/isic2018_task12/   # ISIC Archive metadata snapshot (shipped with the repository)
├── splits/                     # fixed validation hold-out ids (shipped with the repository)
└── processed/<dataset>/        # written by cbm.data.manifests
    ├── manifest.csv            # one row per image
    ├── images_512/<image_id>.jpg       # RGB, longer side 512 px (LANCZOS, JPEG q95), never upscaled
    ├── masks_512/<image_id>_<mask>.png # 0/255 binary masks, nearest-neighbour resized to exactly the cached image size
    ├── build_stats.json        # data checks from the build
    └── per_image_stats.csv     # per-image original size, EXIF orientation, mask pixel counts at full resolution
```

Manifest columns, in order: `image_id, split, image_path` (512 cache, absolute), `orig_path` (raw file), `label`
(int, index into `DATASETS[name]['class_names']`), `label_name`, one column per concept (0/1; NaN = unknown),
`mask_<name>` (cache path; NaN if the image has no such mask), `<concept>_area` / `<concept>_area_img` /
`lesion_area_img` (ISIC 2018 only), dataset metadata, `orig_width, orig_height, width, height`.

```python
from cbm.data.datasets import MedConceptDataset, DATASETS
ds = MedConceptDataset("isic2018_task12", "train", return_masks=True)   # image, label, concepts, concept_known, masks, mask_valid
```

## 1. ISIC 2018 Task 1–2 (`isic2018_task12`)

### Raw data

`scripts/download_isic2018.sh` downloads and unpacks nine archives from
`https://isic-challenge-data.s3.amazonaws.com/2018/`:

```
data/raw/isic2018_task12/
├── ISIC2018_Task1-2_{Training,Validation,Test}_Input/        ISIC_xxxxxxx.jpg
├── ISIC2018_Task1_{Training,Validation,Test}_GroundTruth/    ISIC_xxxxxxx_segmentation.png  (lesion masks)
├── ISIC2018_Task2_Training_GroundTruth_v3/                   ISIC_xxxxxxx_attribute_<attr>.png  (five attribute masks)
├── ISIC2018_Task2_Validation_GroundTruth/
└── ISIC2018_Task2_Test_GroundTruth/
```

### Labels and concepts

- **Official split**: train 2594, val 100, test 1000. Every image has a lesion mask and all five attribute masks
  (`pigment_network, negative_network, streaks, milia_like_cyst, globules`).
- **Concepts** = image-level presence (any foreground pixel in the full-resolution attribute mask).
- **Diagnosis label**: from the `diagnosis_3` field of the ISIC Archive API metadata (snapshot:
  `data/metadata/isic2018_task12/isic_archive_api_metadata_task1-2_{training,validation,test}.csv`, produced by
  `scripts/fetch_isic_metadata.py`). Mapping: `Nevus → nevus (0)`; `Melanoma in situ`, `Melanoma Invasive`,
  `Melanoma, NOS → melanoma (1)`; `Seborrheic keratosis → seborrheic_keratosis (2)`; everything else (including NaN)
  → `other` (label −1, `in_task = 0`). `other` rows stay in the manifest and are dropped by `MedConceptDataset` by
  default; there are 44 of them (train 4, val 6, test 34). The raw `diagnosis_1..5`, `diagnosis_confirm_type`,
  `melanocytic`, age, sex, site, dermoscopic type, licence and attribution columns are kept.
- **Areas** (from the full-resolution masks): `<c>_area` = attribute pixels / lesion-mask pixels, `<c>_area_img` =
  attribute pixels / image pixels, `lesion_area_img` = lesion pixels / image pixels. Attribute masks are not
  constrained to the lesion mask: on average 2.7 % of the pigment-network pixels lie outside it, and
  `pigment_network_area` exceeds 1 for 48 images (max 5.97); the training code clips area targets to [0, 1].
- No positive attribute region disappears after nearest-neighbour downsampling to 512 px (checked for every mask).

Images per split and class:

| | train | val | test | all |
|:--|--:|--:|--:|--:|
| nevus (0) | 1863 | 69 | 690 | 2622 |
| melanoma (1) | 519 | 20 | 200 | 739 |
| seborrheic_keratosis (2) | 208 | 5 | 76 | 289 |
| other (−1) | 4 | 6 | 34 | 44 |
| **total** | 2594 | 100 | 1000 | 3694 |

Concept positive rates:

| concept | train | val | test | positives | median area / lesion (positives) |
|:--|--:|--:|--:|--:|--:|
| pigment_network | 0.587 | 0.73 | 0.648 | 2244 | 0.258 |
| negative_network | 0.073 | 0.09 | 0.076 | 275 | 0.092 |
| streaks | 0.039 | 0.06 | 0.077 | 183 | 0.100 |
| milia_like_cyst | 0.263 | 0.06 | 0.085 | 773 | 0.039 |
| globules | 0.232 | 0.19 | 0.206 | 828 | 0.101 |

Note the distribution shift of milia-like cysts: 26 % positive in training, only 6–9 % in validation and test.

### Split used in the paper

`val` = official validation set (94 in-task images) + a class-stratified 12 % hold-out of the official training set
(311 images; `sklearn.model_selection.train_test_split(test_size=0.12, random_state=0, stratify=label)` over the
in-task training rows sorted by `image_id`). The hold-out ids are fixed in
`data/splits/isic2018_task12_val_holdout_seed0.csv`; `cbm.datamod.val_holdout_ids` copies the file to
`data/processed/isic2018_task12/val_holdout_seed0.csv` on first use, and that file takes precedence once it exists.
Final sizes: **train 2279 / val 405 / test 966**. Config keys: `data.val_from_train_frac: 0.12`,
`data.holdout_seed: 0`.

### Near-duplicate audit (`python -m cbm.data.overlap`)

A 256-bit dHash (16×16) is computed on the 512-px cache images; Hamming distance ≤ 4 counts as a near-duplicate. The
audit writes `data/processed/overlap_report.md`, `overlap_pairs.csv`, `overlap_hashes.csv` and a montage
`overlap_examples.jpg`. Findings on the paper data: within ISIC 2018 Task 1–2 there are 18 test–train, 3 train–val and
1 test–val near-duplicate pairs (different ids, near-identical photographs, same label); there are no near-duplicates
between PH2 and ISIC. The paper keeps the official split for comparability with prior work; global hashes miss crops
and strong re-colouring, so these counts are lower bounds.

## 2. PH2 (`ph2`)

- **Source**: the PH2 database (Mendonça et al., 2013), 200 dermoscopic images, to be requested from the ADDI project
  website. Place the unpacked content under `data/raw/ph2/`: the builder searches recursively for `PH2_dataset.txt`
  (the official annotation table) and for `IMDxxx.{bmp,jpg,png}` images in any sub-directory; both the official layout
  `PH2 Dataset images/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp` and a flat `images/IMDxxx.jpg` layout work.
  `IMDxxx_lesion` masks (if present) are cached as `mask_lesion`; the `_R*` files under `IMDxxx_roi` are ignored.
- **Label**: clinical diagnosis (0 common nevus, 1 atypical nevus, 2 melanoma: 80 / 80 / 40). The histological
  diagnosis is available for 41 images only.
- **Concepts** (14 binary columns, raw codes kept in `*_raw`): `asymmetry_1axis`, `asymmetry_2axes`;
  `pigment_network_atypical` (AT vs T); `dots_globules_typical`, `dots_globules_atypical`; `streaks`,
  `regression_areas`, `blue_whitish_veil` (P vs A); six colours `colour_{white, red, light_brown, dark_brown,
  blue_gray, black}`; plus `n_colours`.
- **Splits**: `fold` = `StratifiedKFold(5, shuffle=True, random_state=0)`; by default fold 0 = test (40), the rest =
  train (160). The EB-CBM evaluator uses **all 200** PH2 images as an external test set of the frozen model
  (melanoma-vs-nevus AUROC/BACC from P(melanoma); concept-transfer AUROC for streaks and dots/globules).
- The paper used a public JPEG mirror of the 200 images (no lesion masks), hence no PH2 grounding metrics in the paper.

## 3. Counterfactual edits (`data/processed/isic2018_task12/cf_512/`)

Produced by `python -m cbm.train.counterfactual --splits train val test` (requires `models/big-lama.pt`). For every
in-task image and every present structure c:

- `minus_<c>`: LaMa inpainting of the expert mask of c dilated by 3 px (structure removed);
- `ctrl_<c>`: inpainting of a control region with the same pixel area, inside the lesion and overlapping none of the
  (3-px dilated) structure masks. The control region is, in order of preference: (1) the structure mask itself
  translated / flipped to a free location (`shift`, shape-matched); (2) a compact blob grown inside the free lesion area
  from a random seed (`blob`); (3) the largest free connected region when the free area is smaller than the target
  (`largest_free`; the area is then smaller than the target); (4) free skin outside the lesion when the lesion has no
  free pixels (`outside_lesion`).

Outputs: `cf_512/<image_id>__<edit>.jpg` (pixels outside the region copied from the cache image),
`cf_512/regions/<image_id>__<edit>.png` (inpainted region, 0/255) and `cf_512/index.csv` (columns: `image_id, split,
label, edit, kind (minus|ctrl), concept, path, region_path, area_px, target_px, method, overlap_px, post_<concept>`).
After a `minus` edit concept c is 0 and another concept c' stays present iff at least 10 % of its pixels survive
outside the inpainted region; `ctrl` edits keep every label. Paper data: 3650 images, 8546 edits (control methods:
919 shift, 2167 blob, 1179 largest_free, 8 outside_lesion). The evaluator can restrict CLS to area-matched pairs
using `area_px` and `target_px`.
