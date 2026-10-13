# Evaluation notes

`python -m cbm.eval.evaluate_run --run_id <run_id> [--gpu 0] [--fast] [--tests ...] [--merge]`

The evaluator computes every metric from the prediction dumps and the checkpoint, writes
`experiments/runs/<run_id>/eval.json` and `eval_summary.csv` (the flattened `summary` section) and produces figures in
`figures/<run_id>/`. It runs automatically after training (`eval.after_train: true`). Model-based tests cache the
backbone features; a full evaluation of the 966 test images usually takes a few minutes.

## 1. Prediction dumps (`preds_{val,test,ph2}.npz`)

| Key | Shape | Meaning |
|---|---|---|
| `image_id` | (N,) str | image ids |
| `y_true`, `y_prob` | (N,), (N, C) | labels, class probabilities |
| `c_true`, `c_prob` | (N, K) | concept labels (NaN = unknown), concept probabilities |
| `area_true`, `area_pred` | (N, K) | structure area / lesion area (ground truth, predicted) |
| `mask_pred` | (N, K+1, 64, 64) uint8 | predicted masks (0..255); planes 0..K−1 = concepts in registry order, K = lesion |
| `logits` | (N, C) | diagnosis logits |
| `c_repr` | (N, K, d) | per-concept representation that reaches the diagnosis head (used by OIS) |
| `head_inputs`, `head_input_names` | (N, D), (D,) | exact input matrix of the diagnosis head and its column names (`group:detail`; groups concept / area / appearance / report / lesion / ...) |
| `report`, `report_names` | (N, 35), (35,) | report descriptors (including the five presence entries) |
| `class_names`, `concept_names` | | class and concept names |

`preds_ph2.npz` additionally holds the three-class probabilities on all 200 PH2 images and `ph2_concepts`.

## 2. Sections of `eval.json`

### `val` / `test` (dump-based)

- `task`: ACC, BACC, macro-F1, AUROC (macro, melanoma one-vs-rest, per class), ECE, NLL.
- `concepts`: per-concept AUROC / F1 and their means.
- `grounding`: Dice / IoU of predicted vs expert masks (threshold 0.5, 64×64; per image and global), pointing
  accuracy (argmax inside the expert mask dilated by 1 px), false-positive rate on images without the structure,
  area MAE.
- `leakage`: 5-fold cross-validated logistic-regression probes (label from soft concept scores vs hard scores vs
  `c_repr`) and OIS (Espinosa Zarlenga et al., 2023).
- `deferral`: accuracy–coverage curves and AUC for max-probability / entropy deferral.

### `ph2`

Melanoma-vs-nevus AUROC / BACC of the frozen model on all 200 PH2 images (from P(melanoma)) and concept-transfer AUROC
for streaks and dots/globules. `--tests ph2 [--merge]` regenerates `preds_ph2.npz` through the model API.

### `calibration`

A single temperature T fitted on the validation dump (minimising the NLL); test ECE / NLL before and after
temperature scaling. Dump-only, no GPU needed (`--tests calibration --merge`). Summary keys: `test_ece_ts`,
`test_nll_ts`, `temperature`.

### `model_tests` (the model is reloaded and called through the model API)

| Test | Content |
|---|---|
| `havasi` | accuracy with ground-truth / hard / soft concepts fed to the diagnosis head (intervention gain, soft–hard gap) |
| `tti` | test-time intervention curves: random order (5 seeds) / uncertainty order / verification order; perfect expert and noisy expert correct with p = 0.8 (3 seeds); area under each curve |
| `spatial` | k predicted concept masks replaced by expert masks (random ×3 / uncertainty order) and all planes at once; change of concept AUROC and BACC |
| `deferral` | accuracy–coverage AUC for max-probability, entropy and influence-weighted verification scores |
| `locality` | Raman-style locality tests: blur / noise outside all structures vs inside each structure vs outside the lesion; change of concept probabilities, TV change of the diagnosis, drift of the lesion descriptor. `fixed_masks_outside_pred_struct` / `fixed_masks_outside_pred_lesion` freeze the masks at the model's own prediction and perturb only outside the region (exactly 0 for crop evidence); `fixed_expert_masks_outside_struct` freezes the masks at the expert masks |
| `counterfactual` | CLS_c, XE_c, diagnosis shift under removal of the most influential concept, clinical direction checks (e.g. removing milia-like cysts should lower P(SK)), mask-area drop (needs the test-split edits in `cf_512`) |
| `artefacts` | rulers / ink marks / vignettes / hairs pasted outside the lesion: flip rate, ΔBACC, concept drift, share of modified images |
| `parts` | statistics of the discovered regions (`discover.n_parts > 0`): area fractions, presence rate, share of mass inside the expert lesion / structures, Dice with each structure, descriptor statistics |
| `report_weights` | weights of the report head and sign checks against the two-step / 7-point rules |

### `sanity`

Only a linear head is retrained (class-balanced logistic regression on standardised inputs, C chosen in
{0.01, 0.1, 1, 10} on validation BACC; the training inputs are computed with the eval transform and cached in
`head_inputs_train.npz`) and its test BACC reported for: the model's own head inputs (`own_head_inputs`), its own
concept probabilities (`own_concepts`), Gaussian random inputs of the same dimensionality (5 seeds), label-shuffled
inputs (5 seeds), oracle ground-truth concepts, oracle ground-truth concepts + √(ground-truth area) + ground-truth
lesion area; plus `model_head_test_bacc` and `chance_bacc`. A faithful bottleneck must collapse to chance on random /
shuffled inputs.

### `ladder` (completeness ladder)

Input groups are accumulated in the order concept → +area → +report → +appearance → +lesion (then +verify / cem_emb /
global / part when present). Reported per step: the test BACC of a logistic regression refitted on that subset
(`refit`; `single_group_refit` gives each group alone, `all_but_concept` the label information in the non-concept
channels), and the BACC of the model's own head with the other inputs zeroed (`head_zeroed`) or set to their training
mean (`head_mean_imputed`).

### `summary`

Flat headline numbers (also in `eval_summary.csv`), e.g. `test_bacc, test_auroc_macro, test_concept_auroc, test_dice,
test_iou, test_pointing, ois, CLS, XE, artefact_flip_rate, artefact_delta_bacc, deferral_auc_maxprob, test_ece_ts,
ph2_auroc, ph2_bacc, locality_gap_blur, task_tv_outside_lesion_blur, fixedmask_task_tv_outside_lesion_blur,
fixedmask_concept_delta_outside_struct_blur, havasi_intervention_gain, tti_area_random, tti_area_random_noisy80,
spatial_area_random, sanity_*, ladder_refit_*`. `src/cbm/scripts/aggregate.py` collects the `summary` of several runs
into one table.

## 3. Figures per run

`figures/<run_id>/`: `masks_overlay.png` (expert and predicted masks; discovered-region contours when present),
`counterfactual.png`, `tti.png`, `spatial.png`, `deferral.png`, `artefact_examples.png`, and `parts_examples.png` for
the prototype / discovered-region variants.

## 4. Recomputing a subset of tests

```bash
python -m cbm.eval.evaluate_run --run_id ebcbm_s0 --gpu 0 --tests sanity ladder locality --merge
```

`--merge` replaces only the requested sections of `eval.json` and keeps everything else; `--fast` computes the
dump-based metrics only (no model reload). Available test names: `calibration havasi tti spatial deferral locality
counterfactual artefacts parts sanity ladder`, plus `dump` (recompute the val/test/ph2 dump metrics) and `ph2`.
