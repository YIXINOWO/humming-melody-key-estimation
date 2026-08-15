# Reproducibility guide

## Reproduction levels

### 1. Code validation without third-party data

```bash
conda env create -f environment.yml
conda activate jasm-revision
python -m pytest revision/tests -q
```

Expected outcome without local Molina annotations: 20 passed, 1 skipped.

### 2. Inspect the released evidence

The following directories can be inspected without downloading raw data:

- `revision/results/molina_nested_cv/`: outer folds, selected parameters,
  outer-test probabilities, predictions, and note-level metrics.
- `revision/results/molina_secondary_analysis/`: frame metrics, bootstrap
  intervals, decoder robustness, and error burden.
- `revision/results/molina_error_localization/`: boundary sensitivity and
  event-level localization.
- `revision/results/molina_tonal_agreement/`: direct pitch-class profile
  agreement and confidence sensitivity.
- `revision/results/humtrans_time_alignment/`: validation-selected timing
  analysis and test results.
- `revision/results/humtrans_aligned_baselines/`: same-test aligned baseline
  comparison.
- `revision/results/rosvot_molina/`: official-checkpoint summary, derived note
  predictions, and per-recording scores.
- `revision/results/manuscript_figures/source_data/`: source data mapped to all
  ten manuscript figures.

### 3. Rerun the Molina nested-CV experiment

Place the authorized Molina files under `audio/` and `gt_files_temp/`, then run:

```bash
python revision/scripts/audit_ground_truth.py
python revision/scripts/run_nested_cv.py --overwrite
python revision/scripts/analyze_molina_results.py
python revision/scripts/analyze_error_localization.py
python revision/scripts/analyze_tonal_agreement.py
python revision/scripts/run_corrected_ablation.py
```

The formal configuration is `revision/config/main_experiment.yaml`. All frame
rows from one recording remain in the same fold. Model and decoder selection
use inner out-of-fold predictions only; each outer-test recording is evaluated
once.

The compressed feature table is read directly by pandas. Its uncompressed
SHA-256 is:

```text
eda1c0664911157ba136fbe0e24e5d4712cba611788e75b0e6a04f8b2189b00f
```

### 4. Fit the frozen external model and run HumTrans

Obtain HumTrans audio, ground-truth MIDI, official split files, and distributed
baseline MIDI files from the upstream project. Use the layout in
`revision/external/README.md`.

```bash
python revision/scripts/fit_external_deployment_model.py
python revision/scripts/evaluate_external_zero_shot.py --split valid --overwrite
python revision/scripts/evaluate_external_zero_shot.py --split test --overwrite
python revision/scripts/analyze_humtrans_time_alignment.py
python revision/scripts/evaluate_humtrans_baselines.py
python revision/scripts/evaluate_humtrans_aligned_baselines.py
```

The deployment `.joblib` files are not included because they total about 120 MB
and can be regenerated from the included feature table and authorized Molina
annotations. The released HumTrans predictions and metrics remain available for
audit.

### 5. Run the official ROSVOT checkpoint

Clone the upstream ROSVOT repository to
`revision/external/repos/ROSVOT-main`, download its official
`checkpoints.zip`, and extract the checkpoint folders inside that repository.
Create the compatibility environment:

```bash
conda env create -f environment-rosvot.yml
python revision/scripts/evaluate_rosvot_molina.py --prepare-only
```

Run the upstream bulk-inference command with the generated
`revision/results/rosvot_molina/manifest.json`, writing output to
`revision/results/rosvot_molina/inference/`. Then evaluate it:

```bash
python revision/scripts/evaluate_rosvot_molina.py
```

The official note-boundary threshold is 0.85. No Molina annotation, timing
shift, or threshold search is used for ROSVOT inference.

### 6. Regenerate figures

After the Molina data are present and all upstream result steps have completed:

```bash
python revision/scripts/plot_all_manuscript_figures.py
```

Outputs are written to `revision/results/manuscript_figures/` as SVG, PDF, PNG,
TIFF, and source-data files. The author-approved Fig. 1/2 raster redraws under
`revision/assets/author_final/` are applied automatically at the end; their
PDF/SVG/TIFF companions embed the same raster image. If a manuscript source
directory is present, PNG files are also copied there; otherwise figure
generation remains standalone.

## Determinism and expected variation

The experiment seed is 2026. Fold membership and RandomForest seeds are stored
in the configuration and fold manifests. Small differences may occur across
platforms or library builds. GPU CREPE outputs can also differ at low numerical
precision, so the released feature table and source data are the reference
artifacts for exact manuscript verification.
