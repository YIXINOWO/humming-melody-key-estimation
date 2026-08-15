# CREPE-guided humming melody estimation

Code and derived results for the manuscript:

**CREPE-guided note-level melody estimation and tonal-representation analysis for monophonic humming**

The system uses full CREPE pitch tracks, 44 frame-level acoustic/context features,
separate RandomForest onset and offset classifiers, and a recording-grouped
5 x 4 nested cross-validation protocol. Tonal analysis is an independent
descriptive branch and is not fed back into note transcription.

## Main reported results

| Evaluation | COnPOff | COnP | COn |
|---|---:|---:|---:|
| Proposed method, Molina nested CV | 0.696 | 0.842 | 0.889 |
| ROSVOT official checkpoint, Molina zero-shot | 0.171 | 0.230 | 0.265 |
| Proposed method, HumTrans native timing | 0.008 | 0.021 | 0.161 |
| Proposed method, HumTrans validation-selected -0.18 s shift | 0.146 | 0.202 | 0.399 |

The 38-recording Molina result contains 2,152 audited reference notes. Its macro
COnPOff is 0.685, with a 95% recording-bootstrap interval of 0.643-0.727. The
mean cosine similarity between note-derived and audio-derived pitch-class
profiles is 0.978. This is representation agreement, not independent key
accuracy, because manual key labels are unavailable.

## Repository structure

```text
.
|-- environment.yml
|-- environment-rosvot.yml
|-- requirements-core.txt
|-- requirements-ml.txt
|-- final_results_for_paper/
|   `-- frame_onset_offset_oof.csv.gz
`-- revision/
    |-- config/main_experiment.yaml
    |-- scripts/
    |-- src/jasm_revision/
    |-- tests/
    `-- results/
```

`revision/results/` contains the manuscript-facing summary tables, nested-CV
fold records, predictions, shareable figure source data, and rendered figures.
The final Fig. 1 and Fig. 2 PNGs are the author-approved redraws stored under
`revision/assets/author_final/`; the plotting script installs those exact files
after generating the code-native schematic versions. Raw audio,
third-party annotations, external repositories, checkpoints, and large model
files are deliberately excluded.

## Environment

Create the main environment:

```bash
conda env create -f environment.yml
conda activate jasm-revision
python -m pytest revision/tests -q
```

The test suite is self-contained except for one corpus-integrity test, which is
automatically skipped when the MTG-QBH/Molina annotations are absent.

GPU-based full-CREPE extraction additionally requires the official PyTorch CUDA
wheels and `requirements-ml.txt`:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-ml.txt
```

ROSVOT inference uses a separate compatibility environment:

```bash
conda env create -f environment-rosvot.yml
```

The environment names used in this project are `jasm-revision` and
`rosvot-inference`.

## Data layout

To rerun the Molina experiments, provide an authorized local copy using this
layout:

```text
audio/
  child1.wav
  ...
gt_files_temp/
  child1.GroundTruth.txt
  ...
```

The full corpus contains 132 waveforms; 38 have note-level annotation files.
Do not commit either directory. The compressed frame table included at
`final_results_for_paper/frame_onset_offset_oof.csv.gz` is the exact input used
by the revised nested-CV experiments.

HumTrans and ROSVOT resources must be obtained from their upstream providers.
Expected paths and verified checkpoint hashes are documented in
`THIRD_PARTY_RESOURCES.md` and `revision/external/README.md`.

## Core reproduction commands

With the Molina audio and annotations in place:

```bash
python revision/scripts/audit_ground_truth.py
python revision/scripts/run_nested_cv.py --overwrite
python revision/scripts/analyze_molina_results.py
python revision/scripts/analyze_error_localization.py
python revision/scripts/analyze_tonal_agreement.py
python revision/scripts/run_corrected_ablation.py
```

For the 132-recording full-CREPE tonal description:

```bash
python revision/scripts/analyze_all_audio_tonal_distribution.py \
  --device cuda --batch-size 128
```

For the complete manuscript figure bundle after the required local data have
been placed:

```bash
python revision/scripts/plot_all_manuscript_figures.py
```

This command automatically reapplies the author-approved Fig. 1 and Fig. 2
raster artwork. Their PDF/SVG/TIFF companions embed the same raster image;
quantitative figures retain editable vector text.

See `REPRODUCIBILITY.md` for the external-corpus and ROSVOT workflow, command
order, expected outputs, and limitations.

## Availability and licensing

The Python code is released under the MIT License. The MIT License applies to
the authors' code, not to third-party audio, annotations, repositories, model
weights, or baseline outputs. Included derived artifacts are provided for
verification of the manuscript; reuse remains subject to the terms of the
underlying resources. Exact waveform samples and copied third-party reference
annotations are not included in the figure source-data bundle.

No funding was received for this study.

## Citation

Please use `CITATION.cff` and cite the associated manuscript when available.
