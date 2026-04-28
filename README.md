# Humming Melody and Key Estimation

This repository contains the code, derived results, and manuscript files for:

**CREPE-guided note-level melody estimation and independent key detection for monophonic humming**

The project studies note-level melody estimation and key detection on monophonic humming recordings. The transcription pipeline combines a pretrained CREPE F0 front end, frame-level acoustic features, and separate RandomForest onset/offset boundary classifiers. Evaluation is performed with recording-grouped five-fold out-of-fold prediction on the annotated subset of the MTG-QBH/Molina corpus.

## Main results

On the 38 annotated MTG-QBH/Molina recordings:

| Metric | Result |
|---|---:|
| COnPOff F-measure | 0.689 |
| COnP F-measure | 0.831 |
| COn F-measure | 0.864 |
| Exact key accuracy | 60.5% |
| Tonic accuracy | 65.8% |
| Mode accuracy | 78.9% |

The strongest directly comparable result reported by Li et al. (2021) reaches 0.610 COnPOff on the same benchmark setting; the proposed system improves this by 7.9 percentage points.

## Repository structure

```text
.
├── finalize_for_paper.py
├── supplementary_experiments.py
├── requirements.txt
├── final_results_for_paper/
│   ├── paper_main_results.csv
│   ├── method_comparison_clean.csv
│   ├── ablation_study.csv
│   ├── bootstrap_significance_clean.csv
│   ├── key_accuracy_metrics.csv
│   ├── all_predictions_onoff.csv
│   ├── frame_onset_offset_oof.csv.gz
│   └── figures and supporting CSV files
└── manuscript/
    ├── manuscript.pdf
    ├── manuscript.tex
    ├── references.bib
    └── manuscript figures
```

## Data availability

The original MTG-QBH/Molina audio recordings are **not redistributed** in this repository. Users should obtain the corpus from its original source and place the files in the expected local layout if they want to rerun the full pipeline.

This repository provides derived experimental artifacts used in the manuscript, including:

- frame-level out-of-fold features and probabilities, compressed as `final_results_for_paper/frame_onset_offset_oof.csv.gz`;
- predicted note files;
- summary tables;
- paper figures.

To decompress the large OOF table:

```bash
gzip -dk final_results_for_paper/frame_onset_offset_oof.csv.gz
```

## Environment

Install the basic analysis dependencies with:

```bash
pip install -r requirements.txt
```

The full F0 extraction pipeline also requires a working CREPE / torchcrepe environment. The supplementary experiments reuse precomputed artifacts and require only the packages listed in `requirements.txt`.

## Reproducing supplementary analyses

The supplementary script expects the derived result files under `final_results_for_paper/`. Some routines also expect local ground-truth annotation files from the corpus in `gt_files_temp/`, which are not redistributed here.

```bash
python supplementary_experiments.py --quick
```

For the full supplementary run:

```bash
python supplementary_experiments.py
```

## Citation

If you use this repository, please cite the associated manuscript:

```text
Liu, J., Wang, C., and Jiang, H. CREPE-guided note-level melody estimation and independent key detection for monophonic humming.
```

## License

Code in this repository is released under the MIT License. Dataset rights remain with the original dataset creators.
