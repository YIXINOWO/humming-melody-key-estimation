# Release manifest

Release date: 2026-08-14

## Included

- Author-written Python package under `revision/src/jasm_revision/`.
- Experiment and figure scripts under `revision/scripts/`.
- Unit tests and the formal YAML configuration.
- Author-approved final Fig. 1 and Fig. 2 raster assets.
- Main and ROSVOT Conda environment definitions.
- Pinned core and GPU dependency lists.
- Compressed historical frame feature/probability table.
- Formal nested-CV fold selections, probabilities, note predictions, and metrics.
- Secondary, error-localization, tonal, ablation, HumTrans, and ROSVOT results.
- All manuscript figures except redundant TIFF copies, plus shareable figure source data.
- MIT code licence, citation metadata, availability documentation, and checksums.

## Excluded

- MTG-QBH/Molina and HumTrans audio or annotations.
- Third-party repository snapshots and dataset archives.
- ROSVOT `checkpoints.zip` and extracted model weights.
- CREPE per-recording caches.
- Regenerable 120 MB external-deployment `.joblib` model files.
- Conda environments, caches, bytecode, and LaTeX build files.
- Decision letters, reviewer correspondence, internal trackers, submission files,
  and private attachments.
- Files containing local absolute paths.
- Exact waveform samples and copied third-party reference annotations used by
  qualitative figure panels.

The generated `SHA256SUMS.txt` is the authoritative file-level inventory for
this package.
