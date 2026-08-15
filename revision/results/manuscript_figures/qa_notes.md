# Manuscript figure bundle QA

Backend: Python/matplotlib. Quantitative figures are exported as editable
SVG/PDF plus 600-dpi PNG/TIFF. Figures 1 and 2 use the author-approved final
raster redraws stored under `revision/assets/author_final/`; their matching
PDF/SVG/TIFF files embed the same raster artwork. Numeric panels read CSV/JSON
artifacts and the historical exploratory panel is explicitly marked as
non-nested provenance.

| Figure | Core conclusion | Archetype | Primary source data | Main reviewer-risk check |
|---|---|---|---|---|
| Fig. 1 | Note transcription and tonal representation are independent branches. | Schematic-led | fig01_pipeline.json | No tonal feedback arrow; tonal labels are descriptive. |
| Fig. 2 | Nested recording-grouped CV prevents outer-test selection leakage. | Schematic-led | fig02_nested_cv.json | Locked outer-test path is outside the inner-search container. |
| Fig. 3 | Proposed nested-CV result is shown alongside published Li et al., local heuristic and fixed ROSVOT reference values. | Quantitative grid | fig03_method_comparison.csv | Higher/lower metric directions and asymmetric ROSVOT training are explicit. |
| Fig. 4 | Temporal lags provide the largest cumulative ablation gain. | Quantitative trend | fig04_corrected_ablation.csv | Exploratory/frozen-decoder status is visible. |
| Fig. 5 | Recording-level macro estimates have reproducible bootstrap intervals. | Quantitative grid | fig05_bootstrap_samples.csv, fig05_bootstrap_ci.csv | n=38 and 10,000 resamples are explicit. |
| Fig. 6 | CREPE-derived features and boundary probabilities are traceable in a representative recording. | Image/quantitative composite | fig06_child1_frames.csv, fig06_child1_waveform.csv | Thresholds and boundary labels are shown. |
| Fig. 7 | Outer-test note intervals can be inspected against the audited reference. | Quantitative interval plot | fig07_child1_reference.csv, fig07_child1_prediction.csv | Reference and prediction use identical time/pitch axes. |
| Fig. 8 | Audio- and note-derived pitch-class profiles agree descriptively. | Quantitative grid | fig08_tonal_agreement.csv | Template labels are not called key accuracy. |
| Fig. 9 | Full-CREPE profiles yield 54 major and 78 minor descriptive labels over 132 recordings. | Quantitative grid | fig09_tonal_labels.csv, fig09_mode_summary.csv | Descriptive labels and lack of manual keys are explicit. |
| Fig. 10 | Hard tonal snapping degraded the historical pipeline. | Quantitative comparison | fig10_historical_key_snapping.csv | Historical non-nested status is prominent. |

Statistics and definitions are retained in the corresponding revision result
directories and manuscript captions.  No data under `final_results_for_paper/`
was overwritten.
