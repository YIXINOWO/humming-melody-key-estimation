# Molina segmentation-error localization

reference-sequence start/end and both sides of an inter-note gap >= 0.30 s.
event time within 0.15 s of a phrase boundary.

| Error | Events | Near boundary | Observed fraction | Time-coverage null | Enrichment | Circular-shift p |
|---|---:|---:|---:|---:|---:|---:|
| split | 397 | 16 | 0.040 | 0.146 | 0.28 | 1.0000 |
| merged | 464 | 2 | 0.004 | 0.144 | 0.03 | 1.0000 |
| spurious | 54 | 4 | 0.074 | 0.150 | 0.49 | 0.9708 |
| all | 915 | 22 | 0.024 | 0.145 | 0.17 | 1.0000 |

The boundary definition and window were varied over the full sensitivity grid in `boundary_sensitivity.csv`; the primary setting was fixed before inspecting the results.
