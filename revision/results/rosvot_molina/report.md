# Official ROSVOT on the 38 Molina recordings

The official M4Singer-trained ROSVOT, RWBD, and RMVPE checkpoints were run
directly on the same 38 waveforms used by the revised Molina evaluation.
The official inference default note-boundary threshold (0.85)
was retained. No Molina annotation, test-derived time shift, or threshold tuning
was used for ROSVOT inference.

| Aggregation | COnPOff | COnP | COn | Reference notes | Estimated notes |
|---|---:|---:|---:|---:|---:|
| Micro | 0.1710 | 0.2295 | 0.2651 | 2152 | 2397 |
| Macro | 0.1875 | 0.2492 | 0.2886 | -- | -- |

These are the standard Molina metrics used for the proposed method, not the
special onset-only, octave-invariant HumTrans metric.
