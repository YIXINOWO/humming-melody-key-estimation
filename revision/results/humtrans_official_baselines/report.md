# HumTrans official-baseline audit

The official HumTrans onset-only, octave-invariant metric is reproduced separately from the standard Molina-style note metrics.

| Model | Split | N | Official F1 (%) reproduced | README F1 (%) | Standard COnPOff micro | Standard COnPOff macro | Reproduction gate |
|---|---:|---:|---:|---:|---:|---:|---|
| VOCANO | valid | 765 | 3.194 | 3.194 | 0.0126 | 0.0122 | PASS |
| VOCANO | test | 769 | 3.352 | 3.352 | 0.0090 | 0.0088 | PASS |
| SheetSage | valid | 765 | 2.702 | 2.702 | 0.0097 | 0.0095 | PASS |
| SheetSage | test | 769 | 3.005 | 3.005 | 0.0096 | 0.0097 | PASS |
| MIR-ST500 | valid | 765 | 6.341 | 6.341 | 0.0316 | 0.0302 | PASS |
| MIR-ST500 | test | 769 | 5.755 | 5.755 | 0.0195 | 0.0195 | PASS |
| JDC-STP | valid | 765 | 6.741 | 6.741 | 0.0321 | 0.0316 | PASS |
| JDC-STP | test | 769 | 5.667 | 5.667 | 0.0192 | 0.0191 | PASS |

The official source script independently returns 3.134% recall for VOCANO/valid, whereas the README table prints 3.314%. Its precision and F1 reproduce exactly, so this is retained as a documented source table discrepancy rather than silently corrected.

Do not place the official HumTrans percentages in the same metric column as Molina COnPOff. The former ignores offsets and searches over global octave shifts with a 1-cent pitch tolerance.
