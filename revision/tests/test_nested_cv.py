from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "revision" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jasm_revision.nested_cv import assert_group_isolation, grouped_splits


def test_grouped_splits_are_isolated_and_complete() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    seen = np.zeros(len(groups), dtype=int)
    for train, test in grouped_splits(groups, n_splits=2, seed=2026):
        assert_group_isolation(groups, train, test)
        seen[test] += 1
    assert seen.tolist() == [1] * len(groups)
