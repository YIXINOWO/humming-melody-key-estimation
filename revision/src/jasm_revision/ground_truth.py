"""Strict, layout-aware readers for Molina/MTG-QBH note annotations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


class GroundTruthFormatError(ValueError):
    """Raised when an annotation cannot be interpreted without guessing."""


@dataclass(frozen=True)
class GroundTruthRecord:
    """Parsed annotations plus source-layout and validation metadata."""

    path: Path
    notes: pd.DataFrame
    layout: str
    nonempty_lines: int
    numeric_tokens: int
    sha256: str
    overlap_count: int


def _numeric_lines(path: Path) -> tuple[list[list[float]], bytes]:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8-sig")
    parsed: list[list[float]] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        fields = line.strip().split()
        if not fields:
            continue
        try:
            parsed.append([float(value) for value in fields])
        except ValueError as exc:
            raise GroundTruthFormatError(
                f"{path}: non-numeric value on line {line_number}: {line!r}"
            ) from exc
    if not parsed:
        raise GroundTruthFormatError(f"{path}: annotation is empty")
    return parsed, raw


def read_ground_truth(path: str | Path) -> GroundTruthRecord:
    """Read either one-note-per-row or one-value-per-row annotations.

    The corpus contains both layouts. Mixed-width rows are rejected because
    flattening them silently would make a damaged file look valid.
    """

    source = Path(path)
    lines, raw = _numeric_lines(source)
    widths = {len(line) for line in lines}

    if widths == {3}:
        layout = "note_rows"
        matrix = np.asarray(lines, dtype=np.float64)
    elif widths == {1}:
        layout = "value_vector"
        values = np.asarray([line[0] for line in lines], dtype=np.float64)
        if values.size % 3:
            raise GroundTruthFormatError(
                f"{source}: value-vector token count {values.size} is not divisible by 3"
            )
        matrix = values.reshape(-1, 3)
    else:
        raise GroundTruthFormatError(
            f"{source}: mixed row widths {sorted(widths)}; expected all 1 or all 3"
        )

    if matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise GroundTruthFormatError(f"{source}: expected finite onset/offset/MIDI triplets")

    notes = pd.DataFrame(matrix, columns=["onset", "offset", "midi"])
    if (notes["onset"] < 0).any():
        raise GroundTruthFormatError(f"{source}: negative onset")
    if (notes["offset"] <= notes["onset"]).any():
        rows = notes.index[notes["offset"] <= notes["onset"]].tolist()
        raise GroundTruthFormatError(f"{source}: offset <= onset at rows {rows[:10]}")
    if ((notes["midi"] < 0) | (notes["midi"] > 127)).any():
        raise GroundTruthFormatError(f"{source}: MIDI value outside [0, 127]")
    if not notes["onset"].is_monotonic_increasing:
        raise GroundTruthFormatError(f"{source}: onsets are not monotonically increasing")

    previous_offsets = notes["offset"].shift(1)
    overlap_count = int((notes["onset"] < previous_offsets).fillna(False).sum())
    notes["hz"] = 440.0 * np.power(2.0, (notes["midi"] - 69.0) / 12.0)

    return GroundTruthRecord(
        path=source,
        notes=notes,
        layout=layout,
        nonempty_lines=len(lines),
        numeric_tokens=int(matrix.size),
        sha256=sha256(raw).hexdigest(),
        overlap_count=overlap_count,
    )

