"""Loading, validation, cleaning and experimental classification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AnalysisData:
    """Cleaned, classified table plus the bookkeeping the report needs."""

    frame: pd.DataFrame
    n_before: int
    n_after: int
    has_probability: bool

    @property
    def n_dropped(self):
        return self.n_before - self.n_after

    def class_counts(self, config):
        poor, intermediate, strong = config.class_labels
        counts = self.frame["experimental_class"].value_counts()

        return {
            "total": len(self.frame),
            "poor": int(counts.get(poor, 0)),
            "intermediate": int(counts.get(intermediate, 0)),
            "strong": int(counts.get(strong, 0)),
        }


def load_table(path, sep=None):
    """Read a csv/tsv/parquet/xlsx table into a DataFrame."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)

    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")

    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    if sep:
        return pd.read_csv(path, sep=sep, engine="python")

    return pd.read_csv(path)


def classify_affinity(ka, config):
    """Map one Ka value onto an experimental class label."""
    poor, intermediate, strong = config.class_labels

    if pd.isna(ka):
        return None
    if ka < config.poor_ka_threshold:
        return poor
    if ka > config.strong_ka_threshold:
        return strong

    return intermediate


def prepare(source, config):
    """
    Validate columns, coerce to numeric, drop unusable rows and add the
    experimental class column.

    Any value that cannot be parsed as a number - blanks, "NA", "N/A", "None",
    "-" and friends - is coerced to NaN, and rows missing either the predicted
    score or Ka are dropped.

    source : DataFrame, or a path (falls back to `config.input_path`).
    """
    if source is None:
        source = config.input_path

    if source is None:
        raise ValueError("No input given: pass a DataFrame or set config.input_path")

    df = source if isinstance(source, pd.DataFrame) else load_table(source, config.sep)

    required = [config.sequence_col, config.affinity_col, config.ka_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    probability_present = (
        config.probability_col is not None and config.probability_col in df.columns
    )

    if config.probability_col is not None and not probability_present:
        warnings.warn(
            f"Probability column {config.probability_col!r} was not found; "
            "probability analysis and plotting will be skipped.",
            stacklevel=2,
        )

    columns = list(required)
    if probability_present:
        columns.append(config.probability_col)

    frame = df[columns].copy()

    for column in (config.affinity_col, config.ka_col):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if probability_present:
        frame[config.probability_col] = pd.to_numeric(
            frame[config.probability_col], errors="coerce"
        )

    n_before = len(frame)
    frame = frame.dropna(subset=[config.affinity_col, config.ka_col])
    frame = frame.reset_index(drop=True)
    n_after = len(frame)

    if n_after == 0:
        raise ValueError(
            "No rows have both a numeric predicted affinity and a numeric Ka; "
            f"check {config.affinity_col!r} and {config.ka_col!r}"
        )

    has_probability = bool(
        probability_present and frame[config.probability_col].notna().any()
    )

    if probability_present and not has_probability:
        warnings.warn(
            "The probability column exists but holds no valid numeric values; "
            "probability analysis and plotting will be skipped.",
            stacklevel=2,
        )

    frame["experimental_class"] = pd.Categorical(
        frame[config.ka_col].map(lambda ka: classify_affinity(ka, config)),
        categories=config.class_order,
        ordered=True,
    )

    return AnalysisData(
        frame=frame,
        n_before=n_before,
        n_after=n_after,
        has_probability=has_probability,
    )


def add_screening_decisions(frame, config):
    """Add `passes_proposed_cutoff` and a three-way `screening_decision` column."""
    from .metrics import retained_mask

    frame = frame.copy()
    scores = frame[config.affinity_col]

    passes = retained_mask(scores, config.proposed_cutoff, config)
    frame["passes_proposed_cutoff"] = passes

    if config.rescue_cutoff is None:
        frame["screening_decision"] = np.where(
            passes, "High priority", "Low priority / reject"
        )
        return frame

    in_rescue = ~passes & retained_mask(scores, config.rescue_cutoff, config)

    frame["screening_decision"] = np.select(
        [passes, in_rescue],
        ["High priority", "Rescue group"],
        default="Low priority / reject",
    )

    return frame
