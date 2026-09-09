"""Configuration for an affinity-screening analysis.

Everything that is dataset- or tool-specific lives in `ScreeningConfig`; the
rest of the package takes one of these and never hard-codes a column name,
threshold or path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOWER_IS_BETTER = "lower_is_better"      # docking scores, e.g. GNINA minimizedAffinity
HIGHER_IS_BETTER = "higher_is_better"    # probabilities, pKd-like scores

SCORE_DIRECTIONS = (LOWER_IS_BETTER, HIGHER_IS_BETTER)


@dataclass
class ScreeningConfig:
    """
    Parameters for one screening analysis.

    Required columns in the input table: `sequence_col`, `affinity_col`,
    `ka_col`. `probability_col` is optional and skipped when absent or empty.

    Experimental classes (Ka in the same units as `ka_col`, typically 10^9 M^-1):
        poor          Ka <  poor_ka_threshold
        intermediate  poor_ka_threshold <= Ka <= strong_ka_threshold
        strong        Ka >  strong_ka_threshold

    Screening rule, depending on `score_direction`:
        lower_is_better   retain when score <  cutoff
        higher_is_better  retain when score >  cutoff
    """

    # --- input ---
    input_path: str | None = None
    sep: str | None = None

    # --- naming ---
    tool_name: str = "Tool"

    # --- columns ---
    sequence_col: str = "sequence_id"
    affinity_col: str = "affinity"
    ka_col: str = "Ka (10^9)"
    probability_col: str | None = None

    # --- semantics ---
    score_direction: str = LOWER_IS_BETTER
    probability_direction: str = HIGHER_IS_BETTER

    # --- experimental classes ---
    poor_ka_threshold: float = 1.0
    strong_ka_threshold: float = 7.0

    # --- screening rule ---
    proposed_cutoff: float = 0.0
    rescue_cutoff: float | None = None
    test_cutoffs: list[float] = field(default_factory=list)

    # --- output ---
    output_dir: str = "screening_analysis"
    sweep_points: int = 600
    dpi: int = 300
    show: bool = False

    # cosmetic: substrings stripped from sequence ids in plot labels
    label_strip: list[str] = field(default_factory=list)

    # ------------------------------------------------------------
    # construction
    # ------------------------------------------------------------

    @classmethod
    def from_json(cls, path):
        """Load a config from a JSON file; unknown keys are rejected loudly."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known

        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. Known keys: {sorted(known)}"
            )

        return cls(**payload)

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def replace(self, **changes):
        merged = {**asdict(self), **changes}
        return type(self)(**merged)

    # ------------------------------------------------------------
    # derived properties
    # ------------------------------------------------------------

    @property
    def lower_is_better(self):
        return self.score_direction == LOWER_IS_BETTER

    @property
    def class_labels(self):
        """(poor, intermediate, strong) labels, built from the thresholds."""
        return (
            f"Poor (<{self.poor_ka_threshold:g})",
            f"Intermediate ({self.poor_ka_threshold:g}-{self.strong_ka_threshold:g})",
            f"Strong (>{self.strong_ka_threshold:g})",
        )

    @property
    def class_order(self):
        return list(self.class_labels)

    @property
    def palette(self):
        poor, intermediate, strong = self.class_labels
        return {poor: "#D95F5F", intermediate: "#9AA3AD", strong: "#238B57"}

    @property
    def retention_rule(self):
        comparison = "<" if self.lower_is_better else ">"
        return f"retain if {self.affinity_col} {comparison} cutoff"

    @property
    def score_hint(self):
        return "more negative is better" if self.lower_is_better else "higher is better"

    def output_path(self, filename):
        return Path(self.output_dir) / filename

    # ------------------------------------------------------------
    # validation
    # ------------------------------------------------------------

    def validate(self):
        """Raise on anything that would silently produce a misleading analysis."""
        problems = []

        if self.score_direction not in SCORE_DIRECTIONS:
            problems.append(
                f"score_direction must be one of {SCORE_DIRECTIONS}, "
                f"got {self.score_direction!r}"
            )

        if self.probability_direction not in SCORE_DIRECTIONS:
            problems.append(
                f"probability_direction must be one of {SCORE_DIRECTIONS}, "
                f"got {self.probability_direction!r}"
            )

        if self.poor_ka_threshold > self.strong_ka_threshold:
            problems.append(
                f"poor_ka_threshold ({self.poor_ka_threshold}) must not exceed "
                f"strong_ka_threshold ({self.strong_ka_threshold})"
            )

        if self.rescue_cutoff is not None:
            if self.lower_is_better and self.rescue_cutoff <= self.proposed_cutoff:
                problems.append(
                    f"with {LOWER_IS_BETTER}, rescue_cutoff ({self.rescue_cutoff}) must "
                    f"be greater than proposed_cutoff ({self.proposed_cutoff}); "
                    "otherwise the rescue band is empty"
                )
            if not self.lower_is_better and self.rescue_cutoff >= self.proposed_cutoff:
                problems.append(
                    f"with {HIGHER_IS_BETTER}, rescue_cutoff ({self.rescue_cutoff}) must "
                    f"be less than proposed_cutoff ({self.proposed_cutoff}); "
                    "otherwise the rescue band is empty"
                )

        if self.sweep_points < 2:
            problems.append("sweep_points must be at least 2")

        if problems:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))

        return self
