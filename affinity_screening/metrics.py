"""Correlations, ROC discrimination and cutoff-by-cutoff screening performance.

Nothing here draws, prints or writes files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from .config import LOWER_IS_BETTER


# ------------------------------------------------------------
# screening rule
# ------------------------------------------------------------

def retained_mask(scores, cutoff, config):
    """Boolean mask of candidates retained at `cutoff`, honouring score direction."""
    scores = pd.Series(scores)
    return scores < cutoff if config.lower_is_better else scores > cutoff


def ranking_score(values, direction):
    """Orient a score so that larger always means 'more likely to be strong'."""
    values = np.asarray(values, dtype=float)
    return -values if direction == LOWER_IS_BETTER else values


# ------------------------------------------------------------
# cutoff performance
# ------------------------------------------------------------

def cutoff_metrics(frame, cutoff, config):
    """Screening performance of a single cutoff, as a flat dict."""
    valid = frame.dropna(subset=[config.affinity_col, config.ka_col])

    retained = retained_mask(valid[config.affinity_col], cutoff, config)
    rejected = ~retained

    ka = valid[config.ka_col]
    poor = ka < config.poor_ka_threshold
    strong = ka > config.strong_ka_threshold
    intermediate = ka.between(
        config.poor_ka_threshold, config.strong_ka_threshold, inclusive="both"
    )

    total_n = len(valid)
    retained_n = int(retained.sum())
    rejected_n = int(rejected.sum())

    poor_total = int(poor.sum())
    strong_total = int(strong.sum())
    intermediate_total = int(intermediate.sum())

    poor_retained_n = int((retained & poor).sum())
    poor_rejected_n = int((rejected & poor).sum())
    strong_retained_n = int((retained & strong).sum())
    strong_rejected_n = int((rejected & strong).sum())
    intermediate_retained_n = int((retained & intermediate).sum())
    intermediate_rejected_n = int((rejected & intermediate).sum())

    def pct(numerator, denominator):
        return 100 * numerator / denominator if denominator > 0 else np.nan

    return {
        "cutoff": cutoff,

        "total_n": total_n,
        "retained_n": retained_n,
        "rejected_n": rejected_n,
        "total_eliminated_pct": pct(rejected_n, total_n),

        "poor_total": poor_total,
        "poor_retained_n": poor_retained_n,
        "poor_rejected_n": poor_rejected_n,
        "poor_eliminated_pct": pct(poor_rejected_n, poor_total),

        "intermediate_total": intermediate_total,
        "intermediate_retained_n": intermediate_retained_n,
        "intermediate_rejected_n": intermediate_rejected_n,

        "strong_total": strong_total,
        "strong_retained_n": strong_retained_n,
        "strong_rejected_n": strong_rejected_n,
        "strong_retained_pct": pct(strong_retained_n, strong_total),

        "rejected_pool_poor_pct": pct(poor_rejected_n, rejected_n),
        "retained_pool_strong_pct": pct(strong_retained_n, retained_n),
    }


def cutoff_table(frame, cutoffs, config):
    """One row of `cutoff_metrics` per cutoff."""
    return pd.DataFrame([cutoff_metrics(frame, c, config) for c in cutoffs])


def summarise_cutoffs(table):
    """The reader-facing subset of `cutoff_table`, with presentable headers."""
    columns = {
        "cutoff": "Cutoff",
        "rejected_n": "Total eliminated (n)",
        "total_eliminated_pct": "Total eliminated (%)",
        "poor_rejected_n": "Poor eliminated (n)",
        "poor_eliminated_pct": "Poor eliminated (%)",
        "strong_retained_n": "Strong retained (n)",
        "strong_retained_pct": "Strong retained (%)",
        "strong_rejected_n": "Strong rejected (n)",
        "rejected_pool_poor_pct": "Rejected pool that is poor (%)",
        "retained_pool_strong_pct": "Retained pool that is strong (%)",
    }

    return table[list(columns)].rename(columns=columns)


def threshold_sweep(frame, config, points=None, pad=0.02):
    """Screening performance across the whole observed score range."""
    points = points or config.sweep_points
    scores = frame[config.affinity_col]

    grid = np.linspace(scores.min() - pad, scores.max() + pad, points)

    return cutoff_table(frame, grid, config)


# ------------------------------------------------------------
# correlations
# ------------------------------------------------------------

def _correlation_pair(frame, column, config):
    """Pearson vs log10(Ka) and Spearman vs Ka for one predictor column."""
    usable = frame[(frame[config.ka_col] > 0) & frame[column].notna()]

    if (
        len(usable) < 3
        or usable[column].nunique() < 2
        or usable[config.ka_col].nunique() < 2
    ):
        return [
            {"statistic": "Pearson r vs log10(Ka)", "value": np.nan,
             "p_value": np.nan, "n": len(usable)},
            {"statistic": "Spearman rho vs Ka", "value": np.nan,
             "p_value": np.nan, "n": len(usable)},
        ]

    log10_ka = np.log10(usable[config.ka_col])

    pearson_r, pearson_p = pearsonr(usable[column], log10_ka)
    spearman_rho, spearman_p = spearmanr(usable[column], usable[config.ka_col])

    return [
        {"statistic": "Pearson r vs log10(Ka)", "value": float(pearson_r),
         "p_value": float(pearson_p), "n": len(usable)},
        {"statistic": "Spearman rho vs Ka", "value": float(spearman_rho),
         "p_value": float(spearman_p), "n": len(usable)},
    ]


def correlation_table(frame, config, has_probability=False):
    """Tidy correlation table for the affinity score and, if present, the probability."""
    rows = []

    for entry in _correlation_pair(frame, config.affinity_col, config):
        rows.append({
            "output": f"{config.tool_name} affinity value",
            "column": config.affinity_col,
            "direction": config.score_direction,
            **entry,
        })

    if has_probability:
        for entry in _correlation_pair(frame, config.probability_col, config):
            rows.append({
                "output": f"{config.tool_name} binary probability",
                "column": config.probability_col,
                "direction": config.probability_direction,
                **entry,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# ROC: strong versus poor
# ------------------------------------------------------------

def roc_analysis(frame, column, config, direction):
    """
    ROC for ranking strong binders above poor ones, ignoring the
    intermediate class. Returns None when the comparison is not possible.
    """
    ka = frame[config.ka_col]
    extremes = frame[
        ((ka < config.poor_ka_threshold) | (ka > config.strong_ka_threshold))
        & frame[column].notna()
    ]

    if extremes.empty:
        return None

    labels = (extremes[config.ka_col] > config.strong_ka_threshold).astype(int)
    scores = ranking_score(extremes[column], direction)

    if labels.nunique() != 2 or len(np.unique(scores)) < 2:
        return None

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)

    n_strong = int(labels.sum())
    n_poor = int((labels == 0).sum())

    # Youden J: the cutoff that maximises (sensitivity + specificity - 1),
    # converted back into the original score scale.
    youden_index = int(np.argmax(tpr - fpr))
    youden_score = thresholds[youden_index]
    youden_cutoff = (
        -youden_score if direction == LOWER_IS_BETTER else youden_score
    )

    return {
        "column": column,
        "direction": direction,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "n_strong": n_strong,
        "n_poor": n_poor,
        "n_pairs": n_strong * n_poor,
        "concordant_pairs": auc * n_strong * n_poor,
        "youden_cutoff": float(youden_cutoff),
        "youden_tpr": float(tpr[youden_index]),
        "youden_fpr": float(fpr[youden_index]),
    }


def roc_results(frame, config, has_probability=False):
    """ROC for the affinity score and, when available, the probability."""
    results = {
        "affinity": roc_analysis(
            frame, config.affinity_col, config, config.score_direction
        )
    }

    if has_probability:
        results["probability"] = roc_analysis(
            frame, config.probability_col, config, config.probability_direction
        )
    else:
        results["probability"] = None

    return results
