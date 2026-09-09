"""
Compare two scoring conditions over the SAME candidates (paired).

Use this when row i of both columns is the same molecule/design scored twice —
with vs without MSA, checkpoint A vs B, 1k vs 10k samples. Because the
observations are paired, it can ask the questions that matter for a screen:
does the ranking survive, and would the same candidates be picked?

    trend        Spearman rho, Kendall tau, Pearson r
    selection    top-k overlap (and the full overlap-vs-k curve)
    shift        Wilcoxon signed-rank, mean/median paired difference
    shape        histogram Jensen-Shannon divergence (raw and standardized),
                 two-sample KS

For two *unpaired* score sets (different candidates, or no row correspondence)
use `distribution_comparison.py` instead — the paired statistics here are
meaningless if the rows do not line up.

Command line
    python paired_score_comparison.py \
        --input e2pico_msa_effect_comparison.csv \
        --col-a estradiol_affinity_pred_value_ensemble_mean \
        --col-b affinity_pred_value_mean \
        --id-col sequence_id \
        --label-a "With MSA" --label-b "Without MSA" \
        --top-ks 5 10 20 --save-fig msa_effect.png --no-show \
        --summary-out msa_summary.csv --details-out msa_details.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, ks_2samp, pearsonr, rankdata, spearmanr, wilcoxon

PANELS = ("scatter", "ranks", "ecdf", "topk", "bland_altman")

__all__ = [
    "load_paired",
    "robust_standardize",
    "js_divergence_histogram",
    "top_k_overlap",
    "top_k_curve",
    "add_ranks",
    "paired_statistics",
    "plot_paired",
    "compare_paired_scores",
]


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

def _read_table(path, sep=None):
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


def load_paired(source, col_a=None, col_b=None, id_col=None, sep=None, verbose=True):
    """
    Build a tidy paired frame with columns `id`, `a`, `b`.

    source : a path or DataFrame (then `col_a`/`col_b` name the columns), or a
             pair of array-likes passed as (values_a, values_b).

    Rows are dropped only when *either* value is missing, which is the only way
    to keep the pairing intact. Filtering each column separately — a list
    comprehension per column, say — silently misaligns the two series whenever
    their missing values sit in different rows.
    """
    if isinstance(source, (str, Path)):
        source = _read_table(source, sep=sep)

    if isinstance(source, pd.DataFrame):
        if col_a is None or col_b is None:
            raise ValueError("Pass col_a and col_b when the source is a table")

        for column in (col_a, col_b):
            if column not in source.columns:
                raise KeyError(
                    f"Column {column!r} not found. Available: {list(source.columns)}"
                )

        values_a = source[col_a]
        values_b = source[col_b]

        if id_col is not None:
            if id_col not in source.columns:
                raise KeyError(
                    f"Id column {id_col!r} not found. Available: {list(source.columns)}"
                )
            ids = source[id_col].to_numpy()
        else:
            ids = np.arange(len(source))

    else:
        values_a, values_b = source
        ids = np.arange(len(pd.Series(values_a)))

    series_a = pd.Series(values_a).reset_index(drop=True)
    series_b = pd.Series(values_b).reset_index(drop=True)

    # Checked before building the frame: pandas would pad the shorter column
    # with NaN, turning a misalignment into a few silently dropped rows.
    if not len(series_a) == len(series_b) == len(ids):
        raise ValueError(
            "Paired inputs must be the same length: "
            f"a={len(series_a)}, b={len(series_b)}, ids={len(ids)}"
        )

    frame = pd.DataFrame(
        {
            "id": ids,
            "a": pd.to_numeric(series_a, errors="coerce"),
            "b": pd.to_numeric(series_b, errors="coerce"),
        }
    )

    n_before = len(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["a", "b"]).reset_index(drop=True)
    n_after = len(frame)

    if verbose and n_after < n_before:
        print(f"dropped {n_before - n_after} row(s) missing one or both values")

    if n_after < 3:
        raise ValueError(
            f"At least three complete paired observations are required, got {n_after}"
        )

    return frame


# ------------------------------------------------------------
# DISTRIBUTION HELPERS
# ------------------------------------------------------------

def robust_standardize(values):
    """
    Median/IQR standardization: removes differences in location and scale while
    retaining distribution shape.
    """
    values = np.asarray(values, dtype=float)

    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if np.isclose(iqr, 0):
        std = np.std(values, ddof=1)

        if np.isclose(std, 0):
            return np.zeros_like(values)

        return (values - np.mean(values)) / std

    return (values - median) / iqr


def js_divergence_histogram(x, y, bins=15, standardize=False):
    """
    Histogram Jensen-Shannon divergence in bits: 0 = identical, 1 = disjoint.

    Both samples are binned on shared edges spanning their combined range.
    The value depends on the number and placement of bins, so compare only
    values computed with the same `bins`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if standardize:
        x = robust_standardize(x)
        y = robust_standardize(y)

    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())

    if np.isclose(lower, upper):
        return 0.0

    edges = np.linspace(lower, upper, bins + 1)

    counts_x, _ = np.histogram(x, bins=edges)
    counts_y, _ = np.histogram(y, bins=edges)

    # A small pseudocount keeps empty bins numerically harmless.
    p = counts_x.astype(float) + 1e-12
    q = counts_y.astype(float) + 1e-12

    p /= p.sum()
    q /= q.sum()

    # scipy returns the JS *distance*, i.e. sqrt(divergence).
    return float(jensenshannon(p, q, base=2) ** 2)


# ------------------------------------------------------------
# SELECTION OVERLAP
# ------------------------------------------------------------

def _top_indices(values, k, higher_is_better):
    order = np.argsort(values)
    return set(order[-k:]) if higher_is_better else set(order[:k])


def top_k_overlap(x, y, k=10, higher_is_better=True):
    """Fraction of the top-k selection shared between the two conditions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    k = min(k, len(x))
    shared = len(
        _top_indices(x, k, higher_is_better) & _top_indices(y, k, higher_is_better)
    )

    return {"k": k, "shared": shared, "overlap_fraction": shared / k}


def top_k_curve(x, y, higher_is_better=True):
    """Overlap fraction for every k from 1 to n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return np.array(
        [
            top_k_overlap(x, y, k=k, higher_is_better=higher_is_better)[
                "overlap_fraction"
            ]
            for k in range(1, len(x) + 1)
        ]
    )


# ------------------------------------------------------------
# RANKS
# ------------------------------------------------------------

def add_ranks(frame, higher_is_better=False):
    """Add rank columns (rank 1 = best), the rank change and the score difference."""
    frame = frame.copy()
    ascending = not higher_is_better

    frame["rank_a"] = frame["a"].rank(method="average", ascending=ascending)
    frame["rank_b"] = frame["b"].rank(method="average", ascending=ascending)

    frame["rank_change"] = frame["rank_b"] - frame["rank_a"]
    frame["absolute_rank_change"] = frame["rank_change"].abs()
    frame["score_difference"] = frame["b"] - frame["a"]

    return frame


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def paired_statistics(frame, higher_is_better=False, bins=15, top_ks=(5, 10, 20)):
    """
    Every paired statistic, as a flat dict.

    Differences are always reported as b - a.
    """
    x = frame["a"].to_numpy()
    y = frame["b"].to_numpy()
    differences = y - x

    spearman_rho, spearman_p = spearmanr(x, y)
    kendall_tau, kendall_p = kendalltau(x, y)
    pearson_r, pearson_p = pearsonr(x, y)

    ks_stat, ks_p = ks_2samp(x, y)

    if np.allclose(differences, 0):
        wilcoxon_stat, wilcoxon_p, rank_biserial = 0.0, 1.0, 0.0
    else:
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                x, y, alternative="two-sided", zero_method="wilcox"
            )
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        nonzero = differences[differences != 0]
        ranks = rankdata(np.abs(nonzero))
        t_plus = ranks[nonzero > 0].sum()
        t_minus = ranks[nonzero < 0].sum()
        rank_biserial = float((t_plus - t_minus) / (t_plus + t_minus))

    ranked = add_ranks(frame, higher_is_better)

    summary = {
        "n_paired": len(frame),

        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),

        "js_divergence_raw": js_divergence_histogram(x, y, bins=bins),
        "js_divergence_standardized": js_divergence_histogram(
            x, y, bins=bins, standardize=True
        ),

        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),

        "wilcoxon_statistic": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "wilcoxon_rank_biserial": rank_biserial,

        "mean_difference_b_minus_a": float(np.mean(differences)),
        "median_difference_b_minus_a": float(np.median(differences)),

        "mean_absolute_rank_change": float(ranked["absolute_rank_change"].mean()),
        "median_absolute_rank_change": float(ranked["absolute_rank_change"].median()),
        "max_absolute_rank_change": float(ranked["absolute_rank_change"].max()),
    }

    for k in top_ks:
        if k <= len(frame):
            overlap = top_k_overlap(x, y, k=k, higher_is_better=higher_is_better)
            summary[f"top_{k}_overlap"] = overlap["overlap_fraction"]
            summary[f"top_{k}_shared"] = overlap["shared"]

    return summary


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

def _panel_scatter(ax, frame, stats, label_a, label_b, **_):
    x = frame["a"].to_numpy()
    y = frame["b"].to_numpy()

    ax.scatter(x, y, s=60, alpha=0.75, edgecolor="black", linewidth=0.5)

    low = min(x.min(), y.min())
    high = max(x.max(), y.max())

    ax.plot([low, high], [low, high], linestyle="--", color="grey",
            linewidth=1.5, label="Identity line")

    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(
        "Paired scores\n"
        f"Spearman ρ = {stats['spearman_rho']:.3f}; "
        f"Kendall τ = {stats['kendall_tau']:.3f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def _panel_ranks(ax, frame, stats, label_a, label_b, annotate=5, **_):
    rank_x = frame["rank_a"].to_numpy()
    rank_y = frame["rank_b"].to_numpy()
    limit = len(frame)

    ax.scatter(rank_x, rank_y, s=60, alpha=0.75, edgecolor="black",
               linewidth=0.5, label="Candidates")

    ax.plot([1, limit], [1, limit], linestyle="--", color="grey",
            linewidth=1.5, label="Identical rank")

    slope, intercept = np.polyfit(rank_x, rank_y, deg=1)
    line_x = np.linspace(1, limit, 200)
    ax.plot(line_x, slope * line_x + intercept, color="darkred",
            linewidth=2, label="Observed trend")

    if annotate:
        for _, row in frame.nlargest(annotate, "absolute_rank_change").iterrows():
            ax.annotate(
                str(row["id"]),
                (row["rank_a"], row["rank_b"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(f"Rank: {label_a}")
    ax.set_ylabel(f"Rank: {label_b}")
    ax.set_title(
        "Candidate rank preservation\n"
        f"mean |Δrank| = {stats['mean_absolute_rank_change']:.1f}"
    )

    # Rank 1 (best) sits at the upper right.
    ax.set_xlim(limit + 1, 0)
    ax.set_ylim(limit + 1, 0)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def _panel_ecdf(ax, frame, stats, label_a, label_b, xlabel="score", **_):
    for values, label in ((frame["a"].to_numpy(), label_a),
                          (frame["b"].to_numpy(), label_b)):
        ordered = np.sort(values)
        ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.step(ordered, ecdf, where="post", linewidth=2.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(
        "Empirical cumulative distributions\n"
        f"KS D = {stats['ks_statistic']:.3f}; p = {stats['ks_p']:.3g}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def _panel_topk(ax, frame, stats, label_a, label_b, higher_is_better=False,
                top_ks=(5, 10, 20), **_):
    x = frame["a"].to_numpy()
    y = frame["b"].to_numpy()
    n = len(frame)

    k_values = np.arange(1, n + 1)
    overlap = top_k_curve(x, y, higher_is_better=higher_is_better) * 100

    ax.plot(k_values, overlap, marker="o", markersize=3.5, linewidth=2,
            label="Observed overlap")

    # Two independent top-k selections from n share k/n of their members.
    ax.plot(k_values, k_values / n * 100, linestyle="--", color="grey",
            linewidth=1.5, label="Expected by chance")

    for k in top_ks:
        if k <= n:
            ax.scatter(k, overlap[k - 1], s=70, zorder=3)
            ax.annotate(
                f"Top {k}: {overlap[k - 1]:.0f}%",
                (k, overlap[k - 1]),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Number of top-ranked candidates selected")
    ax.set_ylabel("Shared candidates (%)")
    ax.set_title("Top-k candidate overlap")
    ax.set_xlim(1, n)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def _panel_bland_altman(ax, frame, stats, label_a, label_b, **_):
    mean_score = (frame["a"] + frame["b"]) / 2
    difference = frame["b"] - frame["a"]

    bias = difference.mean()
    deviation = difference.std(ddof=1)

    ax.scatter(mean_score, difference, s=60, alpha=0.75,
               edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="grey", linestyle="--", linewidth=1.5)
    ax.axhline(bias, color="darkred", linewidth=2, label=f"Bias = {bias:+.3f}")

    for sign in (1, -1):
        ax.axhline(
            bias + sign * 1.96 * deviation,
            color="darkred",
            linestyle=":",
            linewidth=1.4,
            label="95% limits of agreement" if sign == 1 else None,
        )

    ax.set_xlabel("Mean of the two conditions")
    ax.set_ylabel(f"{label_b} − {label_a}")
    ax.set_title(
        "Paired difference vs magnitude\n"
        f"Wilcoxon p = {stats['wilcoxon_p']:.3g}; "
        f"rank-biserial = {stats['wilcoxon_rank_biserial']:+.2f}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


PANEL_FUNCTIONS = {
    "scatter": _panel_scatter,
    "ranks": _panel_ranks,
    "ecdf": _panel_ecdf,
    "topk": _panel_topk,
    "bland_altman": _panel_bland_altman,
}


def plot_paired(
    frame,
    stats,
    panels=("scatter", "ranks", "ecdf", "topk"),
    label_a="Condition A",
    label_b="Condition B",
    xlabel="score",
    higher_is_better=False,
    top_ks=(5, 10, 20),
    annotate=5,
    ncols=2,
    save=None,
    show=True,
    suptitle=None,
):
    """Draw the requested panels into one figure."""
    panels = [p for p in panels if p in PANEL_FUNCTIONS]

    if not panels:
        return None

    ncols = max(1, min(ncols, len(panels)))
    nrows = int(np.ceil(len(panels) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False
    )
    axes = axes.reshape(-1)

    for ax, panel in zip(axes, panels):
        PANEL_FUNCTIONS[panel](
            ax,
            frame,
            stats,
            label_a,
            label_b,
            xlabel=xlabel,
            higher_is_better=higher_is_better,
            top_ks=top_ks,
            annotate=annotate,
        )

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold")

    fig.tight_layout()

    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"figure -> {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------

def compare_paired_scores(
    source,
    col_a=None,
    col_b=None,
    id_col=None,
    label_a="Condition A",
    label_b="Condition B",
    xlabel=None,
    bins=15,
    top_ks=(5, 10, 20),
    higher_is_better=False,
    annotate=5,
    panels=("scatter", "ranks", "ecdf", "topk"),
    plot=True,
    save_fig=None,
    show=True,
    sep=None,
    summary_out=None,
    details_out=None,
    verbose=True,
):
    """
    Full paired comparison.

    Returns (summary_df, detailed_df) — one row of statistics, and the
    per-candidate table with ranks, rank changes and score differences.
    """
    frame = load_paired(source, col_a, col_b, id_col=id_col, sep=sep, verbose=verbose)

    stats = paired_statistics(
        frame, higher_is_better=higher_is_better, bins=bins, top_ks=top_ks
    )

    detailed = add_ranks(frame, higher_is_better=higher_is_better)
    detailed = detailed.rename(columns={"a": label_a, "b": label_b})

    summary_df = pd.DataFrame([stats])

    if verbose:
        print(f"\n=== {label_a} vs {label_b} (paired, n = {stats['n_paired']}) ===")
        for key, value in stats.items():
            if key == "n_paired":
                continue
            shown = f"{value:.4g}" if isinstance(value, float) else value
            print(f"  {key:<38} {shown}")

    if plot:
        plot_paired(
            add_ranks(frame, higher_is_better=higher_is_better),
            stats,
            panels=panels,
            label_a=label_a,
            label_b=label_b,
            xlabel=xlabel or (col_a if isinstance(col_a, str) else "score"),
            higher_is_better=higher_is_better,
            top_ks=top_ks,
            annotate=annotate,
            save=save_fig,
            show=show,
            suptitle=f"{label_a} vs {label_b}",
        )

    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_out, index=False)
        print(f"summary -> {summary_out}")

    if details_out:
        Path(details_out).parent.mkdir(parents=True, exist_ok=True)
        detailed.sort_values("absolute_rank_change", ascending=False).to_csv(
            details_out, index=False
        )
        print(f"details -> {details_out}")

    return summary_df, detailed


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Paired comparison of two scoring conditions over the same "
                    "candidates: rank preservation, top-k overlap, paired shift "
                    "and distribution shape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", required=True, help="table with both score columns")
    p.add_argument("--col-a", required=True, help="first score column")
    p.add_argument("--col-b", required=True, help="second score column")
    p.add_argument("--id-col", help="candidate identifier column (used for labels)")
    p.add_argument("--sep", help="delimiter override for text tables")

    p.add_argument("--label-a", default="Condition A")
    p.add_argument("--label-b", default="Condition B")
    p.add_argument("--xlabel", help="score axis label (default: --col-a)")

    p.add_argument("--higher-is-better", action="store_true",
                   help="higher scores rank better (default: lower is better)")
    p.add_argument("--top-ks", type=int, nargs="*", default=[5, 10, 20],
                   help="selection sizes reported and highlighted")
    p.add_argument("--bins", type=int, default=15,
                   help="histogram bins for the JS divergence")
    p.add_argument("--annotate", type=int, default=5,
                   help="number of largest rank movers to label")

    p.add_argument("--plots", nargs="*", default=["scatter", "ranks", "ecdf", "topk"],
                   choices=[*PANELS, "none"], help="panels to draw")
    p.add_argument("--ncols", type=int, default=2, help="panels per row")

    p.add_argument("--save-fig", help="write the figure here")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--summary-out", help="write the one-row summary CSV here")
    p.add_argument("--details-out", help="write the per-candidate CSV here")
    p.add_argument("--quiet", action="store_true", help="suppress the console summary")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    show = not args.no_show
    if not show:
        plt.switch_backend("Agg")

    panels = [] if "none" in args.plots else args.plots

    compare_paired_scores(
        args.input,
        col_a=args.col_a,
        col_b=args.col_b,
        id_col=args.id_col,
        label_a=args.label_a,
        label_b=args.label_b,
        xlabel=args.xlabel,
        bins=args.bins,
        top_ks=tuple(args.top_ks),
        higher_is_better=args.higher_is_better,
        annotate=args.annotate,
        panels=tuple(panels),
        plot=bool(panels),
        save_fig=args.save_fig,
        show=show,
        sep=args.sep,
        summary_out=args.summary_out,
        details_out=args.details_out,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
