"""
Compare two numeric distributions.

Importable helpers
    distribution_similarity(a, b, ...)   overlaid histograms + Mann-Whitney / KS tests
    ecdf_plot(a, b, ...)                 empirical CDF step plot
    compare_distributions(a, b, ...)     density histograms + KDE curves

Each accepts anything load_values() understands: a DataFrame column, a Series,
a list/array, or a path to a CSV/TSV/Parquet file (with `column`).

Command line
    python distribution_comparison.py --a with_msa.csv --b without_msa.csv \
        --column affinity_pred_value --label-a "With MSA" --label-b "Without MSA" \
        --xlabel affinity_pred_value --plots kde ecdf hist --save-dir figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, mannwhitneyu

__all__ = [
    "load_values",
    "describe",
    "distribution_similarity",
    "ecdf_plot",
    "compare_distributions",
]


# ------------------------------------------------------------
# INPUT HANDLING
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


def load_values(source, column=None, sep=None, dropna=True):
    """
    Coerce `source` into a 1-D float array.

    source : path to csv/tsv/parquet/xlsx, DataFrame, Series, or array-like
    column : required for multi-column frames / table files
    """
    if isinstance(source, (str, Path)):
        source = _read_table(source, sep=sep)

    if isinstance(source, pd.DataFrame):
        if column is not None:
            if column not in source.columns:
                raise KeyError(
                    f"Column {column!r} not found. Available: {list(source.columns)}"
                )
            series = source[column]
        else:
            numeric = source.select_dtypes("number")
            if numeric.shape[1] != 1:
                raise ValueError(
                    "Pass `column`: frame has numeric columns "
                    f"{list(numeric.columns)}"
                )
            series = numeric.iloc[:, 0]
        values = series.to_numpy(dtype=float)

    elif isinstance(source, pd.Series):
        values = source.to_numpy(dtype=float)

    else:
        values = np.asarray(source, dtype=float).ravel()

    if dropna:
        values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("No finite values left after loading")

    return values


def describe(values, label="values"):
    """Return (and print) basic descriptive statistics."""
    values = np.asarray(values, dtype=float)

    stats = {
        "label": label,
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }

    print(
        f"{label:>24}: n={stats['n']:<6d} mean={stats['mean']:>10.4f} "
        f"median={stats['median']:>10.4f} std={stats['std']:>10.4f} "
        f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
    )

    return stats


# ------------------------------------------------------------
# FIGURE PLUMBING
# ------------------------------------------------------------

def _finish(fig, save=None, show=True):
    fig.tight_layout()

    if save:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"figure -> {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _shared_bins(a, b, bins=30):
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))

    if lo == hi:                      # degenerate: all values identical
        lo, hi = lo - 0.5, hi + 0.5

    return np.linspace(lo, hi, bins + 1), lo, hi


def _kde_or_none(values, label):
    """gaussian_kde needs >1 point and non-zero spread."""
    values = np.asarray(values, dtype=float)

    if values.size < 2 or np.ptp(values) == 0:
        print(f"note: skipping KDE for {label!r} (n={values.size}, zero spread)")
        return None

    return gaussian_kde(values)


# ------------------------------------------------------------
# 1. HISTOGRAMS + STATISTICAL TESTS
# ------------------------------------------------------------

def distribution_similarity(
    list1,
    list2,
    plot=None,
    tests=True,
    bins=10,
    save=None,
    show=True,
):
    """
    Overlaid histograms plus Mann-Whitney U and two-sample KS tests.

    plot : dict or None/False. Keys (all optional):
           draw, label1, label2, xlabel, ylabel, title
           Pass plot=False (or {'draw': False}) to skip the figure.
    """
    defaults = {
        "draw": True,
        "label1": "Plot 1",
        "label2": "Plot 2",
        "xlabel": "ΔEnergy",
        "ylabel": "Frequency",
        "title": "Distribution similarity",
    }

    cfg = dict(defaults)
    if isinstance(plot, dict):
        cfg.update(plot)
    elif plot is False or plot is None:
        cfg["draw"] = bool(plot)

    if cfg["draw"]:
        hist_bins, _, _ = _shared_bins(list1, list2, bins)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(list1, bins=hist_bins, alpha=0.6, label=cfg["label1"])
        ax.hist(list2, bins=hist_bins, alpha=0.6, label=cfg["label2"])
        ax.set_xlabel(cfg["xlabel"])
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"])
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        _finish(fig, save, show)

    if not tests:
        return None

    # --- Mann-Whitney U Test ---
    u_stat, p_u = mannwhitneyu(list1, list2, alternative="two-sided")

    # --- Kolmogorov-Smirnov Test ---
    ks_stat, p_ks = ks_2samp(list1, list2)

    # Effect size: common-language / rank-biserial (Cliff's delta)
    n1, n2 = len(list1), len(list2)
    cles = u_stat / (n1 * n2)
    cliffs_delta = 2 * cles - 1

    print(f"Mann-Whitney U Test: U = {u_stat}, p = {p_u:.4g}")
    print(f"Kolmogorov-Smirnov Test: D = {ks_stat:.4f}, p = {p_ks:.4g}")
    print(f"Effect size: CLES = {cles:.3f}, Cliff's delta = {cliffs_delta:+.3f}")

    return {
        "n1": n1,
        "n2": n2,
        "u_stat": float(u_stat),
        "p_u": float(p_u),
        "ks_stat": float(ks_stat),
        "p_ks": float(p_ks),
        "cles": float(cles),
        "cliffs_delta": float(cliffs_delta),
    }


# ------------------------------------------------------------
# 2. EMPIRICAL CDF
# ------------------------------------------------------------

def ecdf_plot(
    list1,
    list2,
    label1="Distribution 1",
    label2="Distribution 2",
    xlabel="value",
    title="Empirical Cumulative Distribution",
    save=None,
    show=True,
):
    x1 = np.sort(list1)
    y1 = np.arange(1, len(x1) + 1) / len(x1)

    x2 = np.sort(list2)
    y2 = np.arange(1, len(x2) + 1) / len(x2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(x1, y1, where="post", lw=2.5, label=label1)
    ax.step(x2, y2, where="post", lw=2.5, label=label2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    return _finish(fig, save, show)


# ------------------------------------------------------------
# 3. DENSITY HISTOGRAMS + KDE
# ------------------------------------------------------------

def compare_distributions(
    list1,
    list2,
    label1="Distribution 1",
    label2="Distribution 2",
    xlabel="ΔEnergy",
    title="Distribution Comparison",
    bins=30,
    save=None,
    show=True,
):
    hist_bins, lo, hi = _shared_bins(list1, list2, bins)
    x = np.linspace(lo, hi, 1000)

    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label in ((list1, label1), (list2, label2)):
        ax.hist(
            data,
            bins=hist_bins,
            density=True,
            alpha=0.30,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )

    for data, label in ((list1, label1), (list2, label2)):
        kde = _kde_or_none(data, label)
        if kde is not None:
            ax.plot(x, kde(x), linewidth=3, label=f"{label} KDE")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    return _finish(fig, save, show)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _slug(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def run_comparison(
    source_a,
    source_b,
    column=None,
    column_a=None,
    column_b=None,
    label_a="Distribution 1",
    label_b="Distribution 2",
    xlabel=None,
    title=None,
    bins=30,
    plots=("kde", "ecdf"),
    tests=True,
    sep=None,
    save_dir=None,
    show=True,
    stats_out=None,
):
    """Load both inputs, draw the requested plots, run the tests."""
    a = load_values(source_a, column_a or column, sep=sep)
    b = load_values(source_b, column_b or column, sep=sep)

    xlabel = xlabel or column_a or column or "value"
    title = title or f"{label_a} vs {label_b}"

    save_dir = Path(save_dir) if save_dir else None
    stem = _slug(f"{label_a}_vs_{label_b}")

    def _path(kind):
        return save_dir / f"{stem}_{kind}.png" if save_dir else None

    print(f"\n=== {title} ===")
    summary = {
        "title": title,
        "xlabel": xlabel,
        "a": describe(a, label_a),
        "b": describe(b, label_b),
    }
    print()

    plots = tuple(plots or ())

    if "kde" in plots:
        compare_distributions(
            a, b,
            label1=label_a, label2=label_b,
            xlabel=xlabel, title=title, bins=bins,
            save=_path("kde"), show=show,
        )

    if "ecdf" in plots:
        ecdf_plot(
            a, b,
            label1=label_a, label2=label_b,
            xlabel=xlabel, title=f"{title} - ECDF",
            save=_path("ecdf"), show=show,
        )

    hist_cfg = {
        "draw": "hist" in plots,
        "label1": label_a,
        "label2": label_b,
        "xlabel": xlabel,
        "title": f"{title} - counts",
    }

    stats = distribution_similarity(
        a, b,
        plot=hist_cfg,
        tests=tests,
        bins=bins,
        save=_path("hist") if hist_cfg["draw"] else None,
        show=show,
    )

    if stats:
        summary["tests"] = stats

    if stats_out:
        stats_out = Path(stats_out)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(summary, indent=2))
        print(f"stats -> {stats_out}")

    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare two numeric distributions (histogram/KDE, ECDF, "
                    "Mann-Whitney U, Kolmogorov-Smirnov).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--a", required=True, help="first dataset (csv/tsv/parquet/xlsx)")
    p.add_argument("--b", required=True, help="second dataset")

    p.add_argument("--column", help="column to compare in both files")
    p.add_argument("--column-a", help="column in --a (overrides --column)")
    p.add_argument("--column-b", help="column in --b (overrides --column)")
    p.add_argument("--sep", help="delimiter override for text tables")

    p.add_argument("--label-a", default="Distribution 1")
    p.add_argument("--label-b", default="Distribution 2")
    p.add_argument("--xlabel", help="x-axis label (default: column name)")
    p.add_argument("--title", help="figure title")
    p.add_argument("--bins", type=int, default=30)

    p.add_argument(
        "--plots",
        nargs="*",
        default=["kde", "ecdf"],
        choices=["kde", "ecdf", "hist", "none"],
        help="figures to draw",
    )
    p.add_argument("--no-tests", action="store_true", help="skip the statistical tests")

    p.add_argument("--save-dir", help="write PNGs here instead of only showing them")
    p.add_argument("--no-show", action="store_true", help="do not open figure windows")
    p.add_argument("--stats-out", help="write the summary as JSON to this path")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    show = not args.no_show
    if not show:
        plt.switch_backend("Agg")

    plots = [] if "none" in args.plots else args.plots

    run_comparison(
        args.a,
        args.b,
        column=args.column,
        column_a=args.column_a,
        column_b=args.column_b,
        label_a=args.label_a,
        label_b=args.label_b,
        xlabel=args.xlabel,
        title=args.title,
        bins=args.bins,
        plots=plots,
        tests=not args.no_tests,
        sep=args.sep,
        save_dir=args.save_dir,
        show=show,
        stats_out=args.stats_out,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
