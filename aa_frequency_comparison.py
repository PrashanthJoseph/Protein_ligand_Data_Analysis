"""
Per-position amino-acid frequency comparison between two design sets.

Given two tables of designed sequences (e.g. BoltzGen CDR-H3 designs from two
runs), build a 20 x L position frequency matrix for each and compare them
position by position: Jensen-Shannon divergence, total-variation distance,
Spearman rho, and per-position absolute differences.

Importable helpers
    clean_and_trim(df, col, left_trim, right_trim)   validate/trim to equal length
    aa_frequency_matrix(seqs)                        20 x L frequency matrix
    compare_position_distributions(freq_a, freq_b)   per-position statistics
    plot_all_position_overlays(freq_a, freq_b)       grouped bars per position
    plot_difference_heatmap(freq_a, freq_b)          (b - a) heatmap
    plot_js_per_position(stats)                      divergence profile

Command line
    python aa_frequency_comparison.py \
        --a variable_designs.csv --b conserved_designs.csv \
        --column designed_sequence --left-trim-a 3 \
        --label-a "Full CDRH3 variable" --label-b "Conserved CDRH3" \
        --plots overlay heatmap js --save-dir figs --no-show \
        --stats-out figs/position_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

__all__ = [
    "AA_ORDER",
    "clean_and_trim",
    "aa_frequency_matrix",
    "position_entropy",
    "compare_position_distributions",
    "plot_all_position_overlays",
    "plot_difference_heatmap",
    "plot_js_per_position",
    "run_comparison",
]


# ------------------------------------------------------------
# SEQUENCE PREP
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


def clean_and_trim(
    df,
    sequence_col,
    left_trim=0,
    right_trim=0,
    insert=None,
    insert_at=None,
    verbose=True,
):
    """
    Pull `sequence_col` out of `df`, keep only sequences of standard amino
    acids, optionally splice in a motif, trim the flanks, and require that
    everything left is the same length.

    insert / insert_at : splice `insert` into every sequence at index
        `insert_at` before trimming — used to re-align a set whose conserved
        stretch was held fixed (and therefore absent) during design.
    """
    raw = df[sequence_col].dropna().astype(str).str.upper().str.strip().tolist()
    n_raw = len(raw)

    seqs = [s for s in raw if s and all(aa in AA_ORDER for aa in s)]
    n_nonstandard = n_raw - len(seqs)

    if insert:
        insert = insert.upper()
        bad = [aa for aa in insert if aa not in AA_ORDER]
        if bad:
            raise ValueError(f"insert contains non-standard residues: {bad}")
        if insert_at is None:
            raise ValueError("`insert_at` is required when `insert` is given")
        seqs = [s[:insert_at] + insert + s[insert_at:] for s in seqs]

    keep = [s for s in seqs if len(s) > left_trim + right_trim]
    n_short = len(seqs) - len(keep)
    seqs = keep

    if not seqs:
        raise ValueError(
            f"No sequences left in {sequence_col!r} after cleaning "
            f"({n_raw} rows in, {n_nonstandard} non-standard, {n_short} too short)"
        )

    if left_trim or right_trim:
        seqs = [
            s[left_trim: len(s) - right_trim if right_trim > 0 else None]
            for s in seqs
        ]

    length_counts = pd.Series([len(s) for s in seqs]).value_counts().sort_index()

    if verbose:
        print(
            f"  {sequence_col}: {n_raw} rows -> {len(seqs)} sequences "
            f"(dropped {n_nonstandard} non-standard, {n_short} too short); "
            f"lengths {dict(length_counts)}"
        )

    if len(length_counts) != 1:
        raise ValueError(
            "Sequences are not all the same length after trimming: "
            f"{dict(length_counts)}. Trim the flanks (--left-trim/--right-trim) "
            "or filter to one length first."
        )

    return seqs


def aa_frequency_matrix(seqs):
    """20 x L DataFrame of amino-acid frequencies, columns numbered 1..L."""
    if not seqs:
        raise ValueError("No sequences given")

    L = len(seqs[0])
    counts = np.zeros((len(AA_ORDER), L))
    row_of = {aa: i for i, aa in enumerate(AA_ORDER)}

    for seq in seqs:
        for i, aa in enumerate(seq):
            counts[row_of[aa], i] += 1

    return pd.DataFrame(
        counts / len(seqs),
        index=AA_ORDER,
        columns=range(1, L + 1),
    )


def position_entropy(freq):
    """Shannon entropy (bits) of each column of a frequency matrix."""
    p = freq.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)

    entropy = -terms.sum(axis=0)
    entropy[entropy == 0] = 0.0          # avoid -0.0 for fully conserved positions

    return pd.Series(entropy, index=freq.columns, name="entropy_bits")


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

def compare_position_distributions(
    freq_a,
    freq_b,
    label_a="Set A",
    label_b="Set B",
    verbose=True,
):
    """Per-position divergence/agreement statistics between two frequency matrices."""
    common_positions = sorted(set(freq_a.columns).intersection(freq_b.columns))

    if not common_positions:
        raise ValueError("The two frequency matrices share no positions")

    ent_a = position_entropy(freq_a)
    ent_b = position_entropy(freq_b)

    results = []

    for pos in common_positions:
        p = freq_a[pos].to_numpy(dtype=float)
        q = freq_b[pos].to_numpy(dtype=float)

        js = float(jensenshannon(p, q, base=2) ** 2)      # divergence, in bits
        tv = float(0.5 * np.abs(p - q).sum())             # total variation

        if np.ptp(p) == 0 or np.ptp(q) == 0:
            rho, pval = np.nan, np.nan                    # constant vector: undefined
        else:
            rho, pval = spearmanr(p, q)

        abs_diff = np.abs(q - p)

        results.append(
            {
                "Position": pos,
                "top_a": freq_a[pos].idxmax(),
                "top_a_freq": float(freq_a[pos].max()),
                "top_b": freq_b[pos].idxmax(),
                "top_b_freq": float(freq_b[pos].max()),
                "Max_abs_diff": float(abs_diff.max()),
                "Mean_abs_diff": float(abs_diff.mean()),
                "Total_variation": tv,
                "JS_divergence": js,
                "Spearman_rho": float(rho),
                "Spearman_p_value": float(pval),
                "Entropy_a_bits": float(ent_a[pos]),
                "Entropy_b_bits": float(ent_b[pos]),
            }
        )

    out = pd.DataFrame(results).rename(
        columns={"top_a": f"Top_{label_a}", "top_b": f"Top_{label_b}"}
    )

    if verbose:
        print(f"\nComparing: {label_a} vs {label_b}")
        print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(
            f"\nMean JS divergence: {out['JS_divergence'].mean():.4f}"
            f"\nMax JS divergence:  {out['JS_divergence'].max():.4f}"
            f" (position {int(out.loc[out['JS_divergence'].idxmax(), 'Position'])})"
            f"\nMean total variation: {out['Total_variation'].mean():.4f}"
            f"\nMean Spearman rho: {out['Spearman_rho'].mean():.4f}"
            f"\nMin Spearman rho:  {out['Spearman_rho'].min():.4f}"
        )

    return out


# ------------------------------------------------------------
# FIGURES
# ------------------------------------------------------------

def _finish(fig, save=None, show=True, tight_rect=None):
    if tight_rect:
        fig.tight_layout(rect=tight_rect)
    else:
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


def plot_all_position_overlays(
    freq_1,
    freq_2,
    top_n=8,
    label_1="Set A",
    label_2="Set B",
    ncols=4,
    figsize=None,
    positions=None,
    save_png=None,
    show=True,
    title="Amino acid frequency overlay per position",
):
    """Grouped bar chart of the top_n residues at every position."""
    if positions is None:
        positions = list(freq_1.columns)

    npos = len(positions)
    ncols = max(1, min(ncols, npos))
    nrows = int(np.ceil(npos / ncols))

    if figsize is None:
        figsize = (4.5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.reshape(-1)

    for i, (ax, pos) in enumerate(zip(axes, positions)):
        vals = pd.DataFrame({label_1: freq_1[pos], label_2: freq_2[pos]})
        vals = (
            vals.assign(max_freq=vals.max(axis=1))
            .sort_values("max_freq", ascending=False)
            .head(top_n)
            .drop(columns="max_freq")
        )

        x = np.arange(len(vals.index))
        width = 0.38

        ax.bar(x - width / 2, vals[label_1], width, label=label_1)
        ax.bar(x + width / 2, vals[label_2], width, label=label_2)

        ax.set_title(f"Position {pos}")
        ax.set_xticks(x)
        ax.set_xticklabels(vals.index, rotation=0)
        ax.set_ylim(0, 1)

        if i % ncols == 0:
            ax.set_ylabel("Frequency")

    for ax in axes[npos:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, fontsize=16, y=0.98)

    return _finish(fig, save_png, show, tight_rect=[0, 0, 1, 0.94])


def plot_difference_heatmap(
    freq_a,
    freq_b,
    label_a="Set A",
    label_b="Set B",
    save_png=None,
    show=True,
    cmap="RdBu_r",
):
    """Heatmap of freq_b - freq_a over all residues and positions."""
    diff = freq_b - freq_a
    vmax = float(np.abs(diff.to_numpy()).max()) or 1.0

    fig, ax = plt.subplots(figsize=(1 + 0.55 * diff.shape[1], 7))
    im = ax.imshow(
        diff.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(diff.shape[1]))
    ax.set_xticklabels(diff.columns)
    ax.set_yticks(np.arange(len(diff.index)))
    ax.set_yticklabels(diff.index)
    ax.set_xlabel("Position")
    ax.set_ylabel("Residue")
    ax.set_title(f"Frequency difference: {label_b} - {label_a}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Δ frequency (+ = enriched in {label_b})")

    return _finish(fig, save_png, show)


def plot_js_per_position(
    stats,
    label_a="Set A",
    label_b="Set B",
    save_png=None,
    show=True,
):
    """Bar plot of the per-position Jensen-Shannon divergence."""
    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(stats), 4.5))

    ax.bar(stats["Position"].astype(str), stats["JS_divergence"], color="#4C72B0")
    ax.axhline(
        stats["JS_divergence"].mean(),
        ls="--",
        lw=1.5,
        color="grey",
        label=f"mean = {stats['JS_divergence'].mean():.3f}",
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("JS divergence (bits)")
    ax.set_ylim(0, max(1.0, float(stats["JS_divergence"].max()) * 1.1))
    ax.set_title(f"Per-position divergence: {label_a} vs {label_b}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    return _finish(fig, save_png, show)


# ------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------

def _slug(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def run_comparison(
    source_a,
    source_b,
    column="designed_sequence",
    column_a=None,
    column_b=None,
    label_a="Set A",
    label_b="Set B",
    left_trim_a=0,
    right_trim_a=0,
    left_trim_b=0,
    right_trim_b=0,
    insert_a=None,
    insert_at_a=None,
    insert_b=None,
    insert_at_b=None,
    query_a=None,
    query_b=None,
    plots=("overlay", "heatmap", "js"),
    top_n=20,
    ncols=4,
    sep=None,
    save_dir=None,
    show=True,
    stats_out=None,
    save_freqs=False,
):
    """Load both design tables, build frequency matrices, compare and plot."""
    df_a = source_a if isinstance(source_a, pd.DataFrame) else _read_table(source_a, sep)
    df_b = source_b if isinstance(source_b, pd.DataFrame) else _read_table(source_b, sep)

    if query_a:
        df_a = df_a.query(query_a)
        print(f"{label_a}: {len(df_a)} rows after query {query_a!r}")
    if query_b:
        df_b = df_b.query(query_b)
        print(f"{label_b}: {len(df_b)} rows after query {query_b!r}")

    print(f"\n=== {label_a} vs {label_b} ===")

    seqs_a = clean_and_trim(
        df_a, column_a or column, left_trim_a, right_trim_a, insert_a, insert_at_a
    )
    seqs_b = clean_and_trim(
        df_b, column_b or column, left_trim_b, right_trim_b, insert_b, insert_at_b
    )

    freq_a = aa_frequency_matrix(seqs_a)
    freq_b = aa_frequency_matrix(seqs_b)

    if freq_a.shape[1] != freq_b.shape[1]:
        raise ValueError(
            "Compared sequence regions are not the same length "
            f"({freq_a.shape[1]} vs {freq_b.shape[1]}). Adjust the trims."
        )

    stats = compare_position_distributions(freq_a, freq_b, label_a, label_b)

    save_dir = Path(save_dir) if save_dir else None
    stem = _slug(f"{label_a}_vs_{label_b}")

    def _path(kind):
        return save_dir / f"{stem}_{kind}.png" if save_dir else None

    plots = tuple(plots or ())

    if "overlay" in plots:
        plot_all_position_overlays(
            freq_a, freq_b,
            top_n=top_n, label_1=label_a, label_2=label_b, ncols=ncols,
            save_png=_path("overlay"), show=show,
        )

    if "heatmap" in plots:
        plot_difference_heatmap(
            freq_a, freq_b, label_a, label_b,
            save_png=_path("heatmap"), show=show,
        )

    if "js" in plots:
        plot_js_per_position(
            stats, label_a, label_b,
            save_png=_path("js"), show=show,
        )

    if stats_out:
        stats_out = Path(stats_out)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_out, index=False)
        print(f"stats -> {stats_out}")

    if save_freqs:
        target = save_dir or Path(".")
        target.mkdir(parents=True, exist_ok=True)
        for freq, label in ((freq_a, label_a), (freq_b, label_b)):
            path = target / f"{_slug(label)}_freq_matrix.csv"
            freq.to_csv(path)
            print(f"frequencies -> {path}")

    return {"freq_a": freq_a, "freq_b": freq_b, "stats": stats,
            "n_a": len(seqs_a), "n_b": len(seqs_b)}


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare per-position amino-acid frequencies between two "
                    "sets of designed sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--a", required=True, help="first design table (csv/tsv/parquet/xlsx)")
    p.add_argument("--b", required=True, help="second design table")

    p.add_argument("--column", default="designed_sequence",
                   help="sequence column in both tables")
    p.add_argument("--column-a", help="sequence column in --a (overrides --column)")
    p.add_argument("--column-b", help="sequence column in --b (overrides --column)")
    p.add_argument("--sep", help="delimiter override for text tables")

    p.add_argument("--label-a", default="Set A")
    p.add_argument("--label-b", default="Set B")

    p.add_argument("--left-trim-a", type=int, default=0)
    p.add_argument("--right-trim-a", type=int, default=0)
    p.add_argument("--left-trim-b", type=int, default=0)
    p.add_argument("--right-trim-b", type=int, default=0)

    p.add_argument("--insert-a", help="motif to splice into every --a sequence")
    p.add_argument("--insert-at-a", type=int, help="0-based index for --insert-a")
    p.add_argument("--insert-b", help="motif to splice into every --b sequence")
    p.add_argument("--insert-at-b", type=int, help="0-based index for --insert-b")

    p.add_argument("--query-a", help="pandas query applied to --a, "
                                     "e.g. 'affinity_pred_value < 0.1'")
    p.add_argument("--query-b", help="pandas query applied to --b")

    p.add_argument("--plots", nargs="*", default=["overlay", "heatmap", "js"],
                   choices=["overlay", "heatmap", "js", "none"],
                   help="figures to draw")
    p.add_argument("--top-n", type=int, default=20,
                   help="residues per position in the overlay plot")
    p.add_argument("--ncols", type=int, default=4,
                   help="overlay panels per row")

    p.add_argument("--save-dir", help="write PNGs here")
    p.add_argument("--no-show", action="store_true", help="do not open figure windows")
    p.add_argument("--stats-out", help="write the per-position statistics CSV here")
    p.add_argument("--save-freqs", action="store_true",
                   help="also write both frequency matrices as CSV")

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
        left_trim_a=args.left_trim_a,
        right_trim_a=args.right_trim_a,
        left_trim_b=args.left_trim_b,
        right_trim_b=args.right_trim_b,
        insert_a=args.insert_a,
        insert_at_a=args.insert_at_a,
        insert_b=args.insert_b,
        insert_at_b=args.insert_at_b,
        query_a=args.query_a,
        query_b=args.query_b,
        plots=plots,
        top_n=args.top_n,
        ncols=args.ncols,
        sep=args.sep,
        save_dir=args.save_dir,
        show=show,
        stats_out=args.stats_out,
        save_freqs=args.save_freqs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
