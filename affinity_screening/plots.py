"""Figures for the screening analysis.

Every function takes already-computed data plus a `ScreeningConfig`, writes one
PNG and returns its path (or None when the figure does not apply).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLOR_STRONG = "#238B57"
COLOR_POOR = "#D95F5F"
COLOR_TOTAL = "#224B7A"
COLOR_PROBABILITY = "#B87818"
COLOR_GRID = "#D9DEE3"


def apply_style():
    """Shared look for every figure in the report."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
        }
    )


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def short_name(sequence_id, config):
    """Shorten a sequence id for plot labels using config.label_strip."""
    text = str(sequence_id)

    for fragment in config.label_strip:
        text = text.replace(fragment, "")

    return text


def _save(fig, path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")

    if config.show:
        plt.show()
    else:
        plt.close(fig)

    return path


# ------------------------------------------------------------
# 1. predicted score versus experimental Ka
# ------------------------------------------------------------

def plot_score_vs_ka(frame, config, path):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    sns.scatterplot(
        data=frame,
        x=config.affinity_col,
        y=config.ka_col,
        hue="experimental_class",
        hue_order=config.class_order,
        palette=config.palette,
        s=90,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    if (frame[config.ka_col] > 0).all():
        ax.set_yscale("log")

    ax.axvline(
        config.proposed_cutoff,
        color=COLOR_TOTAL,
        linestyle="--",
        linewidth=1.8,
        label=f"Proposed cutoff ({config.proposed_cutoff:g})",
    )

    if config.rescue_cutoff is not None:
        ax.axvline(
            config.rescue_cutoff,
            color=COLOR_TOTAL,
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label=f"Rescue cutoff ({config.rescue_cutoff:g})",
        )

    ax.axhline(config.poor_ka_threshold, color="#B33A3A", linestyle=":", linewidth=1.4)
    ax.axhline(config.strong_ka_threshold, color=COLOR_STRONG, linestyle=":", linewidth=1.4)

    strong = frame[frame[config.ka_col] > config.strong_ka_threshold]

    for _, row in strong.iterrows():
        ax.annotate(
            short_name(row[config.sequence_col], config),
            (row[config.affinity_col], row[config.ka_col]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="#183B2B",
        )

    ax.set_xlabel(
        f"{config.tool_name} affinity predicted value\n({config.score_hint})"
    )
    ax.set_ylabel(r"Experimental $K_a$")
    ax.set_title(f"Experimental affinity versus {config.tool_name} score")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=None, frameon=False, loc="lower left")

    clean_axes(ax)
    fig.tight_layout()

    return _save(fig, path, config)


# ------------------------------------------------------------
# 2. trade-off across the whole cutoff range
# ------------------------------------------------------------

def plot_cutoff_tradeoff(sweep, proposed, config, path):
    fig, ax = plt.subplots(figsize=(10, 6))

    series = [
        ("strong_retained_pct", COLOR_STRONG, 2.7,
         f"Strong binders retained (Ka > {config.strong_ka_threshold:g})"),
        ("poor_eliminated_pct", COLOR_POOR, 2.7,
         f"Poor binders eliminated (Ka < {config.poor_ka_threshold:g})"),
        ("total_eliminated_pct", COLOR_TOTAL, 2.3,
         "Total candidate pool eliminated"),
    ]

    for column, color, width, label in series:
        ax.plot(sweep["cutoff"], sweep[column], color=color, linewidth=width, label=label)

    for cutoff in config.test_cutoffs:
        ax.axvline(cutoff, color="#70777F", linestyle="--", linewidth=1, alpha=0.8)
        ax.text(
            cutoff, 3, f"{cutoff:g}",
            rotation=90, va="bottom", ha="right", fontsize=9, color="#50575E",
        )

    for column, color, _, _ in series:
        ax.scatter([config.proposed_cutoff], [proposed[column]],
                   color=color, s=75, zorder=5)

    ax.set_ylim(0, 104)
    ax.set_xlabel(f"Retention cutoff\n{config.retention_rule}")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Screening trade-off across {config.tool_name} cutoffs")
    ax.legend(frameon=False, loc="center right")

    clean_axes(ax)
    fig.tight_layout()

    return _save(fig, path, config)


# ------------------------------------------------------------
# 3. score distributions by experimental class
# ------------------------------------------------------------

def plot_class_distributions(frame, config, has_probability, path):
    n_panels = 2 if has_probability else 1
    figsize = (13, 5.5) if has_probability else (7, 5.5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes.reshape(-1)

    panels = [(config.affinity_col, "Affinity predicted value",
               f"{config.tool_name} affinity predicted value", frame)]

    if has_probability:
        panels.append(
            (config.probability_col, "Binary affinity probability",
             f"{config.tool_name} binary affinity probability",
             frame.dropna(subset=[config.probability_col]))
        )

    for ax, (column, title, ylabel, data) in zip(axes, panels):
        sns.boxplot(
            data=data,
            x="experimental_class",
            y=column,
            order=config.class_order,
            hue="experimental_class",
            hue_order=config.class_order,
            palette=config.palette,
            width=0.55,
            showfliers=False,
            legend=False,
            ax=ax,
        )

        sns.stripplot(
            data=data,
            x="experimental_class",
            y=column,
            order=config.class_order,
            color="#222222",
            size=5,
            jitter=0.18,
            alpha=0.75,
            ax=ax,
        )

        if column == config.affinity_col:
            ax.axhline(config.proposed_cutoff, color=COLOR_TOTAL,
                       linestyle="--", linewidth=1.5)

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        clean_axes(ax)

    fig.suptitle(
        f"{config.tool_name} outputs by experimental affinity class",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()

    return _save(fig, path, config)


# ------------------------------------------------------------
# 4. strong-versus-poor ROC
# ------------------------------------------------------------

def plot_roc(roc, config, path):
    affinity = roc.get("affinity")

    if affinity is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        affinity["fpr"], affinity["tpr"],
        color=COLOR_TOTAL, linewidth=2.7,
        label=f"Affinity predicted value (AUC = {affinity['auc']:.3f})",
    )

    probability = roc.get("probability")
    if probability is not None:
        ax.plot(
            probability["fpr"], probability["tpr"],
            color=COLOR_PROBABILITY, linewidth=2.7,
            label=f"Binary probability (AUC = {probability['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="#8B9299", linestyle="--", linewidth=1.3,
            label="Random ranking (AUC = 0.500)")

    ax.scatter(
        [affinity["youden_fpr"]], [affinity["youden_tpr"]],
        color=COLOR_TOTAL, s=70, zorder=5,
        label=f"Youden-optimal cutoff ({affinity['youden_cutoff']:.3g})",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate\n(fraction of poor binders retained)")
    ax.set_ylabel("True-positive rate\n(fraction of strong binders retained)")
    ax.set_title("Strong-versus-poor binder discrimination")
    ax.legend(frameon=False, loc="lower right")

    clean_axes(ax)
    fig.tight_layout()

    return _save(fig, path, config)


# ------------------------------------------------------------
# 5. selected cutoffs side by side
# ------------------------------------------------------------

def plot_cutoff_comparison(table, config, path):
    if table.empty:
        return None

    x = np.arange(len(table))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    containers = [
        ax.bar(x - width, table["strong_retained_pct"], width,
               color=COLOR_STRONG, label="Strong retained"),
        ax.bar(x, table["poor_eliminated_pct"], width,
               color=COLOR_POOR, label="Poor eliminated"),
        ax.bar(x + width, table["total_eliminated_pct"], width,
               color=COLOR_TOTAL, label="Total pool eliminated"),
    ]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:g}" for c in table["cutoff"]])
    ax.set_ylim(0, 112)
    ax.set_xlabel(f"{config.tool_name} retention cutoff")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Performance of selected {config.tool_name} cutoffs")
    ax.legend(frameon=False, loc="upper right")

    for container in containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    clean_axes(ax)
    fig.tight_layout()

    return _save(fig, path, config)


# ------------------------------------------------------------
# everything, in report order
# ------------------------------------------------------------

def make_all(analysis, results, config):
    """Draw every applicable figure; returns {key: path}."""
    apply_style()

    tool = config.tool_name
    figures = {}

    figures["scatter"] = plot_score_vs_ka(
        analysis.frame, config, config.output_path(f"01_{tool}_score_vs_Ka.png")
    )

    figures["tradeoff"] = plot_cutoff_tradeoff(
        results["sweep"], results["proposed"], config,
        config.output_path(f"02_{tool}_cutoff_tradeoff.png"),
    )

    figures["distributions"] = plot_class_distributions(
        analysis.frame, config, analysis.has_probability,
        config.output_path(f"03_{tool}_class_distributions.png"),
    )

    figures["roc"] = plot_roc(
        results["roc"], config, config.output_path(f"04_{tool}_ROC_strong_vs_poor.png")
    )

    figures["cutoff_comparison"] = plot_cutoff_comparison(
        results["cutoff_table"], config,
        config.output_path(f"05_{tool}_cutoff_comparison.png"),
    )

    return {key: value for key, value in figures.items() if value is not None}
