"""Orchestration: config in, analysis + figures + report + CSVs out."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from . import data as data_module
from . import metrics, plots, report


def display_columns(config, has_probability):
    columns = [config.sequence_col, config.affinity_col]

    if has_probability:
        columns.append(config.probability_col)

    columns += [config.ka_col, "experimental_class", "screening_decision"]

    return columns


def analyse(analysis, config):
    """Every number the report and the figures need. No files, no drawing."""
    frame = analysis.frame

    proposed = metrics.cutoff_metrics(frame, config.proposed_cutoff, config)
    table = metrics.cutoff_table(frame, config.test_cutoffs, config)

    columns = display_columns(config, analysis.has_probability)

    rejected = frame[~frame["passes_proposed_cutoff"]][columns]
    rejected = rejected.sort_values(config.ka_col, ascending=False)

    strong = frame[frame[config.ka_col] > config.strong_ka_threshold][columns]
    strong = strong.sort_values(config.ka_col, ascending=False)

    return {
        "counts": analysis.class_counts(config),
        "correlations": metrics.correlation_table(
            frame, config, analysis.has_probability
        ),
        "roc": metrics.roc_results(frame, config, analysis.has_probability),
        "cutoff_table": table,
        "cutoff_summary": (
            metrics.summarise_cutoffs(table) if not table.empty else pd.DataFrame()
        ),
        "sweep": metrics.threshold_sweep(frame, config),
        "proposed": proposed,
        "rejected_candidates": rejected,
        "strong_binders": strong,
    }


def export_tables(analysis, results, config):
    """Write the classified data, the cutoff metrics and the sweep as CSVs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tool = config.tool_name
    paths = {
        "classified_data": output_dir / f"{tool}_screening_classified_data.csv",
        "cutoff_metrics": output_dir / f"{tool}_cutoff_metrics.csv",
        "threshold_sweep": output_dir / f"{tool}_threshold_sweep.csv",
    }

    analysis.frame.to_csv(paths["classified_data"], index=False)
    results["cutoff_table"].to_csv(paths["cutoff_metrics"], index=False)
    results["sweep"].to_csv(paths["threshold_sweep"], index=False)

    return paths


def print_summary(analysis, results, config):
    """The console view of the analysis."""
    counts = results["counts"]
    proposed = results["proposed"]
    rule = "<" if config.lower_is_better else ">"

    def header(text):
        print("\n" + "=" * 70)
        print(text)
        print("=" * 70)

    print(f"Rows before cleaning: {analysis.n_before}")
    print(f"Rows used for affinity screening: {analysis.n_after}")
    print(f"Probability data available: {analysis.has_probability}")

    header("DATASET SUMMARY")
    print(f"Total variants:        {counts['total']}")
    print(f"Poor, Ka < {config.poor_ka_threshold:g}:          {counts['poor']}")
    print(
        f"Intermediate, {config.poor_ka_threshold:g}-{config.strong_ka_threshold:g}:  "
        f"{counts['intermediate']}"
    )
    print(f"Strong, Ka > {config.strong_ka_threshold:g}:        {counts['strong']}")

    header("CORRELATION ANALYSIS")
    correlations = results["correlations"]

    if correlations.empty:
        print("No correlations could be calculated.")
    else:
        for _, row in correlations.iterrows():
            value = row["value"]
            shown = "n/a" if pd.isna(value) else f"{value:.3f}"
            p_shown = "n/a" if pd.isna(row["p_value"]) else f"{row['p_value']:.4g}"
            print(f"{row['output']} - {row['statistic']}: {shown} (p = {p_shown}, "
                  f"n = {row['n']})")

    header("STRONG-VERSUS-POOR ROC ANALYSIS")

    for key, roc in results["roc"].items():
        if roc is None:
            print(f"{key}: skipped (needs both classes and variable scores)")
            continue

        print(
            f"{key}: AUC = {roc['auc']:.3f} over {roc['n_strong']} x {roc['n_poor']} "
            f"= {roc['n_pairs']} strong-poor pairs "
            f"({roc['concordant_pairs']:.1f} correctly ordered); "
            f"Youden-optimal cutoff = {roc['youden_cutoff']:.3g}"
        )

    if not results["cutoff_summary"].empty:
        header("CUTOFF COMPARISON")
        print(
            results["cutoff_summary"].to_string(
                index=False, float_format=lambda v: f"{v:.1f}"
            )
        )

    header(f"PROPOSED CUTOFF: {config.proposed_cutoff:g}  "
           f"(retain if score {rule} cutoff)")
    print(
        f"Total candidates eliminated: {proposed['rejected_n']}/{proposed['total_n']} "
        f"({proposed['total_eliminated_pct']:.1f}%)"
    )
    print(
        f"Poor binders eliminated:     {proposed['poor_rejected_n']}/"
        f"{proposed['poor_total']} ({proposed['poor_eliminated_pct']:.1f}%)"
    )
    print(
        f"Strong binders retained:     {proposed['strong_retained_n']}/"
        f"{proposed['strong_total']} ({proposed['strong_retained_pct']:.1f}%)"
    )
    print(f"Strong binders rejected:     {proposed['strong_rejected_n']}")
    print(
        f"Rejected candidates that are poor: "
        f"{proposed['rejected_pool_poor_pct']:.1f}%"
    )
    print(
        f"Retained candidates that are strong: "
        f"{proposed['retained_pool_strong_pct']:.1f}%"
    )


def run(
    config,
    source=None,
    make_plots=True,
    make_report=True,
    export=True,
    verbose=True,
):
    """
    Full analysis.

    Returns a dict with the cleaned data, every computed table, the figure
    paths and the report path.
    """
    config.validate()

    if not config.show:
        matplotlib.use("Agg", force=True)

    analysis = data_module.prepare(source, config)
    analysis.frame = data_module.add_screening_decisions(analysis.frame, config)

    results = analyse(analysis, config)

    if verbose:
        print_summary(analysis, results, config)

    figures = plots.make_all(analysis, results, config) if make_plots else {}

    exports = export_tables(analysis, results, config) if export else {}

    report_path = (
        report.write_report(analysis, results, figures, config)
        if make_report
        else None
    )

    if verbose:
        print("\n" + "=" * 70)
        print("OUTPUT FILES")
        print("=" * 70)

        for path in list(figures.values()) + list(exports.values()):
            print(path)

        if report_path:
            print(report_path)

    return {
        "config": config,
        "analysis": analysis,
        "results": results,
        "figures": figures,
        "exports": exports,
        "report": report_path,
    }
