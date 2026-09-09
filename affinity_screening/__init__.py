"""Affinity screening analysis.

Turn a table of predicted binding scores plus experimental Ka values into
correlation statistics, strong-versus-poor ROC discrimination, cutoff
trade-offs, figures and a self-contained HTML report.

    from affinity_screening import ScreeningConfig, run

    config = ScreeningConfig(
        input_path="gnina_best_pose_per_receptor_flex.csv",
        tool_name="GNINA_flex",
        sequence_col="receptor_id",
        affinity_col="minimizedAffinity",
        ka_col="Ka (10^9)",
        proposed_cutoff=-11.5,
        rescue_cutoff=-11.0,
        test_cutoffs=[-9, -10, -11, -11.5, -12],
        output_dir="score_cutoff_analysis_flex",
    )

    outcome = run(config)

Module map
    config.py    ScreeningConfig: columns, thresholds, cutoffs, paths
    data.py      loading, validation, cleaning, experimental classification
    metrics.py   correlations, ROC, per-cutoff performance, threshold sweep
    plots.py     the five figures
    report.py    the self-contained HTML report
    pipeline.py  orchestration and the console summary
    cli.py       argparse entry point
"""

from .config import HIGHER_IS_BETTER, LOWER_IS_BETTER, ScreeningConfig
from .data import AnalysisData, add_screening_decisions, load_table, prepare
from .metrics import (
    correlation_table,
    cutoff_metrics,
    cutoff_table,
    roc_analysis,
    roc_results,
    summarise_cutoffs,
    threshold_sweep,
)
from .pipeline import analyse, export_tables, print_summary, run
from .report import build_html, write_report

__version__ = "0.1.0"

__all__ = [
    "ScreeningConfig",
    "LOWER_IS_BETTER",
    "HIGHER_IS_BETTER",
    "AnalysisData",
    "prepare",
    "load_table",
    "add_screening_decisions",
    "cutoff_metrics",
    "cutoff_table",
    "summarise_cutoffs",
    "threshold_sweep",
    "correlation_table",
    "roc_analysis",
    "roc_results",
    "analyse",
    "export_tables",
    "print_summary",
    "run",
    "build_html",
    "write_report",
    "__version__",
]
