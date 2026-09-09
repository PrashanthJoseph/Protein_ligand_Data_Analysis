"""Command-line entry point: ``affinity-screening`` / ``python -m affinity_screening``."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from .config import HIGHER_IS_BETTER, LOWER_IS_BETTER, ScreeningConfig
from .pipeline import run


def build_parser():
    p = argparse.ArgumentParser(
        prog="affinity-screening",
        description="Score-versus-experiment screening analysis: correlations, "
                    "strong-vs-poor ROC, cutoff trade-offs, figures and an HTML report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", help="JSON config file; flags below override its values")
    p.add_argument("--save-config", help="write the resolved config here and continue")

    p.add_argument("--input", help="input table (csv/tsv/parquet/xlsx)")
    p.add_argument("--sep", help="delimiter override for text tables")
    p.add_argument("--tool-name", help="name used in titles and output filenames")

    p.add_argument("--sequence-col", help="identifier column")
    p.add_argument("--affinity-col", help="predicted affinity/score column")
    p.add_argument("--ka-col", help="experimental Ka column")
    p.add_argument("--probability-col", help="optional binary-probability column")

    p.add_argument(
        "--score-direction",
        choices=[LOWER_IS_BETTER, HIGHER_IS_BETTER],
        help="whether a lower or higher predicted score means a better binder",
    )
    p.add_argument(
        "--probability-direction",
        choices=[LOWER_IS_BETTER, HIGHER_IS_BETTER],
        help="direction of the optional probability column",
    )

    p.add_argument("--poor-ka", type=float, dest="poor_ka_threshold",
                   help="Ka below this is a poor binder")
    p.add_argument("--strong-ka", type=float, dest="strong_ka_threshold",
                   help="Ka above this is a strong binder")

    p.add_argument("--proposed-cutoff", type=float, help="the cutoff being proposed")
    p.add_argument("--rescue-cutoff", type=float,
                   help="second, looser cutoff defining the rescue band")
    p.add_argument("--test-cutoffs", type=float, nargs="*",
                   help="cutoffs compared side by side")

    p.add_argument("--output-dir", help="directory for figures, CSVs and the report")
    p.add_argument("--sweep-points", type=int, help="resolution of the threshold sweep")
    p.add_argument("--dpi", type=int, help="figure resolution")
    p.add_argument("--label-strip", nargs="*",
                   help="substrings removed from sequence ids in plot labels")

    p.add_argument("--show", action="store_true",
                   help="open figure windows instead of writing them only")
    p.add_argument("--no-plots", action="store_true", help="skip all figures")
    p.add_argument("--no-report", action="store_true", help="skip the HTML report")
    p.add_argument("--no-export", action="store_true", help="skip the CSV exports")
    p.add_argument("--quiet", action="store_true", help="suppress the console summary")

    return p


def config_from_args(args):
    """Start from the JSON config (if any) and apply every flag the user set."""
    config = ScreeningConfig.from_json(args.config) if args.config else ScreeningConfig()

    overrides = {
        "input_path": args.input,
        "sep": args.sep,
        "tool_name": args.tool_name,
        "sequence_col": args.sequence_col,
        "affinity_col": args.affinity_col,
        "ka_col": args.ka_col,
        "probability_col": args.probability_col,
        "score_direction": args.score_direction,
        "probability_direction": args.probability_direction,
        "poor_ka_threshold": args.poor_ka_threshold,
        "strong_ka_threshold": args.strong_ka_threshold,
        "proposed_cutoff": args.proposed_cutoff,
        "rescue_cutoff": args.rescue_cutoff,
        "test_cutoffs": args.test_cutoffs,
        "output_dir": args.output_dir,
        "sweep_points": args.sweep_points,
        "dpi": args.dpi,
        "label_strip": args.label_strip,
        "show": True if args.show else None,
    }

    applied = {key: value for key, value in overrides.items() if value is not None}

    return config.replace(**applied)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    if config.input_path is None:
        build_parser().error("no input table: pass --input or set input_path in --config")

    if args.save_config:
        config.to_json(args.save_config)
        print(f"config -> {args.save_config}")

    run(
        config,
        make_plots=not args.no_plots,
        make_report=not args.no_report,
        export=not args.no_export,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
