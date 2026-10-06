import contextlib
import io
import json
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from aa_frequency_comparison import run_comparison as run_aa_comparison
from distribution_comparison import run_comparison as run_distribution_comparison
from paired_score_comparison import compare_paired_scores

try:
    from affinity_screening import ScreeningConfig, run as run_affinity_screening
except ModuleNotFoundError as exc:
    if exc.name not in {"sklearn", "seaborn"}:
        raise
    ScreeningConfig = None
    run_affinity_screening = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "workflows"


class WorkflowExampleTests(unittest.TestCase):
    def test_distribution_summary(self):
        folder = EXAMPLES / "01_distribution_comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            observed = run_distribution_comparison(
                folder / "with_msa.csv",
                folder / "without_msa.csv",
                column_a="estradiol_affinity_pred_value_ensemble",
                column_b="affinity_pred_value",
                label_a="With MSA",
                label_b="Without MSA",
                plots=(),
                show=False,
            )
        expected = json.loads((folder / "expected" / "summary.json").read_text())
        self.assertEqual(observed, expected)

    def test_paired_summary_and_candidate_details(self):
        folder = EXAMPLES / "02_paired_score_comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            summary, details = compare_paired_scores(
                folder / "input.csv",
                col_a="estradiol_affinity_pred_value_ensemble_mean",
                col_b="affinity_pred_value_mean",
                id_col="sequence_id",
                label_a="With MSA",
                label_b="Without MSA",
                top_ks=(5, 10, 15),
                plot=False,
                show=False,
                verbose=False,
            )
        expected_summary = pd.read_csv(folder / "expected" / "summary.csv")
        expected_details = pd.read_csv(folder / "expected" / "details.csv")
        assert_frame_equal(summary, expected_summary, check_dtype=False, atol=1e-12)
        assert_frame_equal(
            details.sort_values("absolute_rank_change", ascending=False).reset_index(drop=True),
            expected_details,
            check_dtype=False,
            atol=1e-12,
        )

    def test_amino_acid_frequency_tables(self):
        folder = EXAMPLES / "03_aa_frequency_comparison"
        with contextlib.redirect_stdout(io.StringIO()):
            observed = run_aa_comparison(
                folder / "full_variable.csv",
                folder / "conserved.csv",
                column="designed_sequence",
                left_trim_a=3,
                label_a="Full variable",
                label_b="Conserved",
                plots=(),
                show=False,
            )
        expected_stats = pd.read_csv(folder / "expected" / "position_stats.csv")
        expected_a = pd.read_csv(folder / "expected" / "Full_variable_freq_matrix.csv", index_col=0)
        expected_b = pd.read_csv(folder / "expected" / "Conserved_freq_matrix.csv", index_col=0)
        expected_a.columns = expected_a.columns.astype(int)
        expected_b.columns = expected_b.columns.astype(int)

        assert_frame_equal(observed["stats"], expected_stats, check_dtype=False, atol=1e-12)
        assert_frame_equal(observed["freq_a"], expected_a, check_dtype=False, atol=1e-12)
        assert_frame_equal(observed["freq_b"], expected_b, check_dtype=False, atol=1e-12)

    @unittest.skipIf(ScreeningConfig is None, "affinity-screening extras not installed")
    def test_affinity_screening_tables(self):
        folder = EXAMPLES / "04_affinity_screening"
        config = ScreeningConfig.from_json(folder / "config.json")
        with contextlib.redirect_stdout(io.StringIO()):
            outcome = run_affinity_screening(
                config,
                make_plots=False,
                make_report=False,
                export=False,
                verbose=False,
            )

        expected_dir = folder / "expected"
        expected_classified = pd.read_csv(
            expected_dir / "GNINA_flex_example_screening_classified_data.csv"
        )
        observed_classified = outcome["analysis"].frame.copy()
        observed_classified["experimental_class"] = observed_classified[
            "experimental_class"
        ].astype(str)
        assert_frame_equal(
            observed_classified,
            expected_classified,
            check_dtype=False,
            atol=1e-12,
        )

        expected_cutoffs = pd.read_csv(
            expected_dir / "GNINA_flex_example_cutoff_metrics.csv"
        )
        expected_sweep = pd.read_csv(
            expected_dir / "GNINA_flex_example_threshold_sweep.csv"
        )
        assert_frame_equal(
            outcome["results"]["cutoff_table"],
            expected_cutoffs,
            check_dtype=False,
            atol=1e-12,
        )
        assert_frame_equal(
            outcome["results"]["sweep"],
            expected_sweep,
            check_dtype=False,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
