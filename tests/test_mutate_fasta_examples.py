import csv
import unittest
from pathlib import Path

from mutate_fasta import generate_job_mutants, load_spec_json, read_fasta


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "mutants"


def generate_example(name, *, max_mutations=None, include_wt=False):
    folder = EXAMPLES / name
    [(header, sequence)] = read_fasta(folder / "input.fasta")
    records = generate_job_mutants(
        header,
        sequence,
        load_spec_json(folder / "spec.json"),
        chain_names=["VL", "VH"],
        max_mutations=max_mutations,
        include_wt=include_wt,
    )
    return records


def expected_records(name, filename):
    return read_fasta(EXAMPLES / name / "expected" / filename)


class MutantExampleTests(unittest.TestCase):
    def assert_matches_expected(self, example_name, records, expected):
        observed = [(record["fasta_header"], record["sequence"]) for record in records]
        self.assertEqual(observed, expected)

        manifest_path = (
            EXAMPLES / example_name / "expected" / "mutant_manifest.csv"
        )
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        self.assertEqual(len(manifest), len(records))
        self.assertEqual(
            [(row["fasta_header"], row["sequence"]) for row in manifest],
            observed,
        )
        self.assertTrue(all("\\" not in row["fasta_path"] for row in manifest))

    def test_cdrl3_alanine_scan(self):
        records = generate_example("01_cdrl3_alanine")
        expected = expected_records(
            "01_cdrl3_alanine", "e2pico_scfvM3rd_muts.fasta"
        )
        self.assertEqual(len(records), 15)
        self.assert_matches_expected("01_cdrl3_alanine", records, expected)

        parent_vh = read_fasta(
            EXAMPLES / "01_cdrl3_alanine" / "input.fasta"
        )[0][1].split(":")[1]
        self.assertTrue(all(record["chains"][1] == parent_vh for record in records))

    def test_cdrh3_alanine_scan_through_triples(self):
        records = generate_example("02_cdrh3_alanine", max_mutations=3)
        expected = expected_records(
            "02_cdrh3_alanine", "e2pico_scfvM3rd_muts.fasta"
        )
        self.assertEqual(len(records), 25)
        self.assert_matches_expected("02_cdrh3_alanine", records, expected)
        self.assertEqual({record["num_mutations"] for record in records}, {1, 2, 3})

    def test_multichain_reversions_recover_literature_wt(self):
        records = generate_example("03_multichain_reversions", include_wt=True)
        expected = expected_records(
            "03_multichain_reversions", "e2pico_scfv_4mut_muts.fasta"
        )
        self.assertEqual(len(records), 16)
        self.assert_matches_expected("03_multichain_reversions", records, expected)

        [(_, literature_wt)] = read_fasta(
            EXAMPLES / "03_multichain_reversions" / "reference_literature_wt.fasta"
        )
        fourfold = next(record for record in records if record["num_mutations"] == 4)
        self.assertEqual(fourfold["sequence"], literature_wt)
        self.assertIn("VL-", fourfold["mutant_name"])
        self.assertIn("VH-", fourfold["mutant_name"])

    def test_include_wt_deduplicates_silent_choice(self):
        records = generate_job_mutants(
            "parent", "AC", {1: {1: "A"}}, include_wt=True, skip_silent=False
        )
        self.assertEqual([(record["mutant_name"], record["sequence"]) for record in records],
                         [("WT", "AC")])


if __name__ == "__main__":
    unittest.main()
