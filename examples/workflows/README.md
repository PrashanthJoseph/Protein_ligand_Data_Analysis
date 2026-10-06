# Analysis workflow examples

These compact examples are derived from the E2 antibody project. Run every
command from the repository root. Text outputs are committed under `expected/`;
commands write fresh results under ignored `output/` directories.

## 1. Unpaired score distributions

Compares representative run-level Boltz-2 affinity values with and without an
MSA. These are treated as distributions, not matched candidate pairs.

```bash
python distribution_comparison.py \
  --a examples/workflows/01_distribution_comparison/with_msa.csv \
  --b examples/workflows/01_distribution_comparison/without_msa.csv \
  --column-a estradiol_affinity_pred_value_ensemble \
  --column-b affinity_pred_value \
  --label-a "With MSA" --label-b "Without MSA" \
  --plots none --no-show \
  --stats-out examples/workflows/01_distribution_comparison/output/summary.json
```

Replace `--plots none` with `--plots kde ecdf hist` and add `--save-dir` to
generate the three figures.

## 2. Paired MSA score comparison

Compares the same 20 candidates scored with and without an MSA. Lower values
rank better, so no `--higher-is-better` flag is used.

```bash
python paired_score_comparison.py \
  --input examples/workflows/02_paired_score_comparison/input.csv \
  --col-a estradiol_affinity_pred_value_ensemble_mean \
  --col-b affinity_pred_value_mean --id-col sequence_id \
  --label-a "With MSA" --label-b "Without MSA" \
  --top-ks 5 10 15 --plots none --no-show \
  --summary-out examples/workflows/02_paired_score_comparison/output/summary.csv \
  --details-out examples/workflows/02_paired_score_comparison/output/details.csv
```

## 3. CDR-H3 amino-acid frequencies

Compares 20 full 15-residue designs with 20 12-residue conserved-library
designs. The three-residue left trim aligns the compared windows.

```bash
python aa_frequency_comparison.py \
  --a examples/workflows/03_aa_frequency_comparison/full_variable.csv \
  --b examples/workflows/03_aa_frequency_comparison/conserved.csv \
  --column designed_sequence --left-trim-a 3 \
  --label-a "Full variable" --label-b "Conserved" \
  --plots none --no-show --save-freqs \
  --save-dir examples/workflows/03_aa_frequency_comparison/output \
  --stats-out examples/workflows/03_aa_frequency_comparison/output/position_stats.csv
```

Replace `--plots none` with `--plots overlay heatmap js` to generate figures.

## 4. GNINA affinity screening

Uses 11 flexible-docking results spanning four poor, four intermediate and
three strong experimental binders. The optional GNINA `CNNscore` is included as
a higher-is-better probability-like output.

```bash
python -m affinity_screening \
  --config examples/workflows/04_affinity_screening/config.json
```

This creates classified data, cutoff metrics, a threshold sweep, five figures
and a self-contained HTML report. The committed expected directory contains the
three deterministic CSV tables; images and HTML are intentionally regenerated.

## Source provenance

- `e2pico_all_revertants_5reps.csv`
- `without_msa_affinity_results_all_runs.csv`
- `e2pico_msa_effect_comparison.csv`
- `Boltzgen/M3rd_workspace/all_designs_metrics_cdrh3_15top.csv`
- `Boltzgen/M3rd_workspace/all_designs_metrics_e2pico_scfvm3rd_cdrh3_conserved.csv`
- `Boltzgen/M3rd_workspace/GNINA/gnina_best_pose_per_receptor_flex.csv`
