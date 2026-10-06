# Mutant-generation examples

These examples use antibody sequences and mutation panels from the E2 antibody
design project. In every input FASTA, the chain order is **VL:VH**, so the CLI
must receive `--chain-names VL VH` in that same order.

Run the commands below from the repository root. Each command writes one FASTA
plus `mutant_manifest.csv` to its `output` directory. The committed `expected`
directory shows the corresponding output from the current implementation.

## 1. M3rd CDR-L3 alanine scan

Mutates VL V89, Q90, Y91 and Y96 to alanine over every mutation order. Four
mutable sites give 4 singles + 6 doubles + 4 triples + 1 quadruple = 15 mutants.

```bash
python mutate_fasta.py \
  examples/mutants/01_cdrl3_alanine/input.fasta \
  examples/mutants/01_cdrl3_alanine/spec.json \
  examples/mutants/01_cdrl3_alanine/output \
  --chain-names VL VH
```

## 2. M3rd CDR-H3 alanine scan through triples

Mutates VH E99, I102, I110, Q111 and D112 to alanine, limiting combinations
to mutation orders 1-3. Five sites give 5 singles + 10 doubles + 10 triples =
25 mutants.

```bash
python mutate_fasta.py \
  examples/mutants/02_cdrh3_alanine/input.fasta \
  examples/mutants/02_cdrh3_alanine/spec.json \
  examples/mutants/02_cdrh3_alanine/output \
  --chain-names VL VH --max-mutations 3
```

## 3. Four-site, two-chain reversion panel

Starts from the four-mutation construct and tries every combination of four
literature-WT reversions: VL V29I, VL M36L, VL G77S and VH Q111L. The unchanged
four-mutation parent is included, followed by 15 reversion combinations. The
fourfold child recovers the supplied literature-WT sequence.

```bash
python mutate_fasta.py \
  examples/mutants/03_multichain_reversions/input.fasta \
  examples/mutants/03_multichain_reversions/spec.json \
  examples/mutants/03_multichain_reversions/output \
  --chain-names VL VH --include-wt
```

## Source provenance

The example inputs and panels were distilled from these project files:

- `Boltzgen/M3rd_workspace/e2pico_scfvM3rd.fa`
- `Boltzgen/M3rd_workspace/e2pico_scfvM3rd_cdrl3_ALA_muts`
- `Boltzgen/M3rd_workspace/e2pico_scfvM3rd_cdrh3_ALA_muts.fa`
- `Boltzgen/M3rd_workspace/e2pico_4mut_revertants.fa`

Only the small sequence fixtures needed to reproduce mutation generation are
included here; prediction scores and structure outputs are not required.
