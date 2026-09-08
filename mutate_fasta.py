# ============================================================
# Generalised combinatorial mutant generator
#
# Input  : a multi-entry FASTA. Each entry is ONE job, whose
#          sequence is chains joined by ":"  (chain1:chain2:...)
# Spec   : {chain: {position: residues}}
#            chain     -> 1-based chain index, or a name from CHAIN_NAMES
#            position  -> residue number (see NUMBERING; default 1-based)
#            residues  -> "A"        alanine scan
#                         "AGV"      or ["A","G","V"] -> try each
# Output : one FASTA per job + one manifest CSV for everything
# ============================================================

from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import pandas as pd

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


# ------------------------------------------------------------
# FASTA IO
# ------------------------------------------------------------

def read_fasta(path):
    """Return [(header, sequence), ...] preserving file order."""
    entries, name, chunks = [], None, []

    with open(path) as fh:
        for line in fh:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(chunks)))
                name, chunks = line[1:].strip(), []
            else:
                chunks.append(line.replace(" ", ""))

    if name is not None:
        entries.append((name, "".join(chunks)))

    if not entries:
        raise ValueError(f"No FASTA records found in {path}")

    return entries


# ------------------------------------------------------------
# SPEC RESOLUTION
# ------------------------------------------------------------

def _resolve_chain(key, n_chains, chain_names):
    """Map a spec key (index or name) onto a 0-based chain index."""
    if isinstance(key, int):
        if not 1 <= key <= n_chains:
            raise ValueError(
                f"Chain {key} out of range (job has {n_chains} chains)"
            )
        return key - 1

    if chain_names and key in chain_names:
        idx = chain_names.index(key)
        if idx >= n_chains:
            raise ValueError(
                f"Chain '{key}' is #{idx + 1} but job has {n_chains} chains"
            )
        return idx

    raise ValueError(
        f"Unknown chain key {key!r}. Use a 1-based index or one of {chain_names}"
    )


def _number_to_index(chain_seq, numbering):
    """
    Build {residue_number: 0-based index} for one chain.

    numbering: None  -> 1..len(seq)
               int   -> first residue of the chain has this number
               list  -> explicit numbering, len must match the chain
    """
    if numbering is None:
        return {i + 1: i for i in range(len(chain_seq))}

    if isinstance(numbering, int):
        return {numbering + i: i for i in range(len(chain_seq))}

    if len(numbering) != len(chain_seq):
        raise ValueError(
            f"Numbering length {len(numbering)} != chain length {len(chain_seq)}"
        )

    return {num: idx for idx, num in enumerate(numbering)}


def _as_residue_list(residues):
    if isinstance(residues, str):
        residues = list(residues)

    residues = [r.upper() for r in residues]

    bad = [r for r in residues if r not in AA20]
    if bad:
        raise ValueError(f"Not standard amino acids: {bad}")

    return residues


def build_sites(chains, spec, chain_names=None, numbering=None):
    """
    Flatten the spec into a list of mutable sites for one job.

    Each site: dict(chain_idx, chain_label, seq_idx, pos, wt, residues)
    """
    numbering = numbering or {}
    sites = []

    for chain_key, positions in spec.items():
        ci = _resolve_chain(chain_key, len(chains), chain_names)
        chain_seq = chains[ci]

        num_map = _number_to_index(
            chain_seq,
            numbering.get(chain_key, numbering.get(ci + 1)),
        )

        label = (
            chain_names[ci]
            if chain_names and ci < len(chain_names)
            else f"c{ci + 1}"
        )

        for pos, residues in positions.items():
            if pos not in num_map:
                raise ValueError(
                    f"Position {pos} not in numbering for chain {chain_key!r}"
                )

            seq_idx = num_map[pos]

            sites.append(
                dict(
                    chain_idx=ci,
                    chain_label=label,
                    seq_idx=seq_idx,
                    pos=pos,
                    wt=chain_seq[seq_idx],
                    residues=_as_residue_list(residues),
                )
            )

    sites.sort(key=lambda s: (s["chain_idx"], s["seq_idx"]))
    return sites


# ------------------------------------------------------------
# MUTATION
# ------------------------------------------------------------

def apply_mutations(chains, picks, multi_chain_labels):
    """
    picks: [(site, mutant_residue), ...]
    Returns (mutant_name, mutated_chains).
    """
    mutated = [list(c) for c in chains]
    labels = []

    for site, mut in picks:
        mutated[site["chain_idx"]][site["seq_idx"]] = mut

        tag = f"{site['wt']}{site['pos']}{mut}"
        labels.append(f"{site['chain_label']}-{tag}" if multi_chain_labels else tag)

    return "_".join(labels), ["".join(c) for c in mutated]


def generate_job_mutants(
    header,
    sequence,
    spec,
    chain_names=None,
    numbering=None,
    min_mutations=1,
    max_mutations=None,
    skip_silent=True,
    include_wt=False,
    name_prefix=None,
):
    """Yield mutant records for a single FASTA entry (one job)."""
    chains = sequence.split(":")
    sites = build_sites(chains, spec, chain_names, numbering)

    if not sites:
        raise ValueError(f"No mutable sites resolved for job {header!r}")

    max_mutations = min(max_mutations or len(sites), len(sites))
    multi_chain = len({s["chain_idx"] for s in sites}) > 1
    prefix = name_prefix or header.split()[0]

    records, seen = [], set()

    if include_wt:
        records.append(
            dict(
                job=header,
                mutant_name="WT",
                fasta_header=f"{prefix}_WT",
                num_mutations=0,
                mutations=(),
                chains=list(chains),
                sequence=":".join(chains),
            )
        )

    for k in range(min_mutations, max_mutations + 1):
        for combo in combinations(sites, k):
            for choice in product(*(s["residues"] for s in combo)):
                picks = list(zip(combo, choice))

                if skip_silent and any(s["wt"] == m for s, m in picks):
                    continue

                mutant_name, mutant_chains = apply_mutations(
                    chains, picks, multi_chain
                )

                mutant_seq = ":".join(mutant_chains)
                if mutant_seq in seen:
                    continue
                seen.add(mutant_seq)

                records.append(
                    dict(
                        job=header,
                        mutant_name=mutant_name,
                        fasta_header=f"{prefix}_{mutant_name}",
                        num_mutations=k,
                        mutations=tuple(
                            f"{s['chain_label']}:{s['wt']}{s['pos']}{m}"
                            for s, m in picks
                        ),
                        chains=mutant_chains,
                        sequence=mutant_seq,
                    )
                )

    return records


def generate_mutant_fastas(
    input_fasta,
    spec,
    out_dir,
    chain_names=None,
    numbering=None,
    min_mutations=1,
    max_mutations=None,
    skip_silent=True,
    include_wt=False,
    suffix="_muts.fasta",
    manifest="mutant_manifest.csv",
):
    """
    Run every entry of `input_fasta` through `spec`.
    One output FASTA per job; returns the combined DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for header, sequence in read_fasta(input_fasta):
        records = generate_job_mutants(
            header,
            sequence,
            spec,
            chain_names=chain_names,
            numbering=numbering,
            min_mutations=min_mutations,
            max_mutations=max_mutations,
            skip_silent=skip_silent,
            include_wt=include_wt,
        )

        job_id = header.split()[0]
        out_path = out_dir / f"{job_id}{suffix}"

        with open(out_path, "w") as fh:
            for rec in records:
                fh.write(f">{rec['fasta_header']}\n{rec['sequence']}\n")

        for rec in records:
            rec["fasta_path"] = str(out_path)

        all_records.extend(records)
        print(f"{job_id}: {len(records):5d} mutants -> {out_path}")

    df = pd.DataFrame(all_records)

    if manifest:
        manifest_path = out_dir / manifest
        df.drop(columns=["chains"]).to_csv(manifest_path, index=False)
        print(f"\nmanifest -> {manifest_path}")

    print(f"total mutants: {len(df)}")
    return df


# ------------------------------------------------------------
# EXAMPLE / CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    WORKSPACE = r"D:\INITO\myProjects\Estradiol\e2pico_antibodies\Boltzgen\M3rd_workspace"

    INPUT_FASTA = WORKSPACE + r"\e2pico_scfv_wt.fasta"     # chain1:chain2 per entry
    OUT_DIR = WORKSPACE + r"\cdrl3_ALA_muts"

    # chain order inside each ":"-joined job entry
    CHAIN_NAMES = ["VH", "VL"]

    # residue numbering per chain; omit a chain for plain 1-based numbering.
    #   int  -> number of the chain's first residue
    #   list -> explicit per-residue numbering
    NUMBERING = {}

    # {chain: {position: residues}}
    #   "A"              -> alanine scan
    #   "AGV" / [...]     -> try each substitution at that position
    SPEC = {
        "VL": {
            89: "A",     # V89
            90: "A",     # Q90
            91: "A",     # Y91
            96: "A",     # Y96
        },
    }

    df = generate_mutant_fastas(
        INPUT_FASTA,
        SPEC,
        OUT_DIR,
        chain_names=CHAIN_NAMES,
        numbering=NUMBERING,
        min_mutations=1,
        max_mutations=None,     # None -> all positions in the spec
        include_wt=True,
    )

    print(df[["job", "mutant_name", "num_mutations"]].head())
