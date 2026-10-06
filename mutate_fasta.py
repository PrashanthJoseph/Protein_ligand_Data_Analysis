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

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import pandas as pd

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

RECORD_COLUMNS = [
    "job",
    "mutant_name",
    "fasta_header",
    "num_mutations",
    "mutations",
    "chains",
    "sequence",
    "fasta_path",
]


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

    if min_mutations < 0:
        raise ValueError("min_mutations must be at least 0")
    if max_mutations is not None and max_mutations < 0:
        raise ValueError("max_mutations must be at least 0 or None")

    max_mutations = min(
        len(sites) if max_mutations is None else max_mutations,
        len(sites),
    )
    if min_mutations > max_mutations:
        raise ValueError(
            f"min_mutations ({min_mutations}) exceeds max_mutations "
            f"({max_mutations})"
        )

    multi_chain = len({s["chain_idx"] for s in sites}) > 1
    prefix = name_prefix or header.split()[0]

    records, seen = [], set()

    if include_wt:
        seen.add(":".join(chains))
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
            rec["fasta_path"] = out_path.as_posix()

        all_records.extend(records)
        print(f"{job_id}: {len(records):5d} mutants -> {out_path}")

    df = pd.DataFrame(all_records, columns=RECORD_COLUMNS)

    if manifest:
        manifest_path = out_dir / manifest
        df.drop(columns=["chains"]).to_csv(manifest_path, index=False)
        print(f"\nmanifest -> {manifest_path}")

    print(f"total mutants: {len(df)}")
    return df


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def load_spec_json(path):
    """Load a JSON mutation spec, converting numeric object keys to integers."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mutation spec must be a JSON object")

    spec = {}
    for chain, positions in payload.items():
        if not isinstance(positions, dict):
            raise ValueError(f"Mutation spec for chain {chain!r} must be an object")

        chain_key = int(chain) if str(chain).isdigit() else chain
        spec[chain_key] = {
            int(position) if str(position).isdigit() else position: residues
            for position, residues in positions.items()
        }

    return spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate combinatorial point-mutant FASTAs from a multi-chain FASTA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_fasta", help="input FASTA; join chains with ':'")
    parser.add_argument("spec_json", help="JSON mutation specification")
    parser.add_argument("out_dir", help="output directory")
    parser.add_argument(
        "--chain-names",
        nargs="+",
        help="chain labels in the same order as the ':'-joined input chains",
    )
    parser.add_argument("--min-mutations", type=int, default=1)
    parser.add_argument("--max-mutations", type=int)
    parser.add_argument(
        "--include-wt",
        action="store_true",
        help="include the unchanged input parent",
    )
    parser.add_argument(
        "--keep-silent",
        action="store_true",
        help="keep choices equal to the parent residue",
    )
    parser.add_argument("--suffix", default="_muts.fasta")
    parser.add_argument("--manifest", default="mutant_manifest.csv")
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not write a combined manifest CSV",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = None if args.no_manifest else args.manifest

    generate_mutant_fastas(
        args.input_fasta,
        load_spec_json(args.spec_json),
        args.out_dir,
        chain_names=args.chain_names,
        min_mutations=args.min_mutations,
        max_mutations=args.max_mutations,
        skip_silent=not args.keep_silent,
        include_wt=args.include_wt,
        suffix=args.suffix,
        manifest=manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
