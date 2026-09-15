"""
select_dataset.py — Deterministic, reproducible selection of the 13-protein baseline set.

Design decisions (confirmed with user):
  * 10 TRAINING proteins drawn from data/training/allosteric_training.csv (allobench+asd).
      - dedup by UniProt (one representative crystal per protein)
      - filter: single encoded chain has enough ground-truth, has_active, sequence present
      - bin by SEQUENCE LENGTH into 10 quantile bins; pick ONE per bin (fixed seed)
        -> spans a length range so we can later study length vs compute/accuracy.
      - apo input = ligand-stripped holo (handled downstream in encoding).
  * 3 VALIDATION proteins = challenge blind targets with TRUE apo structures
      (KRAS G12C 4OBE, BCR-ABL1 1OPL, Cardiac myosin 5TBY). c-Myc excluded (IDP, no pocket).

Selection is a pure function of (input CSVs, SEED, thresholds) => fully reproducible.
Outputs:
  data/dataset_manifest.csv            the 13-protein list (the reproducibility artifact)
  data/dataset_manifest.provenance.json   seed, thresholds, bin edges, input hashes

Run:  python scripts/select_dataset.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO / "data" / "training" / "allosteric_training.csv"
VAL_CSV = REPO / "data" / "validation_data.csv"
OUT_CSV = REPO / "data" / "dataset_manifest.csv"
OUT_PROV = REPO / "data" / "dataset_manifest.provenance.json"

# --- reproducibility knobs ---------------------------------------------------- #
SEED = 42
N_TRAIN = 10                 # number of training proteins (= number of length bins)
MIN_ALLOSTERIC_CHAIN = 4     # need >= this many ground-truth AFRs on the encoded chain
MIN_ACTIVE = 3               # need >= this many active-site residues (source set S)
VAL_NAMES = ["KRAS_G12C", "BCR-ABL1", "Cardiac_myosin"]   # c-Myc excluded (IDP)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_chain_residues(field_val: str, chain: str) -> list[int]:
    """'A:14;A:24;B:32' , chain='A' -> [14, 24]  (keep only the encoded chain)."""
    out: list[int] = []
    if not isinstance(field_val, str):
        return out
    for tok in field_val.split(";"):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        ch, num = tok.split(":", 1)
        if ch.strip() == chain:
            try:
                out.append(int(num.strip()))
            except ValueError:
                pass
    return sorted(set(out))


def count_active(field_val: str) -> int:
    if not isinstance(field_val, str) or not field_val.strip():
        return 0
    return len([t for t in field_val.replace(",", ";").split(";") if t.strip()])


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, dtype=str).fillna("")
    df["seq_len"] = df["sequence"].str.len()
    df["allo_chain"] = [
        parse_chain_residues(r.allosteric_residues, r.chain) for r in df.itertuples()
    ]
    df["n_allo_chain"] = df["allo_chain"].apply(len)
    df["n_active_i"] = df["active_residues_uniprot"].apply(count_active)
    return df


def filter_pool(df: pd.DataFrame) -> pd.DataFrame:
    ok = (
        (df["seq_len"] > 0)
        & (df["n_allo_chain"] >= MIN_ALLOSTERIC_CHAIN)
        & (df["n_active_i"] >= MIN_ACTIVE)
        & (df["has_active"].astype(str).str.strip() == "1")
        & (df["uniprot"].str.strip() != "")
    )
    return df[ok].copy()


def dedup_by_uniprot(df: pd.DataFrame) -> pd.DataFrame:
    """One representative per UniProt: max ground-truth coverage on the encoded chain,
    tie-break deterministically by (n_active desc, pdb id asc)."""
    df = df.sort_values(
        by=["uniprot", "n_allo_chain", "n_active_i", "pdb"],
        ascending=[True, False, False, True],
    )
    return df.groupby("uniprot", as_index=False).first()


def pick_length_bins(df: pd.DataFrame, n_bins: int, seed: int) -> pd.DataFrame:
    """Quantile-bin by seq_len; pick one entry per bin with a fixed RNG."""
    rng = np.random.default_rng(seed)
    # quantile edges -> n_bins bins with ~equal membership
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(df["seq_len"].to_numpy(), qs))
    labels = list(range(len(edges) - 1))
    df = df.copy()
    df["length_bin"] = pd.cut(
        df["seq_len"], bins=edges, labels=labels, include_lowest=True, duplicates="drop"
    )
    picks = []
    for b in sorted(df["length_bin"].dropna().unique()):
        cand = df[df["length_bin"] == b].sort_values("pdb")  # deterministic order
        idx = int(rng.integers(0, len(cand)))
        picks.append(cand.iloc[idx])
    out = pd.DataFrame(picks)
    return out, edges


def build_val() -> pd.DataFrame:
    df = pd.read_csv(VAL_CSV, dtype=str).fillna("")
    df = df[df["entry_id"].isin(VAL_NAMES)].copy()
    df["seq_len"] = df["sequence"].str.len()
    df["allo_chain"] = [
        parse_chain_residues(r.allosteric_residues, r.chain) for r in df.itertuples()
    ]
    df["n_allo_chain"] = df["allo_chain"].apply(len)
    df["n_active_i"] = df["active_residues_uniprot"].apply(count_active)
    df["length_bin"] = -1
    return df


def to_manifest(df: pd.DataFrame, split: str) -> pd.DataFrame:
    cols = {
        "split": split,
        "entry_id": df["entry_id"],
        "uniprot": df["uniprot"],
        "pdb": df["pdb"],
        "chain": df["chain"],
        "seq_len": df["seq_len"],
        "n_allosteric_chain": df["n_allo_chain"],
        "n_active": df["n_active_i"],
        "length_bin": df["length_bin"],
        "source": df["sources"],
        "modulator_alias": df["modulator_alias"],
        "allosteric_residues_chain": [";".join(map(str, x)) for x in df["allo_chain"]],
        "active_residues_uniprot": df["active_residues_uniprot"],
        "sequence": df["sequence"],
    }
    return pd.DataFrame(cols)


def main() -> None:
    pool = filter_pool(load_training())
    uniq = dedup_by_uniprot(pool)
    picks, edges = pick_length_bins(uniq, N_TRAIN, SEED)

    train_m = to_manifest(picks, "train").sort_values("seq_len").reset_index(drop=True)
    val_m = to_manifest(build_val(), "val").sort_values("seq_len").reset_index(drop=True)
    manifest = pd.concat([train_m, val_m], ignore_index=True)
    manifest.to_csv(OUT_CSV, index=False)

    prov = {
        "seed": SEED,
        "n_train": N_TRAIN,
        "thresholds": {
            "min_allosteric_chain": MIN_ALLOSTERIC_CHAIN,
            "min_active": MIN_ACTIVE,
        },
        "binning": {"by": "seq_len", "scheme": "quantile", "edges": edges.tolist()},
        "val_names": VAL_NAMES,
        "pool_sizes": {
            "raw": int(len(load_training())),
            "after_filter": int(len(pool)),
            "after_dedup_uniprot": int(len(uniq)),
        },
        "input_hashes": {
            "allosteric_training.csv": _sha256(TRAIN_CSV),
            "validation_data.csv": _sha256(VAL_CSV),
        },
    }
    OUT_PROV.write_text(json.dumps(prov, indent=2))

    # console summary
    show = manifest.drop(columns=["sequence", "active_residues_uniprot"])
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(show.to_string(index=False))
    print(f"\npool: raw={prov['pool_sizes']['raw']} "
          f"-> filter={prov['pool_sizes']['after_filter']} "
          f"-> dedup={prov['pool_sizes']['after_dedup_uniprot']}")
    print(f"length-bin edges (seq_len): {[round(e) for e in edges]}")
    print(f"wrote {OUT_CSV.relative_to(REPO)} and {OUT_PROV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
