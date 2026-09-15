"""
run_baseline.py — Per-protein baseline simulation + evaluation for the QSW allosteric scanner.

For each protein and each Hamiltonian arm it computes, on the SAME graph:
  quantum   : M̂  average mixing matrix (exact, p=0)        -> z_quantum   (headline)
  classical : e^A communicability (parameter-free)          -> z_comm
  classical : QSW p=1 endpoint (Krylov, finite horizon)     -> z_qsw_p1   (diffusion)
  degenerate: raw graph degree                              -> z_degree
  null      : random                                        -> z_random
  quantum   : QSW p=0 endpoint (Krylov)                     -> z_qsw_p0   (M̂ consistency)
  sweep     : QSW p in [0,1] (Krylov)                       -> AUC-vs-p curve

Arms:
  topology  : GNM contact graph only (real symmetric)        [no ESM-2, CPU-only]
  fusion    : contacts + ESM-2 conservation + ANM chiral     [needs torch/ESM-2]

Outputs per (protein, arm) under out_dir/<entry_id>/<arm>/:
  M_hat.npy  z.npz  metrics.json  qsw_curve.json      (checkpointed: skipped if metrics.json exists)
Aggregate metrics table written to out_dir/metrics.csv.

Local smoke:
  python scripts/run_baseline.py --entries KRAS_G12C --arms topology
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import encoding
import scoring
import walk

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "dataset_manifest.csv"
CACHE = REPO / "results" / "pdb_cache"
OUT = REPO / "results" / "baseline"
QSW_PS = np.round(np.linspace(0.0, 1.0, 11), 3)     # coherence sweep grid
KRYLOV_K = 32                                        # Krylov dim for the QSW sweep (proposal 32-64)
RANDOM_SEED = 12345


def _parse_pos_list(spec: str) -> list[int]:
    if not isinstance(spec, str) or not spec.strip():
        return []
    out = []
    for tok in spec.replace(",", ";").split(";"):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(tok))
            except ValueError:
                pass
    return out


def build_H(g, arm: str, device: str | None):
    if arm == "topology":
        return encoding.build_baseline_H(g)
    if arm == "chiral":            # winning combo (ablation): GNM Cα-7.3Å contacts + ANM chiral, NO ESM
        W = encoding.contact_topology(g, mode="gnm")      # Cα 7.3Å graph (matches the ablation winner)
        Phi = encoding.anm_chiral_phases(g, W)
        return encoding.assemble_hamiltonian(W, c=None, Phi=Phi)
    if arm == "fusion":
        H, _ = encoding.build_feature_H(g, use_conservation=True, use_chiral=True, device=device)
        return H
    raise ValueError(f"unknown arm {arm}")


def run_arm(g, row, arm: str, out_dir: Path, device: str | None) -> dict:
    adir = out_dir / arm
    adir.mkdir(parents=True, exist_ok=True)
    mfile = adir / "metrics.json"
    if mfile.exists():                                # resume: skip completed work
        return json.loads(mfile.read_text())

    t0 = time.perf_counter()
    struct_seq = g.sequence
    ref_seq = str(row["sequence"])
    S_idx = scoring.active_indices(ref_seq, struct_seq, _parse_pos_list(row["active_residues_uniprot"]))
    labels = [g.index_of(r) for r in _parse_pos_list(row["allosteric_residues_chain"])]
    labels = [i for i in labels if i is not None]

    H = build_H(g, arm, device)
    N = H.shape[0]
    A = walk.contact_adjacency(H)
    neigh = scoring.first_neighbours(A, S_idx)
    cand = scoring.candidate_mask(N, S_idx, neigh)
    # min Cα distance to the source set S (for challenge-mandated distance control)
    ca = g.ca_coords
    dS = (np.min(np.linalg.norm(ca[:, None, :] - ca[list(S_idx)][None, :, :], axis=2), axis=1)
          if S_idx else np.zeros(N))

    # --- readouts on the same graph ---
    M_hat = walk.average_mixing_matrix(H)                              # quantum avg-mixing (dephased)
    comm = walk.communicability(A)                                    # classical e^A
    chi, _ = walk.temporal_fluctuation(H, S_idx)                      # R2 temporal fluctuation
    J_adj = walk.directed_transport(H, S_idx)                         # R1 directed flux (adjacency)
    H_lap = walk.laplacian_generator(H)                               # elastic-network generator
    J_lap = walk.directed_transport(H_lap, S_idx)                     # V3 directed flux (Laplacian)
    Jcur_lap = walk.enaqt_directed_current(H_lap, S_idx, 0.0)         # V5 coherent directed current
    z = {
        "quantum": scoring.rank_z_from_matrix(M_hat, S_idx, cand),
        "comm": scoring.rank_z_from_matrix(comm, S_idx, cand),
        "degree": scoring.degree_score(A, cand),
        "random": scoring.random_score(N, cand, RANDOM_SEED),
        "chi": scoring.rank_z_from_vector(chi, cand),
        "flux_adj": scoring.rank_z_from_vector(J_adj, cand),
        "flux_lap": scoring.rank_z_from_vector(J_lap, cand),          # V3 readout
        "flux_current_lap": scoring.rank_z_from_vector(Jcur_lap, cand),  # V5 readout
    }

    # --- QSW coherence sweep in the active-site-anchored Krylov subspace ---
    psi0 = np.zeros(N, dtype=complex)
    if S_idx:
        psi0[list(S_idx)] = 1.0
    else:
        psi0[0] = 1.0
    Qk, Hk = walk.lanczos(H, psi0, KRYLOV_K)
    sweep = walk.qsw_krylov_sweep(Hk, Qk, QSW_PS)
    qsw_curve = {}
    for p, vec in sweep.items():
        zp = scoring.rank_z_from_vector(vec, cand)
        m = scoring.evaluate(zp, labels, cand, dist_to_S=dS)
        qsw_curve[str(p)] = {"auc": m["auc"], "auc_distmatched": m["auc_distmatched"],
                             "hit@5": m["hit@5"], "hit@10": m["hit@10"]}
    z["qsw_p0"] = scoring.rank_z_from_vector(sweep[float(QSW_PS[0])], cand)
    z["qsw_p1"] = scoring.rank_z_from_vector(sweep[float(QSW_PS[-1])], cand)

    # --- evaluate every readout (with distance control) ---
    ev = {name: scoring.evaluate(zv, labels, cand, dist_to_S=dS) for name, zv in z.items()}
    q = ev["quantum"]
    dist_base = q["auc_dist_baseline"]                # geometric confound level (same for all readouts)
    best_classical_dm = max(
        (ev[k]["auc_distmatched"] for k in ("comm", "qsw_p1", "degree")
         if not np.isnan(ev[k]["auc_distmatched"])),
        default=float("nan"),
    )
    runtime = time.perf_counter() - t0

    metrics = {
        "entry_id": row["entry_id"], "split": row["split"], "arm": arm,
        "N": int(N), "seq_len": int(row["seq_len"]),
        "n_active_mapped": len(S_idx), "n_labels_mapped": len(labels), "n_pos": q["n_pos"],
        # raw (distance-confounded) — kept for reference
        "auc_quantum": q["auc"], "hit@5": q["hit@5"], "hit@10": q["hit@10"],
        "prec@5": q["prec@5"], "mwu_p": q["mwu_p"],
        # distance-controlled (the honest, challenge-mandated primary metric)
        "auc_quantum_distmatched": q["auc_distmatched"],
        "auc_dist_baseline": dist_base,
        "delta_dm_vs_chance": (q["auc_distmatched"] - 0.5) if not np.isnan(q["auc_distmatched"]) else float("nan"),
        # new coherence-sensitive readouts (distance-matched) — R1/R2 from the ablation
        "auc_flux_lap_dm": ev["flux_lap"]["auc_distmatched"],      # V3 readout
        "auc_flux_current_lap_dm": ev["flux_current_lap"]["auc_distmatched"],  # V5 readout
        "auc_flux_adj_dm": ev["flux_adj"]["auc_distmatched"],
        "auc_chi_dm": ev["chi"]["auc_distmatched"],
        "hit@5_flux_lap": ev["flux_lap"]["hit@5"], "hit@10_flux_lap": ev["flux_lap"]["hit@10"],
        "hit@5_flux_current_lap": ev["flux_current_lap"]["hit@5"],
        "hit@10_flux_current_lap": ev["flux_current_lap"]["hit@10"],
        "delta_flux_vs_dist": (ev["flux_lap"]["auc_distmatched"] - 0.5)
            if not np.isnan(ev["flux_lap"]["auc_distmatched"]) else float("nan"),
        "auc_comm_dm": ev["comm"]["auc_distmatched"],
        "auc_qsw_p1_dm": ev["qsw_p1"]["auc_distmatched"],
        "auc_qsw_p0_dm": ev["qsw_p0"]["auc_distmatched"],
        "auc_degree_dm": ev["degree"]["auc_distmatched"],
        "delta_quantum_vs_classical_dm": (q["auc_distmatched"] - best_classical_dm)
            if not np.isnan(best_classical_dm) else float("nan"),
        # raw comparators
        "auc_comm": ev["comm"]["auc"], "auc_degree": ev["degree"]["auc"], "auc_random": ev["random"]["auc"],
        "runtime_sec": round(runtime, 2), "krylov_K": int(Hk.shape[0]),
    }

    # --- persist (checkpoint) ---
    np.save(adir / "M_hat.npy", M_hat.astype(np.float32))
    np.savez_compressed(adir / "z.npz", **{k: v for k, v in z.items()}, cand=cand,
                        S_idx=np.array(S_idx), labels=np.array(labels))
    (adir / "qsw_curve.json").write_text(json.dumps(qsw_curve, indent=2))
    mfile.write_text(json.dumps({**metrics, "per_readout": ev}, indent=2))
    return {**metrics, "per_readout": ev}


def run_entry(row: dict, arms: list[str], out_dir: Path, device: str | None) -> list[dict]:
    edir = out_dir / str(row["entry_id"])
    edir.mkdir(parents=True, exist_ok=True)
    g = encoding.parse_structure(str(row["pdb"]), str(row["chain"]), cache_dir=str(CACHE))
    return [run_arm(g, row, arm, edir, device) for arm in arms]


def aggregate(out_dir: Path) -> None:
    rows = []
    for mf in sorted(out_dir.glob("*/*/metrics.json")):
        d = json.loads(mf.read_text())
        d.pop("per_readout", None)
        rows.append(d)
    if rows:
        df = pd.DataFrame(rows).sort_values(["split", "seq_len", "arm"])
        df.to_csv(out_dir / "metrics.csv", index=False)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", nargs="*", default=None, help="entry_ids (default: all)")
    ap.add_argument("--arms", nargs="*", default=["topology", "fusion"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default=None, help="torch device for ESM-2 (fusion arm)")
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST, dtype=str)
    man["seq_len"] = man["seq_len"].astype(int)
    if args.entries:
        man = man[man["entry_id"].isin(args.entries)]
    out_dir = Path(args.out)

    for _, row in man.iterrows():
        try:
            run_entry(row.to_dict(), args.arms, out_dir, args.device)
            print(f"[done] {row['entry_id']} ({row['seq_len']} aa) arms={args.arms}")
        except Exception as e:
            print(f"[FAIL] {row['entry_id']}: {type(e).__name__}: {e}")
    aggregate(out_dir)


if __name__ == "__main__":
    main()
