"""
eval_challenge.py — challenge-literal evaluation (Cleveland Clinic §4.1):
  "assign statistically significantly higher scores to known distal regulatory residues
   compared to (a) random background residues and (b) non-functional surface pockets."

Per target, for the ranking score z(j) (V5 coherent directed current, raw), we run two
one-sided (greater) Mann-Whitney U tests:
    T1:  z(distal positives)  vs  z(random surface background)
    T2:  z(distal positives)  vs  z(non-functional fpocket residues)
Report: U, p-value, rank-biserial effect size r_rb = 2U/(n1 n2) − 1, bootstrap 95% CI on r_rb.
Holm-correct the p-values across targets (per test family). Also report the required top-5
and top-10 hit list (raw ranking, primary) with precision/recall, plus a distance-matched
ranking column as an honesty diagnostic.

fpocket is used for negative set (b); if unavailable, T2 is reported as NA (T1 still runs).
"""
from __future__ import annotations
import json, subprocess, tempfile, shutil, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoding, walk, scoring

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "results" / "pdb_cache"
MAN = REPO / "data" / "dataset_manifest.csv"
OUT = REPO / "results" / "V5_current" / "challenge_eval.csv"
GNM = 7.3
RNG = np.random.default_rng(0)
def _find_fpocket():
    for c in [shutil.which("fpocket"),
              str(Path.home() / "miniconda3/envs/fpocket/bin/fpocket"),
              str(Path.home() / "miniconda3/bin/fpocket")]:
        if c and Path(c).exists():
            return c
    return "fpocket"
FPOCKET = _find_fpocket()


def parse_pos(s):
    return [int(x) for x in str(s).replace(",", ";").split(";") if str(x).strip()]


def surface_residues(g) -> np.ndarray:
    """Boolean mask of solvent-exposed residues (relative SASA > 0.20) via biopython Shrake-Rupley."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
        import gzip
        pdb = CACHE / f"{g.pdb_id.lower()}.pdb.gz"
        parser = PDBParser(QUIET=True)
        with gzip.open(pdb, "rt") as fh:
            struct = parser.get_structure(g.pdb_id, fh)
        ShrakeRupley().compute(struct[0], level="R")
        maxasa = {"A":129,"R":274,"N":195,"D":193,"C":167,"Q":225,"E":223,"G":104,"H":224,
                  "I":197,"L":201,"K":236,"M":224,"F":240,"P":159,"S":155,"T":172,"W":285,"Y":263,"V":174}
        chain = struct[0][g.chain]
        rel = {}
        for res in chain:
            rid = res.id[1]
            try:
                one = encoding._THREE_TO_ONE.get(res.resname, "X")
                rel[rid] = res.sasa / maxasa.get(one, 200)
            except Exception:
                pass
        mask = np.array([rel.get(r.resnum, 1.0) > 0.20 for r in g.residues])
        return mask
    except Exception:
        return np.ones(g.n, dtype=bool)          # fall back: treat all as eligible


def fpocket_residues(g, S_idx, funct_idx) -> list[int]:
    """Residue indices in non-functional fpocket pockets (not overlapping S or functional sites)."""
    if not Path(FPOCKET).exists():
        return []
    import gzip
    with tempfile.TemporaryDirectory() as td:
        pdb = Path(td) / f"{g.pdb_id}.pdb"
        src = CACHE / f"{g.pdb_id.lower()}.pdb.gz"
        with gzip.open(src, "rt") as fi, open(pdb, "w") as fo:
            fo.write(fi.read())
        try:
            subprocess.run([FPOCKET, "-f", str(pdb)], cwd=td, timeout=300,
                           capture_output=True, check=True)
        except Exception:
            return []
        outdir = Path(td) / f"{g.pdb_id}_out" / "pockets"
        if not outdir.exists():
            return []
        funct = set(S_idx) | set(funct_idx)
        resnums = set()
        for pf in sorted(outdir.glob("pocket*_atm.pdb")):
            rns = set()
            for ln in pf.read_text().splitlines():
                if ln.startswith(("ATOM", "HETATM")) and ln[21] == g.chain:
                    try: rns.add(int(ln[22:26]))
                    except ValueError: pass
            idxs = [g.index_of(r) for r in rns]; idxs = [i for i in idxs if i is not None]
            if idxs and not (set(idxs) & funct):        # keep only non-functional pockets
                resnums.update(idxs)
        return sorted(resnums)


def rank_biserial(pos, neg):
    """r_rb = 2U/(n1 n2) − 1  (one-sided greater orientation): +1 pos>neg, 0 no diff."""
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    U, p = mannwhitneyu(pos, neg, alternative="greater")
    return (2 * U / (n1 * n2) - 1), p


def boot_ci(pos, neg, n_boot=2000):
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        pb = RNG.choice(pos, len(pos), replace=True)
        nb = RNG.choice(neg, len(neg), replace=True)
        U = mannwhitneyu(pb, nb, alternative="greater").statistic
        vals.append(2 * U / (len(pos) * len(neg)) - 1)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs pass through)."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order); adj = [np.nan] * len(pvals); run = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i]); run = max(run, a); adj[i] = run
    return adj


def dmatch_rank(z, dist, cand):
    """Distance-matched ranking (diagnostic): standardise z within distance bins."""
    zc = z.copy().astype(float); dc = dist
    ed = np.quantile(dc[cand], np.linspace(0, 1, 7))
    for k in range(6):
        lo, hi = ed[k], ed[k + 1]
        m = cand & (dc >= lo) & (dc <= hi if k == 5 else dc < hi)
        if m.sum() > 2:
            b = zc[m]; sd = b.std(); zc[m] = (b - b.mean()) / sd if sd > 1e-12 else b - b.mean()
    return zc


def topk(order_idx, positives, ks=(5, 10)):
    lab = set(positives); o = {}
    for k in ks:
        hit = len(set(order_idx[:k]) & lab)
        o[f"hit@{k}"] = int(hit > 0); o[f"prec@{k}"] = hit / k
        o[f"recall@{k}"] = hit / len(lab) if lab else np.nan
    return o


def run_target(row):
    g = encoding.parse_structure(str(row["pdb"]), str(row["chain"]), cache_dir=str(CACHE))
    ca = g.ca_coords
    D = np.linalg.norm(ca[:, None] - ca[None, :], axis=2)
    W = ((D < GNM) & (D > 0)).astype(float)
    S = scoring.active_indices(str(row["sequence"]), g.sequence, parse_pos(row["active_residues_uniprot"]))
    labels = [g.index_of(int(r)) for r in str(row["allosteric_residues_chain"]).split(";") if str(r).strip()]
    labels = [i for i in labels if i is not None]
    neigh = scoring.first_neighbours(W, S); cand = scoring.candidate_mask(g.n, S, neigh)
    dS = np.min(D[:, S], axis=1) if S else np.zeros(g.n)

    # V5 readout: coherent directed current on the Laplacian generator (chiral)
    Phi = encoding.anm_chiral_phases(g, W)
    Wz = encoding._zscore_offdiag(W); np.fill_diagonal(Wz, 0.0)
    off = Wz * np.exp(1j * 0.5 * (Phi - Phi.T)); np.fill_diagonal(off, 0.0)
    Hlap = walk.laplacian_generator(off)
    Jcur = walk.enaqt_directed_current(Hlap, S, 0.0)
    z_raw = scoring.rank_z_from_vector(Jcur, cand)                 # diagnostic (proximity-loaded)
    z = scoring.distance_correct(z_raw, dS, cand)                  # DELIVERED z(j): distance-corrected

    # positives (distal, in candidate set)
    pos_idx = [i for i in labels if cand[i]]
    zpos = z[pos_idx]

    # negative set (a): random surface background (non-S, non-neighbour, non-positive)
    surf = surface_residues(g)
    bg_mask = cand & surf; bg_mask[pos_idx] = False
    zbg = z[bg_mask]

    # negative set (b): non-functional fpocket residues
    pocket_idx = fpocket_residues(g, S, [])          # functional = S only here
    pocket_idx = [i for i in pocket_idx if i not in set(pos_idx) and i not in set(S)]
    zpk = z[pocket_idx] if pocket_idx else np.array([])

    # PRIMARY significance on the delivered (distance-corrected) z
    r1, p1 = rank_biserial(zpos, zbg)
    ci1 = boot_ci(zpos, zbg)
    r2, p2 = rank_biserial(zpos, zpk) if len(zpk) else (np.nan, np.nan)
    ci2 = boot_ci(zpos, zpk) if len(zpk) else (np.nan, np.nan)
    # DIAGNOSTIC: same test on the uncorrected raw z (shows proximity-loaded raw fails)
    r1_raw, p1_raw = rank_biserial(z_raw[pos_idx], z_raw[bg_mask])

    # top-k hit lists — delivered (distance-corrected, primary) and raw (diagnostic)
    idx = np.where(cand)[0]
    deliv_order = idx[np.argsort(-z[cand])]
    raw_order = idx[np.argsort(-z_raw[cand])]
    td = topk(deliv_order, pos_idx); tr = topk(raw_order, pos_idx)

    return {
        "entry_id": row["entry_id"], "split": row["split"], "N": int(g.n),
        "n_pos": len(pos_idx), "n_bg": int(bg_mask.sum()), "n_pocket": len(pocket_idx),
        # PRIMARY: delivered distance-corrected z vs each negative set
        "T1_rrb": r1, "T1_p": p1, "T1_ci_lo": ci1[0], "T1_ci_hi": ci1[1],
        "T2_rrb": r2, "T2_p": p2, "T2_ci_lo": ci2[0], "T2_ci_hi": ci2[1],
        # diagnostic: raw (uncorrected) vs random background
        "T1_rrb_raw": r1_raw, "T1_p_raw": p1_raw,
        # required hit list (delivered / distance-corrected, primary)
        "hit@5": td["hit@5"], "prec@5": td["prec@5"], "recall@5": td["recall@5"],
        "hit@10": td["hit@10"], "prec@10": td["prec@10"], "recall@10": td["recall@10"],
        # diagnostic hit list (raw)
        "raw_hit@5": tr["hit@5"], "raw_hit@10": tr["hit@10"],
    }


def main():
    man = pd.read_csv(MAN, dtype=str)
    rows = []
    for _, r in man.iterrows():
        try:
            rows.append(run_target(r)); print(f"[done] {r['entry_id']}", flush=True)
        except Exception as e:
            print(f"[FAIL] {r['entry_id']}: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df["T1_p_holm"] = holm(df["T1_p"].tolist())
    df["T2_p_holm"] = holm(df["T2_p"].tolist())
    df = df.sort_values(["split", "N"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    show = ["entry_id","split","N","n_pos","n_bg","n_pocket","T1_rrb","T1_p_holm","T2_rrb","T2_p_holm","hit@5","recall@10"]
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", lambda x: f"{x:.3f}"):
        print(df[show].to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
