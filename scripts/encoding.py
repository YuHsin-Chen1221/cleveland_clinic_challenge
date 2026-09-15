"""
encoding.py — Feature-encoding for the training-free quantum-walk allosteric scanner.

Builds a sparse **Hermitian** residue-level Hamiltonian H from an apo structure:

    H_ij = w_ij * exp(i * phi_ij)   (i != j)      # off-diagonal hopping (+ chiral phase)
    H_ii = c_i                       (real)        # on-site energy (conservation)

Three channels (all ab-initio from apo coords + sequence — NO molecular dynamics):
  1. topology    w_ij : heavy-atom contact count (<=4.5 A) or GNM Cα contacts (<=7.3 A)
  2. conservation c_i : ESM-2 per-residue log-likelihood (wt-marginal), z-scored -> diagonal
  3. chiral phase phi_ij : ANM lowest-mode projection along edge (antisymmetric) -> directional

Hermiticity is guaranteed by construction (symmetric magnitudes, real diagonal,
antisymmetric phases) and asserted: ||H - H^dagger||_F < 1e-10.

Heavy imports (prody / torch / esm) are done lazily inside functions so the module
imports cheaply for tests that only need part of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ----------------------------------------------------------------------------- #
# Residue parsing
# ----------------------------------------------------------------------------- #

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified / alt names -> parent
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
    "CSO": "C", "PTR": "Y", "SEP": "S", "TPO": "T",
}


@dataclass
class Residue:
    resnum: int
    icode: str
    resname: str
    one: str            # one-letter (X if unknown)
    ca: np.ndarray      # (3,) Cα coordinate
    heavy: np.ndarray   # (n_atoms, 3) heavy-atom coordinates


@dataclass
class ProteinGraph:
    """Ordered residues of one chain + derived index maps."""
    pdb_id: str
    chain: str
    residues: list[Residue]

    @property
    def n(self) -> int:
        return len(self.residues)

    @property
    def sequence(self) -> str:
        return "".join(r.one for r in self.residues)

    @property
    def ca_coords(self) -> np.ndarray:
        return np.array([r.ca for r in self.residues], dtype=float)

    @property
    def resnums(self) -> list[int]:
        return [r.resnum for r in self.residues]

    def index_of(self, resnum: int) -> Optional[int]:
        """Node index for an author residue number (first match, ignores icode)."""
        for i, r in enumerate(self.residues):
            if r.resnum == resnum:
                return i
        return None

    def indices_for(self, resnums: list[int]) -> list[int]:
        out = []
        for rn in resnums:
            idx = self.index_of(rn)
            if idx is not None:
                out.append(idx)
        return out


def parse_structure(pdb_id: str, chain: str, cache_dir: Optional[str] = None) -> ProteinGraph:
    """Fetch (if needed) and parse one chain into an ordered residue list.

    Keeps only standard protein residues that have a Cα atom. Residues are ordered
    by appearance; the resulting sequence/indexing is 1:1 with the graph nodes.
    """
    import os
    import prody

    prody.confProDy(verbosity="none")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        prody.pathPDBFolder(cache_dir)

    ag = prody.parsePDB(pdb_id)
    if ag is None:
        raise RuntimeError(f"Could not fetch/parse PDB {pdb_id}")

    sel = ag.select(f"protein and chain {chain} and not hydrogen")
    if sel is None:
        raise RuntimeError(f"No protein atoms for {pdb_id} chain {chain}")

    residues: list[Residue] = []
    hv = prody.HierView(sel)
    for res in hv.iterResidues():
        resname = res.getResname().strip().upper()
        one = _THREE_TO_ONE.get(resname, "X")
        ca = res.select("name CA")
        if ca is None:                       # skip residues without a Cα
            continue
        ca_xyz = ca.getCoords()[0]
        heavy_xyz = res.getCoords()          # already heavy-only (H excluded in sel)
        residues.append(
            Residue(
                resnum=int(res.getResnum()),
                icode=(res.getIcode() or "").strip(),
                resname=resname,
                one=one,
                ca=np.asarray(ca_xyz, dtype=float),
                heavy=np.asarray(heavy_xyz, dtype=float),
            )
        )

    if not residues:
        raise RuntimeError(f"No Cα residues parsed for {pdb_id} chain {chain}")
    return ProteinGraph(pdb_id=pdb_id.upper(), chain=chain, residues=residues)


# ----------------------------------------------------------------------------- #
# Channel 1 — contact topology (off-diagonal magnitude)
# ----------------------------------------------------------------------------- #

def contact_topology(g: ProteinGraph, mode: str = "heavy", cutoff: Optional[float] = None) -> np.ndarray:
    """Symmetric, zero-diagonal contact weight matrix W (N x N).

    mode="heavy": count heavy-atom pairs (<= cutoff, default 4.5 A) between residue
                  i and j -> weighted contacts (finer than binary).
    mode="gnm"  : binary Cα contact (<= cutoff, default 7.3 A) -> GNM adjacency A.
    """
    from scipy.spatial import cKDTree

    n = g.n
    W = np.zeros((n, n), dtype=float)

    if mode == "gnm":
        cutoff = 7.3 if cutoff is None else cutoff
        ca = g.ca_coords
        tree = cKDTree(ca)
        for i, j in tree.query_pairs(cutoff):
            W[i, j] = W[j, i] = 1.0
        return W

    if mode == "heavy":
        cutoff = 4.5 if cutoff is None else cutoff
        # flatten all heavy atoms, remember which residue each belongs to
        coords, owner = [], []
        for idx, r in enumerate(g.residues):
            coords.append(r.heavy)
            owner.extend([idx] * len(r.heavy))
        coords = np.vstack(coords)
        owner = np.asarray(owner)
        tree = cKDTree(coords)
        for a, b in tree.query_pairs(cutoff):
            ri, rj = owner[a], owner[b]
            if ri != rj:                     # ignore intra-residue atom pairs
                W[ri, rj] += 1.0
                W[rj, ri] += 1.0
        return W

    raise ValueError(f"unknown contact mode {mode!r} (use 'heavy' or 'gnm')")


# ----------------------------------------------------------------------------- #
# Channel 2 — ESM-2 conservation (diagonal on-site energy)
# ----------------------------------------------------------------------------- #

def esm2_conservation(
    sequence: str,
    model_name: str = "esm2_t33_650M_UR50D",
    method: str = "wt_marginal",
    device: Optional[str] = None,
    batch_masked: int = 16,
) -> np.ndarray:
    """Per-residue conservation proxy from ESM-2, length == len(sequence).

    method="wt_marginal" (default, 1 forward pass): log P(wt_i | context) — higher =
        the residue is what the language model expects there (a functional/conservation
        signal). Cheap; good default for a baseline.
    method="masked_marginal" (L forward passes, batched): mask each position, read
        log P(wt_i | context_without_i). More faithful, much slower.

    Returns the RAW per-residue log-likelihood (not yet z-scored); callers z-score
    it before placing it on the diagonal so it is comparable to the topology channel.
    """
    import torch
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()

    L = len(sequence)
    if L > 1022:
        raise ValueError(
            f"sequence length {L} exceeds ESM-2 single-pass limit (1022); "
            "chunking not implemented in this baseline."
        )

    _, _, tokens = bc([("prot", sequence)])
    tokens = tokens.to(device)               # (1, L+2) incl BOS/EOS
    wt_ids = tokens[0, 1:L + 1].clone()      # residue token ids

    logp = np.zeros(L, dtype=float)

    if method == "wt_marginal":
        with torch.no_grad():
            logits = model(tokens)["logits"][0]           # (L+2, vocab)
        lsm = torch.log_softmax(logits[1:L + 1], dim=-1)  # residue positions
        logp = lsm[torch.arange(L), wt_ids].float().cpu().numpy()
        return logp

    if method == "masked_marginal":
        mask_id = alphabet.mask_idx
        positions = list(range(L))
        for start in range(0, L, batch_masked):
            batch_pos = positions[start:start + batch_masked]
            batch = tokens.repeat(len(batch_pos), 1).clone()
            for row, p in enumerate(batch_pos):
                batch[row, p + 1] = mask_id                # +1 for BOS offset
            with torch.no_grad():
                logits = model(batch)["logits"]            # (b, L+2, vocab)
            for row, p in enumerate(batch_pos):
                lsm = torch.log_softmax(logits[row, p + 1], dim=-1)
                logp[p] = float(lsm[wt_ids[p]])
        return logp

    raise ValueError(f"unknown method {method!r}")


# ----------------------------------------------------------------------------- #
# Channel 3 — ANM chiral phases (directional off-diagonal)
# ----------------------------------------------------------------------------- #

def anm_chiral_phases(
    g: ProteinGraph,
    W: np.ndarray,
    n_modes: int = 1,
    cutoff: float = 13.0,
    max_phase: float = np.pi / 2,
) -> np.ndarray:
    """Antisymmetric phase matrix Phi (N x N), nonzero only on existing edges.

    For each contact edge (i,j) we project the ANM slowest-mode relative displacement
    onto the edge direction:
        delta_ij = u_hat_ij . (v_i - v_j)        (averaged over the lowest n_modes)
    delta is antisymmetric (delta_ji = -delta_ij), so mapping phi_ij = max_phase *
    delta_ij / max|delta| yields phi_ji = -phi_ij  ->  Hermiticity is preserved.

    Only the phase accumulated around cycles is physical; per-edge phases carry a
    gauge that a diagonal unitary can absorb. For this baseline we keep the raw
    edge phases (a gauge-fixing pass can be layered on later).
    """
    import prody

    ca = g.ca_coords
    n = g.n

    anm = prody.ANM(g.pdb_id)
    anm.buildHessian(ca, cutoff=cutoff)
    anm.calcModes(n_modes=n_modes, zeros=False)     # exclude 6 trivial modes
    vecs = anm.getEigvecs()                          # (3N, n_modes)

    modes = [vecs[:, m].reshape(n, 3) for m in range(vecs.shape[1])]

    Phi = np.zeros((n, n), dtype=float)
    iu, ju = np.where(np.triu(W, k=1) > 0)           # existing edges (upper tri)
    for i, j in zip(iu, ju):
        d = ca[j] - ca[i]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        u = d / norm
        # average projected relative displacement over the chosen modes
        delta = np.mean([u @ (mode[i] - mode[j]) for mode in modes])
        Phi[i, j] = delta
        Phi[j, i] = -delta

    m = np.max(np.abs(Phi))
    if m > 0:
        Phi = (max_phase / m) * Phi                  # scale to [-max_phase, max_phase]
    return Phi


# ----------------------------------------------------------------------------- #
# Fusion -> Hermitian Hamiltonian
# ----------------------------------------------------------------------------- #

def _zscore_offdiag(W: np.ndarray) -> np.ndarray:
    """Z-score the nonzero off-diagonal entries in place-preserving sparsity."""
    Wsym = 0.5 * (W + W.T)
    np.fill_diagonal(Wsym, 0.0)
    mask = np.triu(np.ones_like(Wsym, dtype=bool), k=1) & (Wsym != 0)
    vals = Wsym[mask]
    if vals.size == 0:
        return Wsym
    mu, sd = vals.mean(), vals.std()
    if sd < 1e-12:
        return Wsym
    Z = np.zeros_like(Wsym)
    nz = Wsym != 0
    Z[nz] = (Wsym[nz] - mu) / sd
    return 0.5 * (Z + Z.T)                            # keep exactly symmetric


def _zscore_vec(c: np.ndarray) -> np.ndarray:
    mu, sd = c.mean(), c.std()
    return (c - mu) / sd if sd > 1e-12 else c - mu


def assemble_hamiltonian(
    W: np.ndarray,
    c: Optional[np.ndarray] = None,
    Phi: Optional[np.ndarray] = None,
    w_topology: float = 1.0,
    w_conservation: float = 1.0,
    zscore: bool = True,
    assert_hermitian: bool = True,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fuse channels into a complex Hermitian Hamiltonian H.

    off-diagonal : w_topology * z(W) * exp(i * Phi)     (symmetric magnitude, antisym phase)
    diagonal     : w_conservation * z(c)                (real)
    """
    n = W.shape[0]
    Wz = _zscore_offdiag(W) if zscore else 0.5 * (W + W.T)
    np.fill_diagonal(Wz, 0.0)

    if Phi is not None:
        Aphi = 0.5 * (Phi - Phi.T)                     # force antisymmetry
        off = w_topology * Wz * np.exp(1j * Aphi)
    else:
        off = (w_topology * Wz).astype(complex)
    np.fill_diagonal(off, 0.0)

    H = off
    if c is not None:
        diag = _zscore_vec(c) if zscore else c.astype(float)
        H = H + np.diag((w_conservation * diag).astype(complex))

    if assert_hermitian:
        err = np.linalg.norm(H - H.conj().T)
        if err > tol:
            raise AssertionError(f"H not Hermitian: ||H - H^dagger||_F = {err:.3e} > {tol:.0e}")
    return H


# ----------------------------------------------------------------------------- #
# High-level builders
# ----------------------------------------------------------------------------- #

def build_baseline_H(g: ProteinGraph, zscore: bool = True) -> np.ndarray:
    """GNM-only baseline: pure Cα contact topology, real symmetric (no diagonal, no phase)."""
    A = contact_topology(g, mode="gnm")
    return assemble_hamiltonian(A, c=None, Phi=None, zscore=zscore)


def build_feature_H(
    g: ProteinGraph,
    use_conservation: bool = True,
    use_chiral: bool = True,
    conservation_method: str = "wt_marginal",
    esm_model: str = "esm2_t33_650M_UR50D",
    device: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """Full 3-channel Hamiltonian. Returns (H, channel_artifacts)."""
    W = contact_topology(g, mode="heavy", cutoff=4.5)

    c = None
    if use_conservation:
        c = esm2_conservation(g.sequence, model_name=esm_model,
                              method=conservation_method, device=device)

    Phi = None
    if use_chiral:
        Phi = anm_chiral_phases(g, W)

    H = assemble_hamiltonian(W, c=c, Phi=Phi)
    artifacts = {"W": W, "c": c, "Phi": Phi}
    return H, artifacts


# ----------------------------------------------------------------------------- #
# Diagnostics
# ----------------------------------------------------------------------------- #

def summarize_H(H: np.ndarray) -> dict:
    n = H.shape[0]
    off = H.copy()
    np.fill_diagonal(off, 0.0)
    nnz = int(np.count_nonzero(np.triu(off, k=1)))
    deg = np.count_nonzero(off, axis=1)
    herm_err = float(np.linalg.norm(H - H.conj().T))
    is_complex = bool(np.iscomplexobj(H) and np.any(np.abs(off.imag) > 1e-12))
    return {
        "N": n,
        "edges": nnz,
        "sparsity": float(nnz / (n * (n - 1) / 2)) if n > 1 else 0.0,
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()) if n else 0,
        "hermiticity_error": herm_err,
        "is_complex_hermitian": is_complex,      # True once chiral phases are on
        "diag_is_real": bool(np.allclose(H.diagonal().imag, 0.0)),
    }
