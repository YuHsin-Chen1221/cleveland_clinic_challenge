"""
make_proposal_figs.py — two decluttered single-panel figures for the proposal:
  docs/figs/prop_fig_coherence.png   (Trotter–decoherence budget, from Fig 4a)
  docs/figs/prop_fig_resource.png    (NISQ resource scaling, from Fig 5b)
Same current best model / data as the notebook; only essential marks kept.
Palette: blue + purple + black/grey; series distinguished by colour, single marker.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import scipy.linalg as sla
import matplotlib as mpl, matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import encoding, walk, scoring

CACHE = REPO / "results" / "pdb_cache"
MAN = pd.read_csv(REPO / "data" / "dataset_manifest.csv", dtype=str)
FIG = REPO / "docs" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
GNM, TARGET = 7.3, "BCR-ABL1"
T_2Q, T2_LIST, T2_LAB = 100e-9, [50e-6, 100e-6, 200e-6], ["50 µs", "100 µs", "200 µs"]
BLUE, PURPLE, BLACK, GREY = "#0072B2", "#7B3FA0", "#222222", "#888888"
T2C = ["#0072B2", "#5A4EA6", "#8E44AD"]

mpl.rcParams.update({
    "savefig.dpi": 220, "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.22, "grid.linewidth": 0.6, "legend.frameon": False,
    "axes.linewidth": 0.9,
})


def build(entry_id):
    r = MAN[MAN.entry_id == entry_id].iloc[0]
    g = encoding.parse_structure(r.pdb, r.chain, cache_dir=str(CACHE))
    D = np.linalg.norm(g.ca_coords[:, None] - g.ca_coords[None, :], axis=2)
    W = ((D < GNM) & (D > 0)).astype(float)
    S = scoring.active_indices(r.sequence, g.sequence,
        [int(x) for x in str(r.active_residues_uniprot).replace(",", ";").split(";") if x.strip()])
    Phi = encoding.anm_chiral_phases(g, W)
    Wz = encoding._zscore_offdiag(W); np.fill_diagonal(Wz, 0.0)
    off = Wz * np.exp(1j * 0.5 * (Phi - Phi.T)); np.fill_diagonal(off, 0.0)
    return g, W, S, walk.laplacian_generator(off)


# ---------------- Fig: coherence budget (single panel) ----------------
g, W, S, H = build(TARGET)
N = H.shape[0]
w = np.linalg.eigvalsh(H); tau = 6.0 / np.abs(w).max()
A = H.diagonal().real.copy(); B = H - np.diag(H.diagonal())
wB, VB = np.linalg.eigh(B)
U_exact = sla.expm(-1j * H * tau)


def U_trot(r):
    dt = tau / r
    step = np.exp(-1j * A * dt)[:, None] * ((VB * np.exp(-1j * wB * dt)[None, :]) @ VB.conj().T)
    U = np.eye(N, dtype=complex)
    for _ in range(int(r)):
        U = step @ U
    return U


r_grid = np.unique(np.round(np.logspace(0, np.log10(256), 22)).astype(int))
eps_T = np.array([np.linalg.norm(U_trot(r) - U_exact, 2) for r in r_grid])
maxdeg = int(W.sum(1).max())
eps_D = lambda r, T2: 1.0 - np.exp(-(r * maxdeg * T_2Q) / T2)
totals = {T2: eps_T + eps_D(r_grid, T2) for T2 in T2_LIST}
rstar = {T2: int(r_grid[np.argmin(totals[T2])]) for T2 in T2_LIST}

fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
ax.loglog(r_grid, eps_T, "o-", color=BLACK, lw=1.8, ms=4, label=r"Trotter error")
for T2, lab, c in zip(T2_LIST, T2_LAB, T2C):
    ax.loglog(r_grid, eps_D(r_grid, T2), "--", color=c, lw=1.3, label=fr"decoherence $T_2$={lab}")
    ax.loglog(r_grid, totals[T2], "-", color=c, lw=2.4, alpha=0.95)
    j = list(r_grid).index(rstar[T2])
    ax.plot(rstar[T2], totals[T2][j], "o", ms=9, color=c, mec=BLACK, mew=0.6, zorder=6)
ax.plot([], [], "o", ms=8, color=GREY, mec=BLACK, mew=0.6, label=r"optimal depth $r^*$")
ax.set_xlabel(r"Trotter steps $r$  (circuit depth)")
ax.set_ylabel("error  (systematic / decoherence)")
ax.legend(fontsize=8.5, loc="lower left")
fig.savefig(FIG / "prop_fig_coherence.png", bbox_inches="tight")
print("saved", FIG / "prop_fig_coherence.png", "  r*:", rstar)

# ---------------- Fig: NISQ resource scaling (single panel) ----------------
r_ref = rstar[100e-6]
rows = []
for _, r in MAN.iterrows():
    gg = encoding.parse_structure(r.pdb, r.chain, cache_dir=str(CACHE))
    Dg = np.linalg.norm(gg.ca_coords[:, None] - gg.ca_coords[None, :], axis=2)
    rows.append((gg.n, int(((Dg < GNM) & (Dg > 0)).sum() // 2)))
res = pd.DataFrame(rows, columns=["N", "edges"]).sort_values("N")
QMAX = 127

fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
ax.axvspan(0, QMAX, color=GREY, alpha=0.10)
ax.scatter(res.N, res.N, marker="o", s=46, color=BLUE, zorder=4, label="qubits (unary $=N$)")
ax.scatter(res.N, r_ref * res.edges, marker="o", s=46, color=PURPLE, zorder=4,
           label=fr"2q-gate count ($r^*\times$ edges)")
ax.axhline(QMAX, ls="--", color=GREY, lw=1)
ax.text(res.N.max() * 0.40, QMAX * 0.80, f"≈{QMAX}-qubit device", fontsize=8.5, color=GREY)
ax.axvline(QMAX, ls=":", color=GREY, lw=1)
ax.text(0.03, 0.95, "NISQ", transform=ax.transAxes, fontsize=9, color=BLUE)
ax.text(0.72, 0.62, "classical-exact", transform=ax.transAxes, fontsize=9, color=GREY)
ax.set_yscale("log")
ax.set_xlabel("system size $N$ (residues)")
ax.set_ylabel("quantum resources (count)")
ax.legend(fontsize=8.5, loc="lower right")
fig.savefig(FIG / "prop_fig_resource.png", bbox_inches="tight")
print("saved", FIG / "prop_fig_resource.png")
