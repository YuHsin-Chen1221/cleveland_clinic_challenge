"""
walk.py — Quantum-walk readouts and their classical comparators, on the SAME graph.

All operators act on the residue Hamiltonian H (complex Hermitian) or its real
contact adjacency A = |offdiag(H)|.  Everything here is exact classical simulation
(the challenge permits this as a quantum-inspired approach); no circuits yet.

Objects
  average_mixing_matrix(H)   -> M̂  (unitary CTQW, deterministic, doubly stochastic)   [p=0]
  communicability(A)         -> e^A            (Estrada, classical diffusive)
  heat_kernel(L, t)          -> e^{-Lt}        (classical continuous-time diffusion)   [p=1-like]
  lanczos(H, psi0, K)        -> (Q, Hk)        (active-site-anchored Krylov reduction)
  qsw_krylov_sweep(...)      -> {p: transport} (coherent<->classical interpolation)    [p in 0..1]
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


# --------------------------------------------------------------------------- #
# p = 0 : unitary average mixing matrix (the headline deterministic object)
# --------------------------------------------------------------------------- #
def average_mixing_matrix(H: np.ndarray, degen_tol: float = 0.0) -> np.ndarray:
    """M̂_ij = lim_T 1/T ∫ |<j|e^{-iHt}|i>|^2 dt = Σ_r |(E_r)_ij|^2.

    degen_tol == 0  -> non-degenerate formula M̂ = |V|^2 @ |V|^2.T  (fast, one matmul).
    degen_tol >  0  -> group eigenvalues within tol and use full spectral idempotents
                       (robust to near-degenerate spectra; the proposal's tolerance scan).
    """
    w, V = np.linalg.eigh(H)
    P2 = np.abs(V) ** 2                      # |v_r(i)|^2, columns = eigenvectors
    if degen_tol <= 0:
        M = P2 @ P2.T
    else:
        order = np.argsort(w)
        w, V = w[order], V[:, order]
        M = np.zeros((H.shape[0], H.shape[0]), dtype=float)
        start = 0
        for k in range(1, len(w) + 1):
            if k == len(w) or (w[k] - w[start]) > degen_tol:
                Vg = V[:, start:k]           # eigenvectors of one (near-)degenerate group
                E = Vg @ Vg.conj().T         # spectral idempotent (projector)
                M += np.abs(E) ** 2
                start = k
    # numerical hygiene: exact double-stochastic normalisation is guaranteed in theory
    return M


# --------------------------------------------------------------------------- #
# parameter-free coherence-sensitive readouts (source-anchored on S)
# --------------------------------------------------------------------------- #
def _source_state(N: int, S_idx: list[int]) -> np.ndarray:
    psi = np.zeros(N, dtype=complex)
    if S_idx:
        psi[list(S_idx)] = 1.0
    else:
        psi[0] = 1.0
    return psi / np.linalg.norm(psi)


def temporal_fluctuation(H: np.ndarray, S_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """R2 — parameter-free temporal fluctuation of source->j transport.

    p_j(t)=|<j|e^{-iHt}|ψ_S>|^2.  Closed-form long-time moments (non-degenerate):
        M_j   = <p_j>_t          = Σ_r |b_r(j)|^2        (mean, average-mixing-like)
        χ_j   = <p_j^2>_t - M_j^2 = M_j^2 - Σ_r |b_r(j)|^4   (temporal variance)
    with b_r(j) = <j|v_r><v_r|ψ_S>.  χ_j is large where many eigenmodes coherently
    interfere in the transport to j — coherence the time-average M̂ washes out.
    Returns (χ, M)."""
    w, V = np.linalg.eigh(H)
    psi = _source_state(H.shape[0], S_idx)
    proj = V.conj().T @ psi                      # <v_r|ψ_S>
    b = V * proj[None, :]                         # b[j,r] = <j|v_r><v_r|ψ_S>
    ab2 = np.abs(b) ** 2
    M = ab2.sum(axis=1)
    chi = M ** 2 - (ab2 ** 2).sum(axis=1)
    return chi, M


def _directed_from_eig(w, V, S, tau):
    U = (V * np.exp(-1j * w * tau)[None, :]) @ V.conj().T
    P = np.abs(U) ** 2                            # asymmetric iff H is complex (chiral)
    return P[S, :].sum(axis=0) - P[:, S].sum(axis=1)


def directed_transport(H: np.ndarray, S_idx: list[int], tau: float | None = None) -> np.ndarray:
    """R1 — net directed flux S->j at the natural time τ=1/bandwidth (spectrum-set, not fitted).

    J_j = Σ_{i∈S} (|U(τ)_ij|^2 - |U(τ)_ji|^2),  U(τ)=e^{-iHτ}.
    Identically zero for real-symmetric (achiral) H; nonzero only when the chiral phase
    breaks time-reversal symmetry — a directed signal no symmetric classical measure expresses."""
    w, V = np.linalg.eigh(H)
    if tau is None:
        tau = 1.0 / max(np.abs(w).max(), 1e-6)
    return _directed_from_eig(w, V, list(S_idx), tau)


def directed_transport_integrated(H: np.ndarray, S_idx: list[int],
                                  n_tau: int = 24, t_max_factor: float = 4.0) -> np.ndarray:
    """L2 — time-integrated directed flux (no single-τ choice).

    J̄_j = (1/T) ∫₀^T J_j(t) dt, accumulated over a τ-grid up to t_max = t_max_factor/bandwidth.
    Removes the arbitrariness of one τ and averages the directed bias over the natural window;
    this is also the object a depth-limited Trotter circuit approximates."""
    w, V = np.linalg.eigh(H)
    bw = max(np.abs(w).max(), 1e-6)
    ts = np.linspace(1.0 / bw / n_tau, t_max_factor / bw, n_tau)
    S = list(S_idx)
    acc = np.zeros(H.shape[0])
    for t in ts:
        acc += _directed_from_eig(w, V, S, t)
    return acc / n_tau


def ness_directed_current(H: np.ndarray, S_idx: list[int], gamma_deph: float,
                          kappa: float = 0.1, Gamma: float = 1.0, K: int = 32) -> np.ndarray:
    """A1+A2 — non-equilibrium steady-state (NESS) directed current with Haken-Strobl dephasing.

    Continuous incoherent pump at the active-site source (rate Γ), uniform drain (rate κ, models
    irreversible signal absorption so a steady state exists), pure dephasing on each Krylov node
    (rate γ_deph, the ENAQT environment). Solve  L ρ_ss + Γ|s><s| = 0  in the source-anchored
    Krylov subspace, then read the net steady-state current into each residue:
        Jcur_a = 2 Im (H_k ρ_ss)_aa ,  lifted to residues by |Q|^2.
    γ_deph = 0 is the coherent NESS; large γ_deph is the classical diffusive limit."""
    N = H.shape[0]
    psi0 = _source_state(N, S_idx)
    Q, Hk = lanczos(H, psi0, K)
    k = Hk.shape[0]
    I = np.eye(k, dtype=complex)
    Lcoh = -1j * (np.kron(I, Hk) - np.kron(Hk.T, I))
    Ldeph = np.zeros((k * k, k * k), dtype=complex)
    for a in range(k):                                  # pure dephasing: L=|a><a|
        P = np.zeros((k, k), dtype=complex); P[a, a] = 1.0
        Ldeph += np.kron(P.conj(), P) - 0.5 * np.kron(I, P) - 0.5 * np.kron(P.T, I)
    Lsup = Lcoh + gamma_deph * Ldeph - kappa * np.eye(k * k, dtype=complex)
    src = np.zeros((k, k), dtype=complex); src[0, 0] = 1.0
    vec_rho = np.linalg.solve(Lsup, -Gamma * src.reshape(-1, order="F"))
    rho = vec_rho.reshape(k, k, order="F")
    node_curr = 2.0 * np.imag(np.diag(Hk @ rho))
    return (np.abs(Q) ** 2) @ node_curr


def enaqt_directed_current(H: np.ndarray, S_idx: list[int], p: float,
                           gamma: float = 1.0, horizon: float | None = None,
                           n_time: int = 60, K: int = 32) -> np.ndarray:
    """L6 — net directed probability current under the QSW at coherence 1-p (ENAQT).

    Builds the active-site-anchored Krylov chain, evolves the open-system (Lindblad) walk at
    dephasing p, time-averages ρ̄, and reads the net directed current into each residue:
        Jcur_i = Σ_k 2 Im( H_ik ρ̄_ki )  lifted to residues by Q.
    A small amount of dephasing (interior p*) can sharpen directed transport (environment-assisted).
    Requires the chiral phase (achiral H gives zero current)."""
    N = H.shape[0]
    psi0 = _source_state(N, S_idx)
    Q, Hk = lanczos(H, psi0, K)
    if horizon is None:
        horizon = _default_horizon(Hk)
    ts = np.linspace(0.0, horizon, n_time)
    Lsup = _lindbladian_qutip(Hk, p, gamma) if _HAS_QUTIP else _lindbladian_scipy(Hk, p, gamma)
    rho_bar = _time_avg_rho(Lsup, Hk.shape[0], ts)          # K×K averaged density matrix
    # bond currents in the Krylov basis, then lift the per-node net current to residues
    Jk = 2.0 * np.imag(Hk * rho_bar.conj().T)              # Jk[a,b] current a<-b (antisym)
    node_curr = Jk.sum(axis=1)                              # net current into Krylov node a
    return (np.abs(Q) ** 2) @ node_curr                    # lift to residues via |Q|^2


# --------------------------------------------------------------------------- #
# classical comparators on the identical graph
# --------------------------------------------------------------------------- #
def contact_adjacency(H: np.ndarray) -> np.ndarray:
    """A = |off-diagonal(H)|, real symmetric, zero diagonal (the shared graph)."""
    A = np.abs(H).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def communicability(A: np.ndarray) -> np.ndarray:
    """Estrada communicability e^{A} on the contact graph (classical, diffusive)."""
    return sla.expm(A)


def graph_laplacian(A: np.ndarray) -> np.ndarray:
    """Kirchhoff/graph Laplacian L = D - A (generator of classical diffusion)."""
    return np.diag(A.sum(axis=1)) - A


def laplacian_generator(H: np.ndarray) -> np.ndarray:
    """Turn a Hamiltonian into its Laplacian-generator form (elastic-network aligned):
    off-diagonal -> -H_off (phases preserved), diagonal -> degree Σ_j|H_ij|.
    The directed-flux readout is much stronger on this generator (ablation)."""
    off = H.copy()
    np.fill_diagonal(off, 0.0)
    L = -off
    np.fill_diagonal(L, np.abs(off).sum(axis=1))
    return L


def normalized_laplacian_generator(H: np.ndarray) -> np.ndarray:
    """L3 — symmetric normalised Laplacian  L_norm = I - D^{-1/2} A D^{-1/2}  (phases kept).
    Removes residual degree/hub bias on heterogeneous-degree contact graphs."""
    off = H.copy()
    np.fill_diagonal(off, 0.0)
    deg = np.abs(off).sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Ln = -(dinv[:, None] * off * dinv[None, :])
    np.fill_diagonal(Ln, np.where(deg > 0, 1.0, 0.0))
    return Ln


def heat_kernel(A: np.ndarray, t: float) -> np.ndarray:
    """Classical continuous-time diffusion propagator e^{-Lt}, L = D - A."""
    return sla.expm(-graph_laplacian(A) * t)


# --------------------------------------------------------------------------- #
# active-site-anchored Krylov reduction (Lanczos)
# --------------------------------------------------------------------------- #
def lanczos(H: np.ndarray, psi0: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q [N x K], Hk [K x K] tridiagonal) with q1 = psi0/||psi0||.

    Source transport <j|e^{-iHt}|psi0> is preserved exactly in the K-dim subspace
    (proposal eq 5): e^{-iHt} psi0 = Q e^{-i Hk t} e1 lifted back by Q.
    """
    N = H.shape[0]
    K = int(min(K, N))
    Q = np.zeros((N, K), dtype=complex)
    alpha = np.zeros(K, dtype=float)
    beta = np.zeros(K, dtype=float)
    q = psi0.astype(complex) / np.linalg.norm(psi0)
    q_prev = np.zeros(N, dtype=complex)
    b_prev = 0.0
    for r in range(K):
        Q[:, r] = q
        Hq = H @ q
        a = np.real(np.vdot(q, Hq))
        alpha[r] = a
        w = Hq - a * q - b_prev * q_prev
        # full reorthogonalisation (stability for the small K we use)
        w -= Q[:, : r + 1] @ (Q[:, : r + 1].conj().T @ w)
        b = np.linalg.norm(w)
        if b < 1e-12 or r == K - 1:
            if r < K - 1:
                Q = Q[:, : r + 1]
                alpha = alpha[: r + 1]
                beta = beta[: r + 1]
            break
        beta[r] = b
        q_prev, q, b_prev = q, w / b, b
    k = Q.shape[1]
    Hk = np.zeros((k, k), dtype=complex)
    for r in range(k):
        Hk[r, r] = alpha[r]
        if r + 1 < k:
            Hk[r, r + 1] = beta[r]
            Hk[r + 1, r] = beta[r]
    return Q, Hk


# --------------------------------------------------------------------------- #
# QSW coherence sweep in the Krylov subspace (p: coherent -> classical)
# Primary implementation: QuTiP (field standard for open quantum systems /
# Lindblad master equations; named in the proposal). SciPy superoperator kept
# as a dependency-light fallback so the module runs without QuTiP.
# --------------------------------------------------------------------------- #
try:
    import qutip as _qt
    _HAS_QUTIP = True
except Exception:  # pragma: no cover
    _HAS_QUTIP = False


def _default_horizon(Hk: np.ndarray) -> float:
    """A few inverse-bandwidth times; same horizon reused across the whole p-sweep."""
    sp = np.abs(np.linalg.eigvalsh(Hk))
    return 6.0 / max(sp.max(), 1e-6)


def _jump_ops(Hk: np.ndarray, p: float, gamma: float) -> list[np.ndarray]:
    """Collapse operators c = √(p·γ)|i><j| along existing edges of Hk (so D scales by p)."""
    K = Hk.shape[0]
    ops = []
    if p > 0:
        for i, j in np.argwhere(np.abs(Hk) > 1e-12):
            if i == j:
                continue
            m = np.zeros((K, K), dtype=complex)
            m[i, j] = np.sqrt(p * gamma)
            ops.append(m)
    return ops


def _lindbladian_qutip(Hk: np.ndarray, p: float, gamma: float) -> np.ndarray:
    """Dense Lindblad superoperator via QuTiP (column-stacking convention)."""
    H = _qt.Qobj((1.0 - p) * Hk)
    c_ops = [_qt.Qobj(m) for m in _jump_ops(Hk, p, gamma)]
    return _qt.liouvillian(H, c_ops).full()


def _lindbladian_scipy(Hk: np.ndarray, p: float, gamma: float) -> np.ndarray:
    """Dense Lindblad superoperator (fallback), column-stacking convention to match QuTiP."""
    K = Hk.shape[0]
    I = np.eye(K, dtype=complex)
    Lsup = -1j * (1 - p) * (np.kron(I, Hk) - np.kron(Hk.T, I))
    for L in _jump_ops(Hk, p, gamma):
        LdL = L.conj().T @ L
        Lsup += np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    return Lsup


def _time_avg_rho(Lsup: np.ndarray, K: int, ts: np.ndarray) -> np.ndarray:
    """ρ̄ = (1/n) Σ_t e^{Lsup t} vec(ρ0),  ρ0 = |0><0| (source = q1).
    One expm of the (K^2 x K^2) superoperator, then cheap matvec stepping — bounded memory."""
    rho0 = np.zeros((K, K), dtype=complex)
    rho0[0, 0] = 1.0
    v = rho0.reshape(-1, order="F")
    step = sla.expm(Lsup * (ts[1] - ts[0]))
    acc = np.zeros((K, K), dtype=complex)
    for _ in ts:
        acc += v.reshape(K, K, order="F")
        v = step @ v
    return acc / len(ts)


def qsw_krylov_transport(
    Hk: np.ndarray,
    Q: np.ndarray,
    p: float,
    gamma: float = 1.0,
    horizon: float | None = None,
    n_time: int = 60,
) -> np.ndarray:
    """Source->residue transport vector under the QSW at coherence 1-p.

    Source is q1 = e1 in the Krylov basis. We time-average the density matrix over
    a finite horizon (well-defined for all p; the p=1 infinite-time average would be
    the trivial stationary state), then lift populations to residues by Q:
        transport[i] = (Q ρ̄ Q^†)_ii ,   returned as a length-N real vector.
    """
    if horizon is None:
        horizon = _default_horizon(Hk)
    ts = np.linspace(0.0, horizon, n_time)
    Lsup = _lindbladian_qutip(Hk, p, gamma) if _HAS_QUTIP else _lindbladian_scipy(Hk, p, gamma)
    rho_bar = _time_avg_rho(Lsup, Hk.shape[0], ts)
    return np.real(np.einsum("ik,kl,il->i", Q, rho_bar, Q.conj()))


def qsw_krylov_sweep(
    Hk: np.ndarray,
    Q: np.ndarray,
    ps: np.ndarray,
    gamma: float = 1.0,
    horizon: float | None = None,
    n_time: int = 60,
) -> dict[float, np.ndarray]:
    """Transport vectors for a grid of coherence values p in [0, 1]."""
    return {
        float(p): qsw_krylov_transport(Hk, Q, float(p), gamma, horizon, n_time)
        for p in ps
    }
