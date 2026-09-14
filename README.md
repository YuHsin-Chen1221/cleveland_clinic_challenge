# Cleveland Clinic Challenge — Quantum Proposal

Phase I concept proposal for the **Global Quantum + AI Challenge 2026 (Cleveland Clinic Enterprise Challenge)**:
*A Training-Free Quantum-Walk Scanner for Cryptic Allosteric Sites in Undruggable Proteins.*

## Layout

```
docs/     LaTeX proposals (tracked on GitHub)
scripts/  encoding / analysis code (local)
data/     validation metadata (local)
```

## Proposals (`docs/`)

- **`QSW_Allosteric_Proposal.tex`** — **current / canonical version** (unary 1D-chain CTQW,
  coherence-budget analysis, Ritz back-mapping §3.5).
- `Quantum_Approach_to_Undruggable_Targets_Proposal.tex` — earlier version (amplitude-encoding
  formulation); kept for reference.

## Build

```bash
cd docs
latexmk -pdf QSW_Allosteric_Proposal.tex
```

Plain `pdflatex` also works (run twice for references). Requires TeX Live with `quantikz`,
`tikz`, `amsmath`, `amssymb`, `booktabs`, `tabularx`, `microtype`, `hyperref`.

## Encoding (`scripts/`)

Builds the sparse **Hermitian** residue Hamiltonian (3 channels: heavy-atom contacts +
ESM-2 conservation diagonal + ANM chiral phase). See `scripts/encoding.py` and
`scripts/run_encoding.py`.
