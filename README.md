# Cleveland Clinic Challenge — Quantum Proposal

Phase I concept proposal for the **Global Quantum + AI Challenge 2026 (Cleveland Clinic Enterprise Challenge)**:
*A Training-Free Quantum-Walk Scanner for Cryptic Allosteric Sites in Undruggable Proteins.*

## Contents

- `Quantum_Approach_to_Undruggable_Targets_Proposal.tex` — LaTeX source
- `Quantum_Approach_to_Undruggable_Targets_Proposal.pdf` — compiled proposal (6 pp)

## Build

```bash
latexmk -pdf Quantum_Approach_to_Undruggable_Targets_Proposal.tex
```

Plain `pdflatex` also works (no CJK / XeLaTeX required):

```bash
pdflatex Quantum_Approach_to_Undruggable_Targets_Proposal.tex
pdflatex Quantum_Approach_to_Undruggable_Targets_Proposal.tex   # 2nd pass for refs
```

Requires a TeX Live install with `tikz` (positioning, shapes.geometric, calc), `amsmath`,
`amssymb`, `booktabs`, `tabularx`, `microtype`, and `hyperref`.
