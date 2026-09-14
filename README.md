# Cleveland Clinic Challenge — Quantum Allosteric-Site Scanner

Phase I of the **Global Quantum + AI Challenge 2026 (Cleveland Clinic Enterprise Challenge)**:
*A Training-Free Quantum-Walk Scanner for Cryptic Allosteric Sites in Undruggable Proteins.*

A continuous-time quantum walk on a sparse **Hermitian** residue Hamiltonian (contact topology +
ESM-2 conservation diagonal + ANM chiral phase); its average mixing matrix is the deliverable
N×N connectivity matrix, ranked for allosteric residues by connectivity to the active site.

## Repository structure

```
docs/                         proposals + labeling rules
  QSW_Allosteric_Proposal.{tex,pdf}                 canonical proposal
  Quantum_Approach_to_Undruggable_Targets_Proposal.{tex,pdf}   earlier version
  rules.md                    ground-truth labeling spec (X / Y / S rules + preconditions)

data/
  validation_data.csv         the 4 challenge targets (KRAS, BCR-ABL1, myosin, c-Myc)
  training/allosteric_training.csv   integrated ASD + AlloBench set (1,333 entries)

results/
  encodings/<target>/         3-feature Hermitian H per target: H,W,c,Phi (.npy) + nodes.csv + summary.json
```

## Datasets — shared schema

`data/validation_data.csv` and `data/training/allosteric_training.csv` use the **same columns**
(same rule, same formatting):

| column | meaning |
|---|---|
| `entry_id, sources, uniprot, pdb, chain` | identity (validation `sources` = `challenge_table`) |
| `modulator_alias, modulator_class, modulator_name` | the bound allosteric modulator |
| `allosteric_residues` (**Y**) | `A:12;A:59;…` in `pdb` author numbering — **4 Å heavy-atom scan** of the modulator |
| `active_residues_uniprot` (**S**) | `13;94;…` in **UniProt** numbering — UniProt Active+Binding (−allosteric) |
| `n_allosteric, n_active, has_active, sequence` | counts + input sequence |

## Labeling rule (see `docs/rules.md`)

- **Y (allosteric)** = protein residues with a heavy atom ≤ **4.0 Å** of the specified allosteric
  modulator (all chains). Validated to reproduce ASD's curated lists (median Jaccard 1.0); 4 Å is the
  AlloBench convention (not a self-defined 5 Å).
- **S (active)** = UniProt `Active site` ∪ `Binding site` (minus allosteric) ∪ M-CSA.
- Encoding runs only on **rule-compliant** entries (non-empty S & Y, allosteric from the 4 Å scan) —
  i.e. KRAS + ABL among the challenge targets (myosin has no drug resolved in 6C1H; c-Myc has no S).

## Build

```bash
cd docs && latexmk -pdf QSW_Allosteric_Proposal.tex   # proposal PDF
```
Requires TeX Live with `quantikz`, `tikz`, `amsmath`, `booktabs`, `tabularx`, `hyperref`.
