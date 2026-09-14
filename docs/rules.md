# Ground-truth labeling rules (X / Y / S)

Pseudo-rules for building allosteric-site ground truth, derived from and validated against
the Allosteric Database (ASD) and AlloBench (2025). These define how we produce the
labels used for the Quantum + AI Challenge and the integrated training set.

Empirical basis: our 4 A heavy-atom scan reproduces ASD's own curated residue lists at
**median Jaccard = 1.0** (3/3 unit tests exact; random sample n=75). 4 A is the AlloBench
convention and the value that matches the data — we do **not** self-define 5 A.

---

## Definitions

- **X (input)** — an apo (or heteroatom-stripped) protein structure: `pdb`, `chain`,
  `uniprot`, `sequence`. This is what a predictor sees.
- **Y (target)** — the allosteric-site residues (AFRs).
- **S (active site)** — orthosteric/catalytic residues; used as the walk source, not a target.

---

## Rule 1 — Allosteric site Y (`allosteric_site_residues`)

```
INPUT : structure (holo, modulator bound), modulator = (resname/chemical-component-id, chain, resnum)
RULE  : Y = { protein residue r :
              min_dist( any heavy atom of r , any heavy atom of modulator ) <= 4.0 A }
        - CUTOFF = 4.0 A  (heavy-atom to heavy-atom, minimum atom-atom distance)
        - search ALL protein chains (a site may lie in a chain other than the ligand's)
        - exclude the modulator itself, water, and other heteroatoms
        - modulator MUST be the *allosteric* small molecule, identified explicitly by
          (component-id, chain, resnum) — NOT "the largest HETATM"
          (e.g. 5MO4 has NIL=orthosteric + AY7=allosteric; 6OIM has MOV)
OUTPUT: residues as (chain, resnum) in the structure's author numbering
```
AlloBench "full pipeline" refinements (optional; matter for the broader DB, not our
single-chain apo targets): scan the biological assembly (`{pdb}-assembly1.cif`), merge
multi-model files, rebuild missing residues (ProMod3, keep lDDT >= 0.8), resolution <= 4 A.

## Rule 2 — Active site S (`active_site_residues`)

```
INPUT : uniprot accession, target structure
RULE  : S = ( UniProt "Active site" features
              UNION UniProt "Binding site" features
              MINUS any binding-site whose ligand/comment contains "alloster" )
            UNION  M-CSA catalytic residues            (M-CSA optional; UniProt primary)
        - S residue numbers come from the UniProt sequence
NUMBER: align the UniProt sequence to the target's modeled sequence and convert to the
        target's residue numbering (sequence alignment, not SIFTS)
OUTPUT: residues as resnum in the target's numbering
```

## Rule 3 — Map holo -> apo (`map_holo_to_apo`)

```
INPUT : holo residues (holo numbering), apo structure
RULE  : align holo and apo modeled sequences; map residue-by-residue to apo numbering
        - bridge isolated substitutions (e.g. the G12C point mutation) by consistent offset
        - handle whole-chain offsets (e.g. ABL isoform 1a/1b ~14-residue shift)
OUTPUT: Y in APO numbering, restricted to apo-resolved residues
```

---

## Preconditions (前提) and fallbacks

| Function | Requires | Assumes | If precondition fails |
|---|---|---|---|
| Y (allosteric) | holo PDB **+ the allosteric ligand resolved in coordinates** (component-id/chain/resnum) + apo PDB (for mapping) | ligand is a bound small molecule, not orthosteric | **cannot scan** -> fall back to literature/mechanism, or find a drug-bound structure |
| S (active) | UniProt accession (+ optional M-CSA) + target structure | UniProt has Active/Binding features; sequence-alignable | no catalytic line (e.g. KRAS is GAP-assisted) -> take GTP binding-site residues |
| map | apo & holo are the same protein, sequence-alignable | same protein (substitutions/offsets allowed) | too divergent -> mark residue unmapped |

**Global invariants**
1. Output Y/S in the **deployment frame = apo numbering**, restricted to apo-resolved
   residues (a residue not modeled in apo cannot be predicted or labeled).
2. The allosteric modulator is **specified explicitly** (never auto-pick largest HETATM).

---

## Training-set construction (ASD + AlloBench)

```
1. Load ASD (ASD_Release_202309_AS) and AlloBench.csv.
2. Normalize to a common schema; keep provenance (asd / allobench / both).
3. Dedup by key = (pdb, modulator_alias, modulator_chain, modulator_resi).
4. Y per entry = AlloBench residue list if non-empty, else ASD residue list
   (both are the 4 A rule applied; AlloBench uses assembly + rebuilt residues).
5. S per entry = AlloBench active_site_residue (UniProt numbering) if present.
6. PRECONDITION FILTER (keep entries that satisfy 前提):
     - allosteric_pdb present
     - modulator_class == Lig (small molecule; drop Ion / Pep / Gas)
     - n_allosteric >= 1  (a resolved allosteric site exists)
   (has_active is flagged, not required — Y is the target; S is auxiliary.)
OUTPUT: data/training/allosteric_training.csv with columns
   entry_id, sources, uniprot, pdb, chain, modulator_alias, modulator_class,
   modulator_name, n_allosteric, allosteric_residues, n_active,
   active_residues_uniprot, has_active, sequence
```

## Validation targets (apply the functions directly)

The 3 required drug pockets are absent from ASD/AlloBench (sotorasib, mavacamten absent;
asciminib present but residue list empty), so we run our own scanner:

| Target | holo + allosteric ligand | Y source | S source |
|---|---|---|---|
| KRAS_G12C | 6OIM + MOV (sotorasib) | 4 A scan -> apo 4OBE numbering | UniProt P01116 (GTP binding) |
| BCR-ABL1 | 5MO4 + AY7 (asciminib) | 4 A scan -> apo 1OPL numbering | UniProt P00519 (+AlloBench active) |
| Cardiac_myosin | 6C1H has **no mavacamten** | fall back: literature/mechanism (flagged) | UniProt P12883 (ATP/P-loop) |
