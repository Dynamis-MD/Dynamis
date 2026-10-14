# Protein Only (Vacuum / Implicit Solvent)

Pure protein structures without explicit water. Represents implicit solvent or gas-phase NNP-MD.

## Files

| File | Atoms | Box (Å) | Density | Source | Shape |
|------|------:|---------|--------:|--------|-------|
| 1aki_vacuum_openmm.xyz | 1,960 | 44×49×40 | 0.0223 | PDB 1AKI | Spherical (lysozyme) |
| 2rik_vacuum_openmm.xyz | 4,377 | 56×99×113 | 0.0070 | PDB 2RIK | Rod (titin I-band) |
| 6p9x_vacuum_openmm.xyz | 22,472 | 387×561×499 | 0.0002 | PDB 6P9X | GPCR, multi-chain (DistMLIP benchmark) |
| groel_vacuum_openmm.xyz | 56,700 | 188×188×82 | 0.0196 | PDB 1GRL | Ring, 7 chains (chaperonin) |
| 2w49_vacuum_openmm.xyz | 153,456 | 2085×2213×833 | 0.00004 | PDB 2W49 | Filament, extremely sparse (DistMLIP benchmark) |
| 9gmo_vacuum.xyz | 228,994 | 211×215×273 | 0.0184 | PDB 9GMO | Large complex |

Old files (superseded by `*_openmm.xyz` versions): `1aki_vacuum.xyz`, `2rik_vacuum.xyz`

## Source and Method

- PDB structures downloaded from RCSB (https://files.rcsb.org/download/XXXX.pdb)
- Fixed with `pdbfixer`: missing residues, missing atoms, added hydrogens at pH 7.0
- Converted to extended XYZ via ASE with tight box (3 Å buffer)
- Conversion script: see `exp_datasets/pdb/README.md`

## System Characteristics

**Compact structures** (density ~0.02 atoms/ų):
- 1AKI, GroEL, 9GMO — atoms fill most of the bounding box
- Strong internal density variation: compact core vs sparse surface

**Sparse/distributed structures** (density <0.001 atoms/ų):
- 2W49 (insect flight muscle) — long filament, atoms scattered across 2000+ Å
- 6P9X (GPCR) — multiple separated chains in large box
- These have extreme spatial non-uniformity: most of the box is empty

## Why Vacuum Proteins Matter

In solvated systems, water (80–98% of atoms) dilutes the protein's non-uniformity:
- 1AKI solvated: I2 = 1.63 (protein only 7.6%)
- 9GMO solvated: I2 = 1.15 (protein diluted at 1.5M atoms)

Pure protein has much larger internal density variation:
- Compact core (α-helix, β-sheet): high atom density
- Loops, turns, surface: lower density
- No water to fill gaps

This represents **implicit solvent NNP-MD** (e.g., AI2BMD by Microsoft), an emerging application where distributed computing is needed for proteins >10k atoms.

## References

- DistMLIP paper (2W49, 6P9X benchmarks): https://arxiv.org/abs/2506.02023
- AI2BMD (vacuum protein NNP-MD): https://github.com/microsoft/AI2BMD
- Allegro SC23 (large-scale NNP-MD): https://arxiv.org/abs/2304.10061
