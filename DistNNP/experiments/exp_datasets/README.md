# Experiment Datasets for Partition Imbalance Analysis

## Getting the data

Dataset files are **not committed to git** — obtain them on demand:

```bash
bash scripts/setup_data.sh      # generate CH4 + download PDB + fetch hosted archive
```

Each dataset is obtained the most reproducible way available (see
`scripts/README.md` for the full policy):

- **Tier 1 — CH4/O2 baselines** (normal-density `splited`): generated locally by
  `scripts/03_generate_ch4.py`, **bit-reproducible** on any platform / NumPy
  version (numpy is the only dep). The overlapping `cluster` variant is opt-in.
- **Tier 2 — raw PDB structures**: downloaded from RCSB by accession.
- **Tier 3 — protein/membrane XYZ**: downloaded as one SHA256-verified archive
  (their pdbfixer/OpenMM origin is not reproducible across tool versions).
- **Tier 4 — large `*_density.xyz`**: external user data (`../data/` symlinks).

## Directory Structure

```
exp_datasets/
├── pdb/                 # Raw PDB source files
├── baselines/           # Uniform and clustered CH4 (controls)
├── protein_solvated/    # Protein + explicit water (PBC, liquid density)
├── protein_vacuum/      # Protein only, no water (tight box)
├── electrochemical/     # Pt(111) slab + explicit water
└── membrane/            # DPPC lipid bilayer + water
```

## All Datasets

### Baselines (CH4:O2, generated — `scripts/03_generate_ch4.py`)
| File | Atoms | Source | Type |
|------|------:|--------|------|
| ch4_Nk_splited.xyz | N×1000 | Self-generated | **Default, normal density** |
| ch4_Nk_cluster.xyz | N×1000 | Self-generated | *Opt-in* extreme imbalance artifact (overlapping; `--variants cluster`) |

### Protein + Water (Solvated)
| File | Atoms | Source | Equilibrated? |
|------|------:|--------|:---:|
| dhfr_23k_openmm.xyz | 23,558 | OpenMM openmmtools | Yes |
| 1aki_density.xyz | 25,846 | PDB + grid water | No |
| 2rik_density.xyz | 112,727 | PDB + grid water | No |
| 9gmo_density.xyz | 1,536,442 | PDB + grid water | No |

### Protein Only (Vacuum)
| File | Atoms | Source | Shape |
|------|------:|--------|-------|
| 1aki_vacuum_openmm.xyz | 1,960 | PDB 1AKI | Spherical |
| 2rik_vacuum_openmm.xyz | 4,377 | PDB 2RIK | Rod |
| 6p9x_vacuum_openmm.xyz | 22,472 | PDB 6P9X (DistMLIP) | GPCR, multi-chain |
| groel_vacuum_openmm.xyz | 56,700 | PDB 1GRL | Ring, 7 chains |
| 2w49_vacuum_openmm.xyz | 153,456 | PDB 2W49 (DistMLIP) | Filament, sparse |
| 9gmo_vacuum.xyz | 228,994 | PDB 9GMO | Large complex |

### Electrochemical Interface
| File | Atoms | Source |
|------|------:|--------|
| pt_water_12k_density.xyz | 14,236 | ASE fcc111 + grid water |
| pt_water_17k_density.xyz | 21,180 | ASE fcc111 + grid water |

### Membrane
| File | Atoms | Source |
|------|------:|--------|
| dppc_membrane_28k.xyz | 28,160 | OpenMM built-in DPPC.pdb (equilibrated) |
| apoa1_92k_openmm.xyz | 92,224 | OpenMM GitHub (protein + POPC membrane + water) |

## Data Sources

| Source | Systems | How obtained |
|--------|---------|-------------|
| RCSB PDB | 1AKI, 2RIK, 9GMO, 1GRL, 2W49, 6P9X | wget from https://files.rcsb.org/download/ |
| OpenMM/openmmtools | DHFR, ApoA1, DPPC | pip install openmmtools; GitHub download |
| DistMLIP paper | 2W49, 6P9X | PDB download (benchmark systems from https://arxiv.org/abs/2506.02023) |
| Self-generated | CH4, Pt slab | ASE build + grid water fill |

## Generation Scripts

| Script | Purpose |
|--------|---------|
| test/generate_solvated.py | Protein + grid water solvation |
| test/generate_electrochemical.py | Pt slab + water |
| exp_datasets/pdb/README.md | PDB → XYZ conversion scripts |
