# Experiment Datasets for Partition Imbalance Analysis

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

### Baselines
| File | Atoms | Source | Type |
|------|------:|--------|------|
| ch4_8k_splited.xyz | 8,001 | Self-generated | Uniform gas |
| ch4_8k_cluster.xyz | 8,001 | Self-generated | Artificial cluster |

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
