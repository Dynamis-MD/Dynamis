# PDB Source Files

Raw PDB files downloaded from RCSB and OpenMM.

## Files

| File | Source | Description | Atoms (heavy) | Converted to |
|------|--------|-------------|---------------:|-------------|
| 1GRL.pdb | RCSB PDB | GroEL chaperonin, 7 chains | 29,274 | protein_vacuum/groel_vacuum_openmm.xyz (56,700 with H) |
| 2W49.pdb | RCSB PDB | Insect flight muscle (DistMLIP benchmark) | ~50k | protein_vacuum/2w49_vacuum_openmm.xyz (153,456 with H) |
| 6P9X.pdb | RCSB PDB | GPCR protein (DistMLIP benchmark) | ~7k | protein_vacuum/6p9x_vacuum_openmm.xyz (22,472 with H) |
| apoa1.pdb | OpenMM GitHub | ApoA1 + POPC + water (pre-equilibrated) | 92,224 (all-atom) | protein_solvated/apoa1_92k_openmm.xyz |

## Download URLs

```bash
# RCSB PDB
wget https://files.rcsb.org/download/1GRL.pdb
wget https://files.rcsb.org/download/2W49.pdb
wget https://files.rcsb.org/download/6P9X.pdb

# OpenMM ApoA1 (use raw GitHub URL)
wget https://raw.githubusercontent.com/openmm/openmm/master/examples/apoa1.pdb
```

## Conversion

### Solvated (ApoA1 — already has water + PBC):
```python
from openmm.app import PDBFile
from openmm.unit import angstrom
from ase import Atoms
from ase.io import write
import numpy as np

pdb = PDBFile('apoa1.pdb')
pos = np.array([[p.value_in_unit(angstrom) for p in v] for v in pdb.getPositions()])
elements = [a.element.symbol for a in pdb.topology.atoms()]
box = np.array([v.value_in_unit(angstrom) for v in pdb.topology.getPeriodicBoxVectors()])
atoms = Atoms(symbols=elements, positions=pos, cell=box, pbc=True)
atoms.wrap()
write('output.xyz', atoms, format='extxyz')
```

### Vacuum (needs PDB fix + hydrogen addition):
```python
from pdbfixer import PDBFixer
from openmm.unit import angstrom
from ase import Atoms
from ase.io import write
import numpy as np

fixer = PDBFixer(filename='INPUT.pdb')
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

pos = np.array([[p.value_in_unit(angstrom) for p in v] for v in fixer.positions])
elements = [a.element.symbol for a in fixer.topology.atoms()]
pmin, pmax = pos.min(0), pos.max(0)
box = pmax - pmin + 6.0
shifted = pos - pmin + 3.0
atoms = Atoms(symbols=elements, positions=shifted, cell=np.diag(box), pbc=True)
write('output_vacuum.xyz', atoms, format='extxyz')
```

## References

- DistMLIP (2W49, 6P9X): https://arxiv.org/abs/2506.02023
- OpenMM ApoA1: https://github.com/openmm/openmm/tree/master/examples
