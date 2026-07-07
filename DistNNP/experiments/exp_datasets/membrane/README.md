# Lipid Membrane (Bilayer + Water)

## Current Files

| File | Atoms | Source | Equilibrated? | Description |
|------|------:|--------|:---:|-------------|
| dppc_membrane_28k.xyz | 28,160 | OpenMM built-in DPPC.pdb | Yes | Pure DPPC lipid bilayer + water (no protein) |
| **apoa1_92k_openmm.xyz** | **92,224** | **OpenMM GitHub PDB** | **Yes** | **ApoA1 protein + POPC membrane + water** |

## Additional Membranes (Available Locally)

OpenMM has 7 pre-equilibrated bilayers. Convert any with:

```python
from openmm.app import PDBFile
from openmm.unit import angstrom
from ase import Atoms
from ase.io import write
import numpy as np, os

CONDA = os.environ['CONDA_PREFIX']
DATA = f'{CONDA}/lib/python3.12/site-packages/openmm/app/data'

for lipid in ['POPC', 'POPE', 'DMPC', 'DLPC', 'DLPE', 'DOPC']:
    pdb = PDBFile(f'{DATA}/{lipid}.pdb')
    pos = np.array([[p.value_in_unit(angstrom) for p in v] for v in pdb.getPositions()])
    elements = [a.element.symbol for a in pdb.topology.atoms()]
    box = np.array([v.value_in_unit(angstrom) for v in pdb.topology.getPeriodicBoxVectors()])
    atoms = Atoms(symbols=elements, positions=pos, cell=box, pbc=True)
    atoms.wrap()
    write(f'{lipid.lower()}_membrane.xyz', atoms, format='extxyz')
    print(f'{lipid}: {len(atoms)} atoms')
```

| Lipid | Atoms | Notes |
|-------|------:|-------|
| DPPC | 28,160 | ← current |
| POPC | 32,512 | Most common in MD |
| POPE | 31,360 | E. coli membrane |
| DMPC | 26,624 | Shorter tails |
| DLPC | 25,088 | Even shorter |
| DLPE | 27,776 | PE headgroup |
| DOPC | 33,024 | Unsaturated |

## Manual Download Required

### GPCRmd (membrane PROTEIN + lipid + water, 50-100k atoms)

Membrane protein systems from published MD simulations. Adds protein non-uniformity to membrane layering.

1. Browse https://www.gpcrmd.org/
2. Select a system (e.g., β2-adrenergic receptor)
3. Download GRO + XTC files
4. Extract 1 frame and convert:

```python
import MDAnalysis as mda  # pip install MDAnalysis
from ase import Atoms
from ase.io import write
import numpy as np

u = mda.Universe('system.gro', 'trajectory.xtc')
u.trajectory[-1]  # last frame
pos = u.atoms.positions
elements = [mda.topology.guessers.guess_atom_element(name) for name in u.atoms.names]
box = u.dimensions[:3]
atoms = Atoms(symbols=elements, positions=pos, cell=np.diag(box), pbc=True)
write('gpcr_membrane.xyz', atoms, format='extxyz')
```

**Requires**: MDAnalysis (`pip install MDAnalysis`) and manual download from website.
