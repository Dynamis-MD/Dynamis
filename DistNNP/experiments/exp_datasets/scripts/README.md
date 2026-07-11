# Dataset Generation & Download Scripts

Datasets are **not committed to git** (they were ~350 MB). Instead they are
obtained on demand — generated locally where that is reproducible, downloaded
and checksum-verified where it is not.

## One command

```bash
bash setup_data.sh          # generate CH4 + download PDB + fetch hosted archive
```

Other modes: `--generate-only` (just the CH4 baselines, numpy-only),
`--no-archive` (CH4 + PDB), `--from-scratch` (rebuild everything with the
version-dependent tools; not recommended — see below).

> **CH4 variants.** By default only the **normal-density `splited`** baseline is
> generated (physical spacing, min interatomic distance ~1.09 Å). The `cluster`
> variant — CH4 crammed into a corner so atoms *overlap by design* — is an
> intentional extreme for **partition-imbalance stress only**; it is opt-in
> (`python 03_generate_ch4.py --variants cluster`), not shipped, and must never
> be fed to an energy model.

## Reproducibility tiers

The datasets fall into four tiers by *how reproducible they are*, and each is
obtained the way that best fits:

| Tier | Data | How | Reproducible? |
|---|---|---|---|
| **1** | `baselines/` CH4/O2 | **generate** (`03_generate_ch4.py`) | **Yes — bit-exact** on any platform / NumPy version |
| **2** | `pdb/` raw structures | **download** RCSB (`01_download_pdbs.sh`) | Yes — same bytes by accession (RCSB may remediate; checksums informational) |
| **3** | `protein_vacuum/`, `membrane/`, `protein_solvated/dhfr`, `capsid/` | **download** hosted archive (`download_datasets.sh`) | Protein/membrane: no across tool versions → shipped as a checksummed `.tar.xz`. Capsid: reproducible from RCSB (`07`) but rides the archive for convenience |
| **4** | `*_density.xyz` large solvated systems | user-provided (`../data/` symlinks) | External — regenerate with 04/06 or provide separately |

### Why Tier 1 can be generated but Tier 3 cannot

`03_generate_ch4.py` was rewritten to be **version-independent**. It hardcodes
the rigid CH4 (tetrahedron, C–H 1.09 Å) and O2 (1.21 Å) geometry, places
molecules on a deterministic grid, and draws only:

* IEEE-754 correctly-rounded ops (`+ − × ÷ √`) for coordinates,
* integer-only RNG (`RandomState.permutation` / `randint`) for the grid shuffle
  and orientation choice,
* exact signed-permutation rotation matrices (entries −1/0/1) — **no
  sin/cos/exp/log on the position path**.

Result: **positions are bit-identical everywhere.** Momenta use a *separate*
frozen `RandomState` stream (`standard_normal`); they are identical across
NumPy versions and may differ only in the last ULP across libm — physically
irrelevant, since MD re-thermalizes velocities.

Tier 3 (`02`/`05`) depends on **pdbfixer** hydrogen-placement templates and
**OpenMM** solvation/equilibration, whose output changes between releases (and,
for OpenMM, between CPU/GPU). It cannot be reproduced byte-for-byte from a
script, so it is distributed as a checksummed archive instead of regenerated.

## Hosting the Tier 3 archive (shared cache + GitHub Release)

`download_datasets.sh` fetches `distnnp_exp_datasets_v1.tar.xz` (~23 MB, now
includes the HIV capsid) and verifies it against `checksums.txt`. Sources are
tried in order, and the **first that passes the checksum wins**:

1. `DISTNNP_ARCHIVE_LOCAL` — an explicit local copy (never downloaded/deleted).
2. **Shared cache** — `DISTNNP_ARCHIVE_SHARED_DIR/<archive>`
   (default `/projects/bcpt/public/distnnp_datasets/`). On Delta the archive
   lives here so the download happens **once for the whole team**; a fresh
   download is cached back automatically. Only a miss (or a stale copy) hits the
   network. Seed it manually with:
   ```bash
   mkdir -p /projects/bcpt/public/distnnp_datasets
   cp distnnp_exp_datasets_v1.tar.xz /projects/bcpt/public/distnnp_datasets/
   ```
3. `gh release download` — authenticated (required while `szaman19/DistNNP` is
   **private**: release assets inherit repo visibility, so an unauthenticated
   wget 404s).
4. `wget` from the Release URL — works only once the asset is public.

The archive is distributed as a **GitHub Release asset** — it lives with the
repo and has a stable URL, but is **not** committed to git history (so
`git clone` stays lean). Create it once:

```bash
gh release create datasets-v1 distnnp_exp_datasets_v1.tar.xz \
    --title "DistNNP experiment datasets v1" \
    --notes "pdbfixer/OpenMM protein/membrane + HIV capsid XYZ (Tier 3). SHA256 in checksums.txt."
```

Override any source:

```bash
export DISTNNP_ARCHIVE_REPO="owner/repo"       # different fork
export DISTNNP_ARCHIVE_TAG="datasets-v2"       # different release tag
export DISTNNP_ARCHIVE_URL="https://.../distnnp_exp_datasets_v1.tar.xz"  # full URL
export DISTNNP_ARCHIVE_LOCAL="/path/to/distnnp_exp_datasets_v1.tar.xz"   # local copy, no download
export DISTNNP_ARCHIVE_SHARED_DIR="/path/to/shared"   # shared cache ("" disables)
```

(A DOI-citable copy on Zenodo/HF is optional and only needed for paper
citation; the download path above does not require it.)

Rebuild the archive (if the Tier 3 inputs change) with:

```bash
tar -C ../ -cf - protein_vacuum membrane/apoa1_92k_openmm.xyz \
    protein_solvated/dhfr_23k_openmm.xyz capsid \
    | xz -9 -T0 > distnnp_exp_datasets_v1.tar.xz
sha256sum distnnp_exp_datasets_v1.tar.xz    # update checksums.txt (+ refresh the shared cache)
```

## Dependencies

```bash
pip install numpy                    # 03 (Tier 1) — the ONLY hard requirement
pip install ase scipy                # 04/06 (Tier 4 regeneration)
conda install -c conda-forge pdbfixer openmm openmmtools   # 02/05 (Tier 3 rebuild)
```

`03_generate_ch4.py` no longer needs **rdkit** (the old conformer dependency was
dropped). Tiers 2 and 3 need only `wget` + `tar`/`xz` for the default download
path.

## Scripts

| Script | Tier | Description |
|--------|------|-------------|
| `setup_data.sh` | all | One-click: generate + download + verify |
| `03_generate_ch4.py` | 1 | **Version-independent** CH4/O2 generator (numpy only) |
| `01_download_pdbs.sh` | 2 | Download raw PDB/CIF from RCSB + OpenMM |
| `download_datasets.sh` | 2+3 | PDB + hosted archive, SHA256-verified |
| `checksums.txt` | — | SHA256 manifest for all tiers |
| `02_convert_vacuum.py` | 3 | PDB → vacuum XYZ (pdbfixer) — *version-dependent* |
| `05_generate_openmm.py` | 3 | OpenMM equilibrated systems — *version-dependent* |
| `04_generate_solvated.py` | 4 | Protein + grid water at liquid density |
| `06_generate_electrochemical.py` | 4 | Pt(111) slab + explicit water |
| `07_retrieve_hiv.py` | 3 | HIV-1 capsid (3J3Q) RCSB mmCIF → XYZ + nested shell subsets (stdlib only) |
| `build_all.sh` | all | Regenerate everything from scratch (needs all deps) |

## Quick reference: system sizes

| Dataset file | Atoms | Script | Key parameters |
|---|---:|---|---|
| ch4_Nk_splited.xyz | N×1000 | 03 | **default, normal density** — CH4 x>0 / O2 x<0, box=round(60·(N/8)^⅓) Å, seed 42 |
| ch4_Nk_cluster.xyz | N×1000 | 03 | *opt-in extreme* (`--variants cluster`) — CH4 packed into 30% linear extent, **atoms overlap by design**; imbalance stress only, not pinned |
| 1aki_vacuum_openmm.xyz | 1,960 | 02 | PDB 1AKI, pH 7.0 H-addition |
| 2rik_vacuum_openmm.xyz | 4,377 | 02 | PDB 2RIK |
| groel_vacuum_openmm.xyz | 56,700 | 02 | PDB 1GRL |
| 2w49_vacuum_openmm.xyz | 153,456 | 02 | PDB 2W49 |
| 6p9x_vacuum_openmm.xyz | 22,472 | 02 | PDB 6P9X |
| 9gmo_vacuum.xyz | 228,994 | 02 | PDB 9GMO (from .cif via pdbfixer) |
| dhfr_23k_openmm.xyz | 23,558 | 05 | openmmtools DHFRExplicit |
| 1aki_density.xyz | 25,846 | 04 | 1aki_vacuum + grid water |
| 2rik_density.xyz | 112,727 | 04 | 2rik_vacuum + grid water |
| 9gmo_density.xyz | 1,536,442 | 04 | 9gmo_vacuum + grid water |
| apoa1_92k_openmm.xyz | 92,224 | 05 | OpenMM ApoA1 PDB (pre-equilibrated) |
| dppc_membrane_28k.xyz | 28,160 | 05 | OpenMM built-in DPPC |
| pt_water_12k_density.xyz | 14,236 | 06 | Pt 20×20×4 + 40 Å water |
| pt_water_17k_density.xyz | 21,180 | 06 | Pt 24×24×4 + 40 Å water |
| 3j3q_capsid.xyz | 2,440,800 | 07 | HIV-1 capsid (PDB 3J3Q), heavy atoms only |
| 3j3q_capsid_sub350.xyz | 175,218 | 07 | 3J3Q shell cube L=350 (nested) |
| 3j3q_capsid_sub150.xyz | 33,212 | 07 | 3J3Q shell cube L=150 (nested) |

CH4 sizes N ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} (thousands of atoms).
