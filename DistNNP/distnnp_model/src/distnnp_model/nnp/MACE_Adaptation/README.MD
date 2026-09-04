# DistMACE Thin Adapter: Design and Support Model

## Purpose

`DistMACEThinAdapter` is the supported bridge between DistNNP's distributed
atom ownership model and MACE's `ScaleShiftMACE` inference path. Its purpose is
not to reimplement partitioning or to modify a loaded MACE model. Instead, it
keeps the boundary narrow:

- DistNNP owns partitioning, halo membership, peer order, and repartitioning.
- The adapter converts the current owned-plus-halo state into the minimum MACE
  graph required to evaluate rank-owned atoms.
- MACE owns its learned representation, interaction/product blocks, readouts,
  and scale/shift semantics.
- The adapter returns rank-owned potential energy and forces derived from that
  same owned energy.

The implementation is deliberately explicit rather than subclass-based. It
does not mutate `model.__class__`, install diagnostics in the forward path, or
expose execution-mode switches in production inference.

## Adapter contract

The adapter accepts a raw, public MACE foundation model loaded by
`load_raw_mace_model(...)`. It requires the `ScaleShiftMACE`-compatible surface:
`r_max`, `atomic_numbers`, node/edge embeddings, matched `interactions` and
`products`, `readouts`, `atomic_energies_fn`, and `scale_shift`.

The adapter supports either one final readout or one readout per interaction
layer. It resolves a requested MACE head at construction time; ambiguous
multi-head models must name a head explicitly. `atomic_number_map` is an
optional input-side remapping mechanism, not a conversion of model chemistry.
Every exchanged species must map to exactly one atomic number supported by the
loaded checkpoint.

The public return contract is:

```text
owned_energy, owned_forces = adapter(state)
```

`owned_energy` is local to a rank. Callers that need a system energy must sum
it once across ranks. `owned_forces` corresponds only to rank-owned atoms in
the current local ordering. Static parity preserves MACE units (eV and
eV/Angstrom); the trajectory harness presently performs its own force-unit
conversion for the DistNNP Verlet integrator.

## Design phases

### 1. Buffer initialization and partition validation

`initialize_buffers(...)` is required after initial partitioning and after
every repartition. It validates DistNNP's peer-grouped CSR send/receive
metadata, including offset monotonicity, send-index bounds, and contiguous
ghost-slot layout. It then recreates position, atom-type, and feature
exchangers and invalidates static caches.

This is a deliberate correctness boundary: a state carrying a new partition
cannot be evaluated with exchangers or cached attributes built for an older
partition. The adapter records the partition-object identity and rejects stale
use rather than attempting to infer a safe update.

### 2. Static species and head preparation

On the first forward for a partition, `_build_static_partition_cache(...)`
exchanges atom types once, applies optional atomic-number remapping, and builds
MACE node attributes, batch metadata, pointer metadata, head index, expected
row count, and owned-row count. These values depend on partition membership and
species, not positions, so caching keeps them out of the steady-state path.

The cache is invalidated by `initialize_buffers(...)`. A workflow that changes
species without rebuilding partition buffers is outside the current contract.

### 3. Structural view and graph construction

Each call clones owned positions into a floating-point autograd leaf and
exchanges positions to construct `[owned | halo]` rows. DistNNP's periodic
radius-graph implementation produces edge indices and cell offsets. The adapter
retains only edges whose destination is owned; halo atoms remain valid source
nodes but never receive rank-owned output responsibility.

This owned-destination filtering is the central ownership decision. It prevents
duplicate node-energy accumulation across ranks while preserving all geometric
senders required by local MACE messages. The MACE batch dictionary keeps the
complete structural view but supplies this filtered edge list.

For nonperiodic fixtures, `canonicalize_nonperiodic_geometry(...)` can place
coordinates in a padded orthorhombic cell before halo construction. It is a
test/input preparation utility, not a second graph or neighbor-list policy.

### 4. Explicit distributed MACE forward

`ThinDistScaleShiftMACERuntime.forward(...)` mirrors the relevant
`ScaleShiftMACE` stages:

1. Prepare graph vectors, lengths, heads, and atom ranges with MACE's
   `prepare_graph` helper.
2. Compute baseline atomic energies, node embedding, spherical harmonics,
   radial features, and optional pair repulsion.
3. Run each interaction/product pair.
4. Exchange the owned feature rows after every non-final product block, then
   reconstruct the halo feature rows through DistNNP communication.
5. Apply the configured readout(s), scale/shift the learned contribution, add
   baseline energy, and sum only owned node energies.

The exchange boundary is after `product(...)` because later interaction layers
need remote many-body features, while the first interaction can be evaluated
from exchanged structure alone. The runtime lazily sizes its feature exchanger
from the first node-feature tensor and asserts that later layers keep the same
width.

### 5. Force recovery and lifecycle cleanup

The adapter differentiates only `owned_local_energy` with respect to the owned
position leaf and returns its negative gradient. This preserves force ownership
and avoids MACE's broader force/stress/virial output path. Runtime state is
installed immediately before forward and always cleared in `finally`, including
when MACE raises.

## Design choices and consequences

| Choice | Rationale | Consequence |
| --- | --- | --- |
| Composition over dynamic subclassing | Preserve loaded-model identity and a small inspectable hot path. | New MACE forward-schema changes must be reflected explicitly in the runtime. |
| Owned-destination graph edges | Each rank computes only the energy it owns. | Global energy requires caller-side reduction. |
| Position exchange with autograd connectivity | Halo-dependent energy contributes correctly to owned-position forces. | The exchanged structural view is rebuilt every call. |
| Feature exchange after non-final products | Later MACE layers see remote updated many-body features. | The current implementation assumes a stable 2-D feature width. |
| Cached species-side tensors | Avoid repeat type exchange and one-hot construction. | `initialize_buffers(...)` is mandatory after repartition. |
| No diagnostics in the hot path | Keep production behavior and benchmarks focused. | Deep stage inspection belongs to the experimental top-down fallback. |

## Family support

The loader recognizes public `mp`, `off`, `omol`, and `polar` foundation-model
families. Recognition is not equivalent to thin-adapter support: the adapter
only supports a family/checkpoint once its raw model satisfies the contract
above and passes static parity plus distributed trajectory validation.

Current supported target: MP/MPA `ScaleShiftMACE` checkpoints, including the
standard `medium-mpa-0` trajectory baseline. The thin runtime is intentionally
not yet a generic abstraction over arbitrary MACE architectures, specialized
conditioning paths, stress/virial output, or changing per-layer feature widths.

`off` and `polar` remain loadable candidates, not documented support claims,
until they have the same contract and validation evidence. This distinction
prevents a successful public-calculator load from being mistaken for distributed
semantic support.

## Immediate OMOL support plan

OMOL is the next family to qualify because the loader already exposes
`mace_omol`. Qualification should proceed without adding family-specific
branches to the thin hot path unless the raw model contract proves different.

1. **Characterize a raw OMOL checkpoint.** Load the intended checkpoint with
   `return_raw_model=True`; record its Python class, `heads`, supported atomic
   numbers, cutoff, interaction/product/readout counts, feature shapes, and
   available pair-repulsion behavior. Compare these with the current
   `ScaleShiftMACE` contract.
2. **Establish molecular fixtures.** Select an OMOL-supported molecular fixture
   containing only checkpoint-supported elements, plus a nonperiodic halo case.
   Canonicalize nonperiodic geometry before partitioning and retain the exact
   fixture and head in reference-artifact metadata.
3. **Prove static semantics first.** Generate detached calculator and raw-core
   references, then require two-rank strict energy/force parity, global vertex
   ownership assembly, and buffer-reinitialization parity.
4. **Run trajectory qualification.** Use the existing trajectory architecture
   only after the monolithic OMOL calculator can serve as the singleton oracle.
   Confirm force-unit handling, head selection, metadata equality, early-frame
   agreement, RMSD schedule, drift, and RDF behavior.
5. **Promote only after parity.** If OMOL passes unchanged, list its exact
   checkpoint/head/fixture as supported. If its raw forward schema differs,
   introduce a small named runtime-contract adapter outside the steady-state
   communication logic; do not add scattered `model_family == "omol"` branches.

The immediate acceptance criterion is therefore not merely successful loading:
it is unchanged ownership semantics and independent calculator/core-model
parity for a documented OMOL checkpoint.

## Experimental top-down fallback

`DistMACETopDownAdapter` is retained for diagnostic investigation only. It
modifies the raw model's runtime class and carries richer observation and mode
machinery. It must not share a raw model instance with the thin adapter in one
process. It is not the supported production path, benchmark interface, or
default source of correctness claims.

For commands, SLURM variables, artifact locations, and test orchestration, see
`distnnp_model/tests/mace/README.md`.
