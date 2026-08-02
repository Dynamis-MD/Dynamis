# DistMACE Final Adapter

------------------------------------------------------------

## Overview

`DistMACEAdapter` is the current presentation-ready single-layer distributed
MACE adaptation. It combines:

- structural halo reconstruction
- one internal learned-feature exchange layer
- distributed communication for that layer
- ownership-aware local energy
- autograd-recovered owned local forces
- centralized validation for halo and no-halo runs

The main implementation lives in:

- `final_adapter.py`
- `mace_subclass_boundary.py`

------------------------------------------------------------

## How The Prototype Phases Built Up To This Adapter

The final adapter is a direct consolidation of the earlier prototype phases,
not a disconnected rewrite.

1. `single_gpu_adapter.py`
   This established the basic DistNNP-to-MACE contract on one rank. It proved
   that a `DistNNPState` could be translated into public MACE inputs and that
   MACE could satisfy the DistNNP energy/forces interface at all.

2. `phase1_raw_exchange_adapter.py`
   This added the first true distributed bridge:
   local owned atoms -> local plus halo reconstruction -> standard MACE
   forward. It also introduced the reusable structural exchange mechanics that
   the final adapter still inherits through `MACEPotentialModelWithRawExchange`.

3. `phase2_owned_energy_adapter.py`
   This introduced the ownership-aware scalar objective, moving away from raw
   total MACE energy toward an owned local energy rule.

4. `phase3_owned_force_adapter.py`
   This rebuilt the force path around autograd on the owned local energy,
   proving that local forces should be recovered from backward rather than
   sliced from the raw MACE force output.

5. `phase4_internal_feature_exchange_adapter.py`
   This validated one internal learned-feature exchange point inside the MACE
   interaction stack, specifically after `linear_up(...)` and before
   `conv_tp(...)`.

6. `final_adapter.py`
   This merges those ingredients into one stable adapter surface:
   `DistMACEAdapter` orchestrates the full distributed path while
   `DistMACEModelSubclassBoundary` owns the internal model seam.

------------------------------------------------------------

## Final Adapter Structure

- `DistMACEAdapter`
  DistNNP-facing orchestrator. It owns the end-to-end flow and remains the
  only object the DistNNP side should call directly.

- `DistMACEModelSubclassBoundary`
  Dedicated internal MACE seam. It installs the subclass boundary on the
  selected interaction block and owns the distributed execution context for
  the internal feature exchange.

- `StructuralHaloBuilder`
  Builds the differentiable local-plus-halo structural view from
  `DistNNPState`.

- `MACEGraphBuilder`
  Builds the public-MACE batch dictionary while reinjecting live positions to
  preserve autograd.

- `OwnedEnergyHead`
  Applies the current ownership rule:
  `outputs["node_energy"][:num_local].sum()`.

- `OwnedForceRecoverer`
  Computes owned local forces from autograd on the owned scalar energy.

- `ValidationDiagnostics`
  Reports the adapter’s distributed, ownership, exchange, and gradient state.

------------------------------------------------------------

## The Communication Layer We Enabled

The current policy is frozen in `FeatureExchangePolicy`:

- `name = "layer_1_only"`
- `interaction_layer = 1`
- `boundary = "post_linear_up_pre_conv_tp"`

Code references:

- Policy definition:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:42)

- Adapter creates and installs the boundary:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:309)

- Boundary chooses the target interaction block by `interaction_layer`:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:62)

What this means in practice:

- We are not exchanging raw coordinates at every message-passing stage.
- We exchange one learned node-feature tensor at the chosen internal MACE
  interaction block.
- The exchanged tensor starts from the owned local rows only, then is rebuilt
  as local plus halo through `FeaturePushToExchange`.

------------------------------------------------------------

## How Subclassing Achieves The Internal Exchange

The key Step 2 decision was to stop treating internal exchange as a long-term
 adapter-side patch and move it into a dedicated subclass boundary.

Code references:

- Boundary class:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:30)

- Dynamic subclass creation for the model root:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:44)

- Dynamic subclass creation for the selected interaction block:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:149)

- Installed subclass metadata exposed back to diagnostics:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:184)

How it works:

1. `DistMACEAdapter` constructs `DistMACEModelSubclassBoundary`.
2. The boundary selects interaction block `1`.
3. It creates a subclass of that block’s original class.
4. That subclass overrides `handle_lammps(...)`.
5. Inside the override, when distributed IPC exchange is active, it:
   - slices the owned local feature rows
   - exchanges them with `FeaturePushToExchange`
   - reconstructs the local-plus-halo feature tensor
   - returns the exchanged tensor into the MACE interaction flow

This is why subclassing matters here: the internal exchange point now lives at
the model boundary itself, rather than being treated as incidental adapter
glue.

------------------------------------------------------------

## Why `handle_lammps(...)` Matters

LAMMPS is useful here less because we are targeting LAMMPS directly today and
more because MACE already exposes an internal control point named
`handle_lammps(...)` on the interaction block. That hook gives us a natural
model-internal seam without rewriting the entire interaction stack.

Code reference:

- Override site:
  [mace_subclass_boundary.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/mace_subclass_boundary.py:89)

Relevant influence on the implementation:

- If `lammps_class is not None`, we defer to the original implementation.
- If we are not in distributed IPC mode, we leave features untouched.
- Only when the distributed execution context is active do we replace the
  learned local features with exchanged local-plus-halo features.

This is valuable because:

- it preserves the original MACE/LAMMPS pathway instead of breaking it
- it gives us a narrow, model-internal override point
- it avoids a much larger rewrite of the interaction block logic

So in this adapter, the presence of LAMMPS support in MACE effectively gives us
an already-existing hook surface that is convenient for distributed
instrumentation.

------------------------------------------------------------

## Ownership And Force Recovery

Ownership-aware energy:

- `OwnedEnergyHead` is defined here:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:146)

- Current rule:
  `owned_local_energy = outputs["node_energy"][:num_local].sum()`

Force recovery:

- `OwnedForceRecoverer` is defined here:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:175)

- Gradient path:
  it differentiates owned local energy with respect to both local positions and
  exchanged positions, then reports halo-gradient activity as part of the
  diagnostics.

------------------------------------------------------------

## Validation Surface

The milestone validation is centralized in:

- `ValidationDiagnostics`:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:223)

- Milestone summary logic:
  [final_adapter.py](/W:/Building/Summer%20Research/DistNNP/DistNNP/distnnp_model/src/distnnp_model/nnp/MACE_Adaptation/final_adapter.py:403)

Important outputs include:

- `subclass_boundary_installed`
- `subclass_boundary_executed`
- `feature_exchange_used_ipc`
- `halo_gradient_active`
- `milestone_objectives_satisfied`

This is what makes the halo and no-halo runs interpretable as a weekly
milestone rather than just raw debugging output.

------------------------------------------------------------

## Extension To Multiple Layers

Yes, this design can extend to multiple layers.

Why:

- The current single-layer schedule is already represented as a policy.
- The internal boundary is separated into its own module.
- Structural halo preparation is independent of the exchange schedule.
- Ownership and force recovery are downstream of the policy choice.

What would need to grow:

- `FeatureExchangePolicy` must support more than `layer_1_only`
- `DistMACEModelSubclassBoundary` must instrument multiple interaction blocks
- validation must report per-layer execution and shapes

The important point is that multi-layer support looks like a policy-and-
instrumentation extension, not a fresh architectural rewrite.

------------------------------------------------------------

## Execution Commands

Halo:

```bash
torchrun --standalone --nproc_per_node=2 -m distnnp_model.nnp.MACE_Adaptation.final_adapter --builtin-system adapter_boundary_halo --model-family mp --model medium-mpa-0 --device cuda
```

No halo:

```bash
torchrun --standalone --nproc_per_node=2 -m distnnp_model.nnp.MACE_Adaptation.final_adapter --builtin-system adapter_xyz6 --model-family mp --model medium-mpa-0 --device cuda
```

Halo JSON:

```bash
torchrun --standalone --nproc_per_node=2 -m distnnp_model.nnp.MACE_Adaptation.final_adapter --builtin-system adapter_boundary_halo --model-family mp --model medium-mpa-0 --device cuda --output-json build/distmace_single_layer_halo.json
```

No-halo JSON:

```bash
torchrun --standalone --nproc_per_node=2 -m distnnp_model.nnp.MACE_Adaptation.final_adapter --builtin-system adapter_xyz6 --model-family mp --model medium-mpa-0 --device cuda --output-json build/distmace_single_layer_no_halo.json
```
