# MACE Adaptation Baselines

## Step 1 status

This folder currently freezes two baseline paths that should remain available
while the first distributed MACE prototype is built:

1. `single_gpu_adapter.py`
   This is the single-rank DistNNP-to-MACE contract baseline.
   It proves that a `DistNNPState` can be translated into public MACE graph
   inputs and that MACE can satisfy the DistNNP energy/forces model contract
   without any distributed communication.

2. `phase1_raw_exchange_adapter.py`
   This is the raw distributed structural-wrapper baseline.
   It proves the outer distributed bridge only:
   `DistNNPState -> local partition -> local plus halo reconstruction -> MACE`.
   It is intentionally not the final distributed MACE prototype.

## Immediate prototype target

All next implementation work should aim at one coherent prototype with the
following three capabilities:

1. Owned local energy
   The rank-local scalar energy must include owned atoms only and avoid halo
   double counting.

2. Local force recovery through autograd-aware exchange
   Owned local forces should be recovered from the owned scalar energy, not by
   slicing the returned MACE force tensor.

3. One internal learned-feature exchange point
   The prototype must include one concrete model-internal feature exchange
   insertion point so the adaptation moves beyond raw structural exchange.

## What should stay stable

- Keep `single_gpu_adapter.py` usable as the simplest contract-check path.
- Keep `phase1_raw_exchange_adapter.py` usable as the outer distributed wrapper
  baseline.
- Do not collapse all new prototype logic into either baseline file unless the
  change clearly preserves that file's original debugging role.

## Recommended implementation sequence after Step 1

1. Stabilize the raw distributed wrapper as the outer shell.
2. Identify the MACE boundary for ownership-aware local energy.
3. Rebuild local force recovery around backward on owned energy.
4. Choose one MACE internal learned-feature tensor worth exchanging.
5. Assemble a new prototype-focused implementation path beside the baselines.

## Validation criteria for Step 1

Step 1 is considered successful if:

- baseline responsibilities are explicit,
- the prototype target is fixed and easy to reference,
- and future changes can be judged against the baseline/prototype split.

## Final adapter

The consolidated final-adapter documentation now lives in:

- `FINAL_ADAPTER_README.md`

That document covers:

- how the prototype phases built up to the final adapter
- the subclass boundary and exchanged layer
- the role of the `handle_lammps(...)` hook
- ownership-aware energy and autograd-based force recovery
- current milestone validation and execution commands
