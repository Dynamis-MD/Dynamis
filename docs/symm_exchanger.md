# CUDA-graph-safe halo exchangers (`SymmMemExchanger`, `PaddedSymmMemExchanger`)

Modules: `dynamis_model/comm/symm_exchanger.py`, `dynamis_model/comm/symm_exchanger_padded.py`.
Check script: `benchmarks/uma_fused/test_padded_exchanger.py`.

## Why

A whole-call CUDA graph (one graph per model call, W > 1 ranks) has to replay the halo exchange. The existing
single-node `GPUDirectExchanger` (CUDA IPC) can be captured but is **not replay-safe**: each push syncs with
`cuStreamWriteValue32(own counter, g)` + `cuStreamWaitValue32(peer counters >= g)`, where `g` is a host-side
`++call_counter_`. Capture freezes `g` into the graph's stream mem-op nodes, so from the second replay on the wait is
already satisfied and a late peer's ghost rows are read stale (previous step's values). `PythonNCCLExchanger` cannot
be captured at all: it syncs with the host on every push to get the `all_to_all` splits.

## SymmMemExchanger

* Buffers come from `torch.distributed._symmetric_memory.empty` + `rendezvous`. The forward buffer has rows
  `[local | ghosts]`; the backward buffer holds the gradient rows for this rank's send list.
* Forward push: copy local rows into the own buffer, then `index_copy_` each send list straight into the peer's buffer
  (`hdl.get_buffer(peer)`) at the peer's ghost offsets. Sync uses `hdl.barrier(channel=0)`, which runs on the device
  (signal pads), so nothing depends on a host counter. Output is `[n_local + n_ghost, F]`, the same layout as
  `GPUDirectExchanger`: ghosts ordered by source rank.
* Backward: copy each ghost slice into the owner's gradient buffer, barrier, then `index_add_` onto the local rows.
* Pooled (Exchanger API of #148): one instance per model, sized to `max_feature_dim`. A push of width `F` uses a
  `[rows, F]` view of the same buffer. Consecutive pushes reuse that buffer, so every push **starts** with an extra
  barrier: a peer must have consumed push k before this rank writes push k+1.
* After setup, a push is only stream-ordered torch ops on fixed buffers. It has no host sync, so it can be captured
  and gives correct results on replay.
* Re-setup (halo rebuild or repartition) keeps the symmetric buffers while they are large enough. They are allocated
  at 1.25x the largest row count over all ranks (+8). The keep-or-reallocate decision uses all-gathered sizes, so
  every rank takes the same branch and the collective `rendezvous` stays matched. Every re-setup bumps `generation`
  because the index tensors and row counts change, so a captured graph that includes it in its key must re-capture.

## PaddedSymmMemExchanger

Goal: a halo rebuild (same partition, new ghost lists) must not change anything that a captured graph has baked in.

* **Fixed per-peer ghost slots.** Each directed pair (q -> r) gets a capacity `cap[q, r] >= count`. The forward
  output on rank r is `[local | slot from q0 | slot from q1 | ...]` and `n_full = n_local + sum_q cap[q, r]`. The
  destination ranges and backward slices therefore depend only on `n_local` and the capacities, so they are static.
* **Headroom.** `cap = ceil(count * GHOST_HEADROOM) + 16` with `GHOST_HEADROOM = 1.3` by default (env
  `DYNAMIS_PAD_HALO_HEADROOM`). Measured on CH4 2k/16k with W=2: ghost counts change by at most 2 % per rebuild but
  drift up 32-59 % over 1000 steps. With 1.3 there was at most one re-allocation per direction per 1000 steps; with
  1.10 there were 2-3.
* **Dummy entries.** The padded send index is a persistent tensor whose *content* is rewritten in place at every
  re-setup. Entries past the real count point at local row 0, so pad ghost rows hold a copy of a real row. Callers
  must keep pad rows out of the neighbour list. The exchanger exposes `real_rows`, `n_real` and `pad_rows` so that
  the caller can build the radius graph on real rows and remap it. Pad rows then have zero gradient, and the dummy
  backward entries add exactly +0.0 to row 0.
* **Re-capture rules.** Capacities are kept while every pair fits and `n_local` is unchanged on every rank. In that
  case only the index content is rewritten: `index_generation` += 1 and the graph stays valid. Otherwise the
  capacities are recomputed, buffers are re-allocated if needed, and `generation` += 1, which forces a re-capture. The
  decision uses all-gathered data, so all ranks re-capture at the same call.
* `begin_setup_round()` / `end_setup_round()` (optional, call them on every rank) share the all-gather and the derived
  index maps across the exchangers that one `initialize_buffers` sets up with the same `partition_info`.

## Limitations

* Single node only: symmetric memory needs NVLink/P2P between the ranks. Multi-node graph mode has no replay-safe
  exchanger yet.
* fp32 features only.
* Re-allocating the buffers (`symm_mem.empty` + `rendezvous`) is collective and slow: about 1.3 s per halo rebuild
  for a UMA model when it happened on every rebuild, versus about 5-9 ms with buffer reuse. It also invalidates
  captured graphs. Headroom keeps it rare but cannot avoid it after large growth or a repartition that changes
  `n_local`.
* All ranks must replay their graphs in lockstep (same number of pushes and barriers).
* No Python-side drift-flag piggyback. The Simulator falls back to its `all_reduce` drift mode.

## Enabling / disabling

This PR adds the classes only. Pass one explicitly, for example `UMAModel(..., exchanger=SymmMemExchanger)`. The
follow-up UMA integration (`pr/uma-fused-integration`) wires them in as follows:

* `UMAModel(fused_graph=True)` with W > 1 on one node defaults to `PaddedSymmMemExchanger`.
  `DYNAMIS_PAD_HALO=0` selects the unpadded `SymmMemExchanger` instead.
* `GPUDirectExchanger` stays the default for eager/per-op runs and W = 1. Graph mode refuses it with a
  `RuntimeError`; `DYNAMIS_UNSAFE_GRAPH_EXCHANGER=1` overrides this for benchmarking only.
* `DYNAMIS_PAD_HALO_HEADROOM` (default 1.3) sets the ghost-slot headroom.

## Verification

* Replay safety (`benchmarks/uma_fused/dist_capture_test.py`, added with the UMA benchmarks PR; 2 ranks, F = 1152; one forward push captured; the last rank sleeps inside its graph before pushing;
  6 replays with new inputs). Max ghost error per replay: `GPUDirectExchanger` 0 1 1 1 1 1 (stale from replay 2 on),
  `SymmMemExchanger` 0 0 0 0 0 0.
* `torchrun --nproc_per_node=2 benchmarks/uma_fused/test_padded_exchanger.py`: padded vs unpadded, eager and with one
  graph replayed across 5 re-setups (the last one overflows the capacity and re-captures), plus pooled widths
  3/1/64/3. Results: forward bitwise equal, backward < 1e-6, ALL OK.
* Full UMA model, CH4, W=2, TF32. Forces with `SymmMemExchanger` vs `GPUDirectExchanger` agree to 4.0e-4 relative,
  and vs W=1 to 6.5e-4 (noise level). Per-layer push 0.21-0.32 ms vs 0.71-0.86 ms; backward push 0.04 vs 0.4 ms.
* 1000-step NVE MD, W=2, whole-call graph, padded exchanger with headroom 1.3. Graph captures drop from 115 to 2
  (2k, repartition every 250 steps) and from 129 to 4 (16k). Mean step: 21.0 -> 15.6 ms (2k) and 45.2 -> 39.3 ms
  (16k). Forces vs unpadded agree to 3.7e-4 relative; energy conservation is unchanged.
