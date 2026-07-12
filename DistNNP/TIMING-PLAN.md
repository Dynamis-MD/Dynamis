# DistNNP Timing Design Plan

This document is the plan of record for adding performance timing to DistNNP. It exists so the work can be **split across several people** — each workstream in [Workstreams](#workstreams) is independently ownable once the core `Timer` (W1) lands.

Status: **proposal — nothing implemented yet.** Open questions are collected at the bottom; they should be settled before W2+ starts.

---

## Goals

- Answer the question the project actually cares about: **where does a timestep go, and how much of it is communication?** (CLAUDE.md: "halo exchange volume dominates at scale.")
- Timing is a **library-level facility, not an opt-in the caller bolts on.** A user running a simulation should get timings by flipping one switch — not by wrapping their own code in timer blocks.
- Instrumentation lives **as close to the work as possible**, which in this codebase means **C++ over Python wherever the work is in C++**. Python-side timers around a C++ call measure the wrong thing (see [Why C++](#why-c-and-not-python)).
- **Near-zero cost when off**, and *honest* when on: the act of measuring must not serialize a pipeline that is asynchronous by design.
- Every record is attributable: **rank, process ID, device ID, hostname**.

### Non-goals

- Replacing `nsys` / `torch.profiler` for kernel-level attribution. This is coarse-grained, always-available, in-band accounting — a complement to a profiler, not a substitute.
- A live dashboard. Output is a file; post-processing lives in `post_processing/`.

---

## What gets timed

Two levels. The **component** level answers "which phase is slow"; the **class** level answers "compute or communication".

### Components (one timer each)

| Component | Where it lives | Notes |
|---|---|---|
| `partition` | `distnnp_state` — `SFCPartitioner::partition`, `KDTreePartitioner::partition`, `SimpleSpatialPartitioner::partition` | Time the virtual `partition()` entry point so every partitioner is covered by construction. |
| `build_graph` | `distnnp_state` — `radius_graph_pbc` (`src/csrc/graph/`) | Called both standalone and from inside auto-weighted `partition()`. Nesting must not double-count — see [Nesting](#nesting-and-double-counting). |
| `build_halo_maps` | `distnnp_state` — `AbstractPartitioner::build_halo_maps` | Shared by all partitioners; one instrumentation site covers all. |
| `halo_exchange` | `distnnp_model` — the feature push forward/backward | The headline communication number. |
| `model_forward` | `distnnp_model` — `DistNNPModel.__call__` | The **full model cycle**, per iteration: exchange + graph + energy + backward. |
| `integrator_step` | `distnnp_integrator` — each integrator's `step()` | Includes the halfstep kernels; excludes the model calls it makes (those are attributed to `model_forward`). |
| `iteration` | `distnnp` — the `Simulator` step loop | The denominator everything else is a fraction of. |

### Compute vs. iteration

**The headline number.** `iteration` is the wall time of one full timestep — the denominator. The question is: *what fraction of that was actually spent computing, and what fraction went to everything else the iteration has to do?*

Every timer therefore carries a **class** tag, and the classes partition the iteration:

| Class | What it covers |
|---|---|
| `COMPUTE` | Real math: force/energy evaluation, graph build, integrator kernels |
| `COMM` | Moving data between ranks: halo exchange, collectives |
| `OTHER` | Everything else the iteration costs: partitioning bookkeeping, allocation, Python-level orchestration, framework overhead |

The report we want falls straight out of summing **self-time** (not inclusive time — see [Nesting](#nesting-and-double-counting)) by class:

```
iteration      12.4 ms   100%
  compute       7.1 ms    57%     <-- the number to drive up
  comm          4.2 ms    34%
  other         1.1 ms     9%
```

The three classes sum to the iteration by construction, so any time that isn't attributed to a timed region shows up as an `unaccounted` remainder rather than silently vanishing. If that remainder is large, the instrumentation has a hole in it — which is itself worth knowing.

**One subtlety that must not be glossed over:** a collective's measured duration is *not* pure network time. A rank that reaches an all-reduce early sits inside the collective waiting for the slowest rank — so a naive `COMM` bucket silently absorbs **load imbalance** and reports it as network cost. That is a misleading number to optimize against: it says "reduce halo volume" when the real fix may be "partition better."

Proposal: split it out. Record a `WAIT` class by timing a barrier immediately *before* each collective; the barrier absorbs the imbalance, and the collective that follows measures something much closer to real transport. This costs an extra barrier per collective, so it belongs behind the `detail` level rather than always on. At `basic`, `COMM` means "comm + imbalance" and must be documented as such.

### Levels

One knob, three settings, so the cost of measuring scales with how much you asked for:

| Level | Records | Extra sync/barriers |
|---|---|---|
| `off` (default) | nothing | none |
| `basic` | `iteration` + the component timers | none beyond what the code already does |
| `detail` | adds `WAIT` barriers, per-rank per-step records | yes — perturbs, use deliberately |

---

## Why C++ (and not Python)

Two independent reasons, and both matter:

1. **A Python timer around a C++/CUDA call measures the wrong quantity.** CUDA work is asynchronous: `radius_graph_pbc(...)` returns as soon as the kernels are *launched*, not when they *finish*. A `time.perf_counter()` (or a `torch.cuda.Event` pair placed in Python) around that call therefore measures launch overhead, or — if you insert a `synchronize()` to fix it — measures the correct duration *while destroying the pipelining you are trying to evaluate*. The timer has to be inside the C++ function, on the same stream as the work.

2. **Library-level means the caller does nothing.** If timing is a set of `with Timer(...)` blocks in Python, then every new call site, experiment script, and downstream user has to remember to add them — and the ones that forget are silently unmeasured. Putting the timer inside `partition()` / `build_halo_maps()` / `radius_graph_pbc` means *every* caller is instrumented, forever, including ones not written yet.

Python-side timers remain appropriate for the handful of things that genuinely *are* Python: the `Simulator` iteration boundary and the model's Python-level orchestration. The Python `Timer` should be a thin binding over the same C++ registry, not a second implementation — one registry, one output file, one set of semantics.

> The UMA model in `../refs/` has a timing implementation worth reading for reference, but it is **not** the shape we want — it is bolted on at the call sites rather than living in the library.

---

## Core design

### The timer

```cpp
namespace distnnp_state {

enum class TimerClass : uint8_t { Compute, Comm, Wait, Other };

class Timer {
public:
    // Records into the global registry under `key`. Non-copyable, RAII:
    // the destructor stops it, so an early return or a throw cannot leak
    // a half-open timer.
    Timer(std::string_view key, TimerClass cls, cudaStream_t stream = nullptr);
    ~Timer();

    static void set_level(Level lvl);
    static bool enabled();          // inlined; the whole class no-ops when off
    static void set_step(uint64_t step);   // stamps subsequent records
};

} // namespace distnnp_state
```

Usage inside the library — one line, no ceremony, no way to forget to stop it:

```cpp
PartitionMessagePassingInfo SFCPartitioner::partition(const MolecularInfo &state,
                                                     const DistEnvironment &env)
{
    DISTNNP_TIME("partition", TimerClass::Compute);   // macro → compiles to nothing when disabled
    ...
}
```

### The high-precision host clock — use `steady_clock`, *not* `high_resolution_clock`

This is the C++ high-precision timer, and there is a trap in the naming worth stating plainly, because the obvious choice is the wrong one.

The clock actually *called* "high resolution" is `std::chrono::high_resolution_clock` — but it is only a typedef, and **on libstdc++ (our toolchain) it is an alias for `system_clock`**. `system_clock` is wall-clock time, subject to NTP discipline, and can jump **backwards** mid-run — which produces *negative* durations in a long job on a cluster whose nodes are NTP-corrected. The clock with "high resolution" in its name is the one that can silently corrupt our measurements.

Use **`std::chrono::steady_clock`**:

- **Identical precision.** On Linux/libstdc++, `steady_clock` is `clock_gettime(CLOCK_MONOTONIC)` — the same nanosecond representation as `high_resolution_clock`, and in practice sub-100 ns real resolution via the vDSO (no syscall on the fast path). That is one to two orders of magnitude finer than the regions we time — the margin we want. Nothing is given up on precision.
- **Monotonic.** It only ever moves forward. This is the property `high_resolution_clock` fails to guarantee.
- Statically assert `steady_clock::is_steady` so a platform that violates this fails loudly at compile time rather than reporting garbage at runtime.

Store durations as **integer nanoseconds** (`duration_cast<nanoseconds>`), not `double` seconds; convert to a display unit only at output. Accumulating floating-point at ns scale over millions of regions loses precision exactly where we care about it.

`rdtsc` is deliberately **not** used: it requires calibration and is unreliable across cores and frequency/power states.

### Host vs. device time — the central mechanic

A host clock alone is not sufficient, and a `cudaDeviceSynchronize()`-per-timer is not acceptable. The timer holds **both**:

- **Host time** — `steady_clock`, as above. Measures the CPU-side cost: launch overhead, allocation, any host-side algorithm.
- **Device time** — a pair of `cudaEvent_t` recorded on the work's stream. This measures when the GPU actually did the work.

Crucially, **the elapsed time is not read at `stop()`.** `cudaEventElapsedTime` blocks until the event completes; calling it inside the step loop would sync the very pipeline we are measuring. Instead `stop()` only *records* the event and pushes the (start, stop) pair onto a pending queue. The pairs are **harvested later** — at end of iteration, or end of run — when a sync is happening anyway. Events come from a **pool** so the hot path never calls `cudaEventCreate`.

The gap between host and device time is itself the most useful signal in the whole system: `host >> device` means the GPU is starved and you are launch-bound; `device >> host` means the GPU is the bottleneck and the CPU is racing ahead.

### Nesting and double-counting

`build_graph` is called *inside* auto-weighted `partition()`, and `model_forward` contains `halo_exchange`. So timers nest, and naively summing all keys will exceed wall-clock time.

Each record therefore stores its **parent key** (a thread-local timer stack, pushed/popped by the RAII ctor/dtor). Post-processing can then present either an inclusive tree or exclusive self-time. **No arithmetic that assumes the keys are disjoint** should appear anywhere — the sum of all keys is meaningless; the sum of all *self-times* equals the iteration.

### Identity

Every record carries the full identity required to reassemble a distributed run:

| Field | Source |
|---|---|
| `rank` | `DistEnvironment::global_rank` |
| `local_rank` | `DistEnvironment::local_rank` |
| `pid` | `getpid()` |
| `device_id` | `cudaGetDevice()` |
| `hostname` | `gethostname()` |
| `run_id` | reuse the `TrajectoryWriter` run UUID so a timing file can be joined to its trajectory |

Captured **once** at registry init, not per record.

---

## Output

Two shapes were proposed in the meeting. They are not equivalent, and the difference matters more than it looks:

**Option A — accumulate, then write.** The registry aggregates in memory; the caller (or the `Simulator` at shutdown) queries it and writes once.

**Option B — stream to an `ostream`; each `stop()` emits a line.**

**Recommendation: A as the primary path, with B available as a debug mode.** The reason is the async issue above: to emit a *device* duration at `stop()` you must first block on the CUDA event — so Option B, applied to GPU work, forces a synchronization on every single timed region and measures a pipeline it has itself destroyed. It also puts file I/O in the step loop. Option B is genuinely useful for host-only timers and for crash-resilience (you keep the records written before the hang), so it should exist — but it cannot be the default for device timing.

Middle ground worth adopting: accumulate in memory, **flush every N steps**, so a crashed run still yields most of its data without paying I/O per region.

Format: **JSON Lines**, one record per line, one file per rank (`timing_rank{N}.jsonl`), merged in post-processing. Line-oriented so a truncated file from a killed job is still parseable up to the last complete line — a single JSON object is not.

```json
{"run_id":"...","rank":3,"pid":12904,"device_id":3,"host":"gpu-042","step":17,
 "key":"halo_exchange","parent":"model_forward","class":"comm",
 "host_us":812.4,"device_us":774.1}
```

### Writer atomicity

The requirement: **one or more writers must be able to append to the same place without producing a half-and-half line.** A record from rank 3 spliced into the middle of a record from rank 5 is not merely ugly — it is unparseable, and it corrupts the two records on either side of the splice.

**The rule that makes it work: build the entire line into a buffer, then emit it in exactly one `write(2)` call.**

```cpp
// Build the whole record — including the trailing '\n' — into one buffer.
std::string line = format_record(rec);   // thread-local, reused across records

// One syscall. Loop, because write() may legally return a short count.
const char* p = line.data();
size_t remaining = line.size();
while (remaining > 0) {
    ssize_t n = ::write(fd, p, remaining);   // fd opened O_WRONLY|O_APPEND|O_CREAT
    if (n < 0) { if (errno == EINTR) continue; break; }
    p += n; remaining -= n;
}
```

What this buys, and what it rules out:

- **`std::ofstream` and `operator<<` are disqualified.** Streaming a record piecewise (`os << key << "," << time`) issues *multiple* write calls, and another writer can land between any two of them. Same for `printf`-per-field. The build-then-emit-whole discipline **is** the mechanism. A "pass an `iostream`" API is fine to *offer* — but internally it must assemble the line and emit it in one call, not stream fields through.
- **`O_APPEND` is what makes the append race-free** on a local filesystem: the kernel takes the inode lock, so the seek-to-end and the write are one indivisible step. Without `O_APPEND`, two writers compute the same end offset and clobber each other.
- **Threads within one process** need nothing further — a single `write()` on a shared fd is atomic against other writes to that inode, so no mutex is required *provided nobody buffers*.

**The catch that decides the design — `O_APPEND` is not atomic on NFS.** The NFS protocol has no append operation; the client implements `O_APPEND` by fetching the file size and then writing at that offset, and that read-modify-write races between processes on different nodes. **Delta's `/u` is NFS** (verified on the box). Lustre is better but carries its own caveats under concurrent append. So a single shared append target across ranks cannot be relied on — on exactly the cluster we run on.

**Therefore: one file per rank is the default** (`timing_rank{N}.jsonl`), merged in post-processing. This sidesteps cross-process atomicity entirely — no shared cursor, no NFS append race — and costs only a merge step we want anyway. Within a rank, threads share the fd and the single-`write()` rule keeps them safe. The shared-file path stays available and correct on a local filesystem (node-local scratch, `/tmp`), but it is not the default. This mirrors `TrajectoryWriter`, which already avoids a shared append cursor by `pwrite`-ing at pre-computed disjoint offsets.

---

## Workstreams

Ordered by dependency. **W1 blocks everything; W2–W5 are parallel** once its header lands — which is the point of settling this design first.

- [ ] **W1 — Core timer (`distnnp_state`)** *(blocking; one owner)*
  - [ ] `Timer` + `TimerClass` + level switch in `include/distnnp_state/timing.hpp`
  - [ ] Registry: identity capture, thread-local parent stack, CUDA event pool, deferred harvest
  - [ ] `DISTNNP_TIME(...)` RAII macro; compiles out when off
  - [ ] JSONL writer + flush-every-N: build-line-then-single-`write()`, `O_APPEND`, per-rank file by default
  - [ ] Writer test: N threads hammering one fd produce N well-formed lines, zero splices
  - [ ] pybind11 bindings: `Timer`, `set_level`, `set_step`, `dump` — the Python side is a *binding*, not a reimplementation
  - [ ] Unit tests: off = zero records and no CUDA calls; nesting produces correct parent chains; harvest is correct
- [ ] **W2 — `distnnp_state` instrumentation** — `partition` (all three partitioners), `build_graph`, `build_halo_maps`
- [ ] **W3 — `distnnp_model` instrumentation** — `halo_exchange` (fwd + bwd), `model_forward`, `COMM` tagging, and the `WAIT`-barrier at `detail` level
- [ ] **W4 — `distnnp_integrator` instrumentation** — `integrator_step` in Euler / Verlet / Langevin
- [ ] **W5 — `distnnp` wiring** — `SimulatorConfig.timing_level`, `DISTNNP_TIMING` env override, `set_step()` per iteration, `iteration` timer, flush on shutdown
- [ ] **W6 — Post-processing** — merge per-rank JSONL; per-phase breakdown; compute-vs-comm split; rank-imbalance view. Lives in `post_processing/`.
- [ ] **W7 — Validation** — overhead measurement (`off` vs `basic` must be within noise); cross-check a known phase against `nsys` so we trust the numbers.

---

## Open questions

1. **Which side of the `COMPUTE` line does each region fall on?** The classes only mean something if we agree on the boundary. Proposed: force/energy eval, graph build, and integrator kernels are `COMPUTE`; halo exchange and collectives are `COMM`; partitioning bookkeeping, allocation, and Python orchestration are `OTHER`. The debatable one is **partitioning** — it does real math (SFC keys, KD-tree bisection) but exists only to serve the distribution, so an argument exists for either bucket. Suggest `COMPUTE` for the kernel work with the surrounding bookkeeping as `OTHER`, but this should be a group call.
2. **Is the `WAIT` barrier worth its cost?** It is the only way to stop load imbalance from masquerading as network time, but it perturbs the run. Proposed as `detail`-only. Alternative: skip it and accept that `COMM` is really "comm + imbalance", documented as such.
3. **Per-step records, or aggregate-only?** Per-step is far more useful (you see the first-step warmup, the repartition spikes, the drift) but the file grows as `steps × keys × ranks`. At `basic`, do we keep per-step or only running totals? Suggest: per-step, since ~10 keys × 1000 steps × 8 ranks is still a small file.
4. **Where does the Python-level `Timer` binding stop?** The `Simulator` loop clearly needs one. Does anything else in Python want one, or is everything else reachable from C++?
5. **`run_id` sharing with `TrajectoryWriter`** — needs a small refactor so both take the same UUID. Worth doing, and cheap, but it touches the writer.
