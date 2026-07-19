# DistNNP Timing Design Plan

This document is the plan of record for adding performance timing to DistNNP. It exists so the work can be **split across several people** — each workstream in [Workstreams](#workstreams) is independently ownable once the core `Timer` (W1) lands.

**Phasing is deliberate: timing before writing.** The whole timing infrastructure — the core timer and every component's instrumentation, producing a correct in-memory compute/comm/iteration breakdown — must be working end-to-end before persistence to disk is built at all. The output/writer design further down ([Output](#output), [Writing the file](#writing-the-file)) is recorded now so it doesn't have to be re-derived later, but it is **Phase 2** work and does not start until Phase 1 is done; see [Workstreams](#workstreams).

Status: **proposal — nothing implemented yet.** Open questions are collected at the bottom; they should be settled before W2+ starts.

---

## Goals

- Answer the question the project actually cares about: **where does a timestep go, and how much of it is communication?** (CLAUDE.md: "halo exchange volume dominates at scale.")
- Timing is a **library-level facility, not an opt-in the caller bolts on.** A user running a simulation should get timings by flipping one switch — not by wrapping their own code in timer blocks.
- Instrumentation lives **as close to the work as possible**, which in this codebase means **C++ over Python wherever the work is in C++**. Python-side timers around a C++ call measure the wrong thing (see [Why C++](#why-c-and-not-python)).
- **Near-zero cost when off**, and *honest* when on: the act of measuring must not serialize a pipeline that is asynchronous by design.
- Output is **self-describing**: every file carries version, run date, full identity (rank, pid, device id, hostname), and the call order of timed regions — so it is readable later with no external context.

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

A collective's measured duration is *not* pure network time: a rank that reaches an all-reduce early sits inside the collective waiting for the slowest rank, so a naive `COMM` bucket silently absorbs **load imbalance** and reports it as network cost. That is a misleading number to optimize against — it says "reduce halo volume" when the real fix may be "partition better."

Proposal: split it out. Record a `WAIT` class by timing a barrier immediately *before* each collective; the barrier absorbs the imbalance, and the collective that follows measures something much closer to real transport. This costs an extra barrier per collective, so it belongs behind the `detail` level rather than always on. At `basic`, `COMM` means "comm + imbalance" and must be documented as such.

### Levels

One knob, three settings, so the cost of measuring scales with how much you asked for:

| Level | Records | Extra sync/barriers |
|---|---|---|
| `off` (default) | nothing | none |
| `basic` | `iteration` + the component timers | none beyond what the code already does |
| `detail` | adds `WAIT` barriers (pre-collective sync) | yes — perturbs, use deliberately |

---

## Why C++ (and not Python)

Two independent reasons, and both matter:

1. **A Python timer around a C++/CUDA call measures the wrong quantity.** CUDA work is asynchronous: `radius_graph_pbc(...)` returns as soon as the kernels are *launched*, not when they *finish*. A `time.perf_counter()` (or a `torch.cuda.Event` pair placed in Python) around that call therefore measures launch overhead, or — if you insert a `synchronize()` to fix it — measures the correct duration *while destroying the pipelining you are trying to evaluate*. The timer has to be inside the C++ function, on the same stream as the work.

2. **Library-level means the caller does nothing.** If timing is a set of `with Timer(...)` blocks in Python, then every new call site, experiment script, and downstream user has to remember to add them — and the ones that forget are silently unmeasured. Putting the timer inside `partition()` / `build_halo_maps()` / `radius_graph_pbc` means *every* caller is instrumented, forever, including ones not written yet.

Python-side timers remain appropriate for the handful of things that genuinely *are* Python: the `Simulator` iteration boundary and the model's Python-level orchestration. The Python `Timer` should be a thin binding over the same C++ registry, not a second implementation — one registry, one output file, one set of semantics.

> The UMA model in `../refs/` has a timing implementation worth reading for reference, but it is **not** the shape we want — it is bolted on at the call sites rather than living in the library.

---

## Core design

**Guiding principle: the simplest thing that works, easy to change later.** The first version deliberately does the least it can — explicit start/stop, one time per region, accumulate by key, write JSON after the run. Anything fancier (per-call disambiguation, in-C++ nesting resolution, live streaming) is left out until a concrete need forces it in. This keeps the core small and the design cheap to revise as requirements shift.

### The timer

Explicit `start()` / `stop()`, **not** RAII. A single `Timer` can be reused for many
start/stop pairs. Each `stop()` appends one record — `{step, key, class, where, time}` —
to the in-memory registry, stamped with the current step. Records are **not** merged in
C++; summing by key (or by step) is a post-processing step. This keeps the core trivial
and leaves the raw per-occurrence data intact for timelines.

> **Why not RAII?** An earlier draft timed a *scope* (start in the constructor, stop in
> the destructor). Rejected: it forces every region into its own block and silently folds
> end-of-scope destructors into the measurement. Explicit start/stop measures exactly the
> lines between the two calls.

```cpp
namespace distnnp_state {

enum class TimerClass : uint8_t { Compute, Comm, Wait, Other };
enum class Where      : uint8_t { Host, Device };   // Host = std::chrono; Device = CUDA events

class Timer {
    enum class State { Stopped, Started };
public:
    // Explicit and reusable. start() saves the key + start time; stop() computes
    // the delta and records it under `key`. Supports many start/stop pairs.
    //
    // `where` chooses the mechanism:
    //   Host   — std::chrono around a CPU function.
    //   Device — a cudaEvent_t pair recorded on `stream`, for a GPU kernel; the
    //            elapsed time is read later, not at stop() (see below).
    void start(std::string key, TimerClass cls,
               Where where = Where::Host, cudaStream_t stream = nullptr);
    void stop();

    static void set_level(Level lvl);
    static bool enabled();
    static void set_step(uint64_t step);   // stamps subsequent records
};

} // namespace distnnp_state
```

Usage inside the library — the caller picks host vs. device per region:

```cpp
Timer t;

t.start("build_halo_maps", TimerClass::Compute);            // Host: pure CPU work
// ... work ...
t.stop();

t.start("build_graph", TimerClass::Compute, Where::Device); // Device: times the kernels
// ... launch kernels ...
t.stop();
```

When a key needs to be formatted (e.g. `"build_graph for graph {}"`), build it with
`std::format` (C++20) or `vsnprintf` over a `va_list` — **not** `stream << a << b`,
which is not thread-safe and is hard to make so.

**Repeated calls.** If a key fires several times in one step, each occurrence is just
another record with the same `key` and `step` — we do **not** number them or tell them
apart. Disambiguating "the 3rd `halo_exchange` of step 12" is out of scope for v1;
post-processing sums the occurrences. Keeping it out keeps the core small.

**Compiling timing out entirely.** The runtime `off` level makes timing cheap; a
production build can make it *free* by compiling the timing code away completely behind
a build flag (e.g. `-DDISTNNP_NO_TIMING`), so there is no chance of the measurement
perturbing a benchmark. `off` is the runtime default; the build flag is the
belt-and-suspenders for shipping a release.

> The exact call-site ergonomics — string keys vs. an enum — are still being settled by
> the team, who will provide a target usage example. This section will be updated to
> match; the shape above (explicit start/stop, host-or-device, accumulate by key) is the
> agreed foundation.

### The host clock

Use a `std::chrono` clock — `high_resolution_clock` is fine. At the millisecond-to-
sub-microsecond granularity we time, the choice of standard clock does not matter for
our benchmarking. Store durations as **integer nanoseconds** (`duration_cast<nanoseconds>`)
and convert to a display unit only at output, so accumulation over many regions doesn't
lose precision.

### Host vs. device time

The caller picks the mechanism per region via `Where`:

- **`Where::Host`** — the `std::chrono` clock above. Right for CPU work (`build_halo_maps`, host-side algorithm, launch overhead).
- **`Where::Device`** — a `cudaEvent_t` pair recorded on the work's stream. Right for a GPU kernel, because a host clock around a kernel launch measures the *launch*, not the GPU execution.

**The one rule for device timers: the elapsed time is not read at `stop()`.** `cudaEventElapsedTime` blocks until the events complete; reading it inside the step loop would sync the very pipeline we are measuring. So `stop()` only *records* the end event; the (start, stop) pairs are **read later** — at end of run (or the every-N-steps flush) — when the data is being written out anyway.

That deferred read is the only real complexity on the device path; everything else is one `time_us` per region, host or device, whichever the caller asked for.

### Nesting and double-counting

`build_graph` is called *inside* auto-weighted `partition()`, and `model_forward` contains `halo_exchange`. So timers nest, and naively summing all keys will exceed wall-clock time.

To keep the core simple, the timer does **not** try to resolve this itself — no thread-local parent stack, no self-time arithmetic in C++. It just records each region with its key and its start/stop, plus the **call order** in which regions were entered (see [Output](#output)). Reconstructing the inclusive tree vs. exclusive self-time is a **post-processing** job — the recorded order is exactly what a breakdown plot or timeline needs. The one rule the consumers must respect: **the raw sum of all keys is meaningless** because they overlap; only self-times (computed in post-processing) sum to the iteration.

### Identity and run metadata

The file should be **self-describing** — readable months later with no external context, the way `TrajectoryWriter` stamps its output. This metadata is captured **once** at registry init and written as a header block (see [Output](#output)), not repeated per record:

| Field | Source |
|---|---|
| `version` | DistNNP version / git commit, like the trajectory writer |
| `date` | wall-clock timestamp the run started |
| `rank` | `DistEnvironment::global_rank` |
| `local_rank` | `DistEnvironment::local_rank` |
| `world_size` | `DistEnvironment::world_size` |
| `pid` | `getpid()` |
| `device_id` | `cudaGetDevice()` |
| `hostname` | `gethostname()` |
| `run_id` | reuse the `TrajectoryWriter` run UUID so a timing file can be joined to its trajectory |

Plus the **call order** of the timed keys — the sequence of regions as they were first entered — so a consumer can rebuild breakdown plots and timelines without hard-coding the pipeline. Writing it into the file keeps the log self-contained.

---

## Output

**Analyze after the run; writing may be incremental.** What happens *after* the run is the **processing** — merging the per-rank files, computing the breakdown, drawing timelines. That is never live. Writing the raw records to disk, on the other hand, can happen incrementally during the run: a **flush every N steps** bounds memory and keeps most of the data if a job is killed, without paying I/O per region. Records may also just be held in memory and written once at shutdown — both are fine; the flush cadence is a tuning knob, not a design commitment.

The one constraint is on the **device** path: reading a `cudaEvent_t` elapsed time (`cudaEventElapsedTime`) blocks on the event, so device durations are read **at the flush boundary** (every N steps, or end of run) — never at each `stop()`, where it would sync the very pipeline being measured. Host durations have no such constraint.

**Format: JSON**, so it is trivially parsable and self-describing. Each file carries one **metadata header** (the identity + run metadata + call order from [Identity](#identity-and-run-metadata)) followed by the per-region records. One file per rank (`timing_rank{N}.jsonl`), merged in post-processing.

Records are **JSON Lines** (one object per line) so a file from a killed job is still readable up to its last complete line — a single giant JSON object is not. The header is the first line:

```json
{"type":"meta","version":"0.3.1+g60df325","date":"2026-07-19T14:02:11Z","run_id":"...",
 "rank":3,"local_rank":3,"world_size":8,"pid":12904,"device_id":3,"host":"gpu-042",
 "call_order":["iteration","model_forward","halo_exchange","build_graph","integrator_step"]}
{"type":"rec","step":17,"key":"halo_exchange","class":"comm","where":"device","time_us":774.1}
{"type":"rec","step":17,"key":"build_graph","class":"compute","where":"device","time_us":133.7}
```

Each record reports a single `time_us` from whichever mechanism was used (`where` says host or device), so there is no ambiguity about what was measured. The `call_order` in the header is what lets post-processing rebuild the nesting and draw timelines without knowing the pipeline in advance.

### Writing the file

**Each rank writes its own file** — so there is no cross-rank contention to defend against in the first place. Rank 3 and rank 5 never touch the same file; a record from one can't splice into a record from the other. This is deliberate, and it's the same reason `TrajectoryWriter` avoids a shared append cursor: a single shared append target *would* need atomic cross-process appends, and `O_APPEND` **is not atomic on NFS** — the client implements it as fetch-size-then-write-at-offset, which races between nodes, and **Delta's `/u` is NFS** (verified on the box). Per-rank files sidestep that entirely, at the cost of a merge step post-processing wants anyway.

That leaves only one real concern: **multiple threads within a single rank** writing to the same fd. If that happens, the rule is **build the whole line into a buffer, then emit it in exactly one `write(2)` call** — a single `write()` to a shared fd is atomic against other writes to the same inode, so two threads can't interleave a line and no mutex is needed.

```cpp
// Build the whole record — including the trailing '\n' — into one buffer.
std::string line = format_record(rec);

// One syscall. Loop, because write() may legally return a short count.
const char* p = line.data();
size_t remaining = line.size();
while (remaining > 0) {
    ssize_t n = ::write(fd, p, remaining);
    if (n < 0) { if (errno == EINTR) continue; break; }
    p += n; remaining -= n;
}
```

This is why **`std::ofstream` / `operator<<` are disqualified for the record path**: streaming a record piecewise (`os << key << "," << time`) issues *multiple* write calls, and a second thread can land between any two of them. Build the line, emit it whole.

**Therefore: one file per rank is the default** (`timing_rank{N}.jsonl`), merged in post-processing. This sidesteps cross-process atomicity entirely — no shared cursor, no NFS append race — and costs only a merge step we want anyway. Within a rank, threads share the fd and the single-`write()` rule keeps them safe. The shared-file path stays available and correct on a local filesystem (node-local scratch, `/tmp`), but it is not the default. This mirrors `TrajectoryWriter`, which already avoids a shared append cursor by `pwrite`-ing at pre-computed disjoint offsets.

---

## Workstreams

Ordered by dependency, and split into two **phases on purpose**: the whole timing infrastructure (core timer + every component instrumented + the compute/comm/iteration breakdown actually working) must be done and correct with an **in-memory-only** registry before any persistence format is built. Writing to disk is a separate, later concern — it must not gate or shape the design of the timer itself.

**Phase 1 — Timing (no writer).** W1 blocks everything; W2–W5 are parallel once its header lands.

- [ ] **W1 — Core timer (`distnnp_state`)** *(blocking; one owner)*
  - [ ] `Timer` (host `start`/`stop` + device via `Where`) + `TimerClass` + level switch in `include/distnnp_state/timing.hpp`
  - [ ] Registry: identity + run metadata capture, call-order tracking, deferred read of device (`cudaEvent_t`) timers — **in-memory only, no disk I/O**
  - [ ] Compile-time `-DDISTNNP_NO_TIMING` build flag that removes timing code entirely; runtime `off` level otherwise
  - [ ] A trivial debug accessor (e.g. print the registry to stdout) so W2–W5 can sanity-check their own timers without the real writer existing yet. This is scaffolding, not the output design.
  - [ ] pybind11 bindings: `Timer`, `set_level`, `set_step` — the Python side is a *binding*, not a reimplementation
  - [ ] Unit tests: off = zero records and no CUDA calls; host and device timers both record; deferred device read is correct
- [ ] **W2 — `distnnp_state` instrumentation** — `partition` (all three partitioners), `build_graph`, `build_halo_maps`
- [ ] **W3 — `distnnp_model` instrumentation** — `halo_exchange` (fwd + bwd), `model_forward`, `COMM` tagging, and the `WAIT`-barrier at `detail` level
- [ ] **W4 — `distnnp_integrator` instrumentation** — `integrator_step` in Euler / Verlet / Langevin
- [ ] **W5 — `distnnp` wiring** — `SimulatorConfig.timing_level`, `DISTNNP_TIMING` env override, `set_step()` per iteration, `iteration` timer
- [ ] **Phase 1 exit criterion** — with every component from W2–W5 instrumented, the in-memory registry must show a correct compute/comm/other breakdown of a real `iteration` (self-time, summing to the iteration per [Nesting](#nesting-and-double-counting)), inspected via the W1 debug accessor. Nothing in Phase 2 starts until this is true.

**Phase 2 — Writing (starts only once Phase 1's exit criterion holds).**

- [ ] **W6 — Writer** — JSONL writer + flush-every-N: per-rank file, build-line-then-single-`write()` for thread safety (see [Writing the file](#writing-the-file)); replaces W1's debug accessor as the real `dump`; flush on `Simulator` shutdown.
  - [ ] Writer test: N threads hammering one fd produce N well-formed lines, zero splices
- [ ] **W7 — Post-processing** — merge per-rank JSONL; per-phase breakdown; compute-vs-comm split; rank-imbalance view. Lives in `post_processing/`.
- [ ] **W8 — Validation** — overhead measurement (`off` vs `basic` must be within noise); cross-check a known phase against `nsys` so we trust the numbers.

---

## Open questions

1. **Which side of the `COMPUTE` line does each region fall on?** The classes only mean something if we agree on the boundary. Proposed: force/energy eval, graph build, and integrator kernels are `COMPUTE`; halo exchange and collectives are `COMM`; partitioning bookkeeping, allocation, and Python orchestration are `OTHER`. The debatable one is **partitioning** — it does real math (SFC keys, KD-tree bisection) but exists only to serve the distribution, so an argument exists for either bucket. Suggest `COMPUTE` for the kernel work with the surrounding bookkeeping as `OTHER`, but this should be a group call.
2. **Is the `WAIT` barrier worth its cost?** It is the only way to stop load imbalance from masquerading as network time, but it perturbs the run. Proposed as `detail`-only. Alternative: skip it and accept that `COMM` is really "comm + imbalance", documented as such.
3. **Where does the Python-level `Timer` binding stop?** The `Simulator` loop clearly needs one. Does anything else in Python want one, or is everything else reachable from C++?
4. **`run_id` sharing with `TrajectoryWriter`** — needs a small refactor so both take the same UUID. Worth doing, and cheap, but it touches the writer.
5. **Key type — string or enum?** (Pending: the team will provide a target usage example.) Strings dump straight into readable JSON and plots; enums are cheaper and typo-proof but need a name table for output. Leaning string for v1 simplicity.

**Resolved (this revision):** timers are explicit `start`/`stop`, not RAII; host vs. device is a per-region choice (`Where`), device timing is `cudaEvent_t`-based with a deferred read; we do **not** disambiguate repeated calls to the same key; records are **per-occurrence, stamped with the step** (summed by key in post-processing, not merged in C++); output is post-run JSON with a self-describing metadata header (version, date, identity, call order).
