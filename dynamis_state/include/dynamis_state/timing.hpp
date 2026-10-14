#pragma once

// The compile-time on/off decision is inherited from the build of dynamis_state
// via this generated header — never defined per-consumer. See timing_config.hpp.in.
#include "dynamis_state/timing_config.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <cuda_runtime_api.h>

#ifdef DYNAMIS_TIMING
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_set>
#include <vector>
#endif

// Unlike `assert`, arguments here are still evaluated when timing is off — never
// resolve a CUDA stream at the call site; pass plain `start(key, cls, where)` and
// let the library resolve it internally, after the level check.

namespace dynamis_state {

// `Total` is the denominator the others are a fraction of, not a partition
// member — post-processing must not fold it into a bucket.
enum class TimerClass : uint8_t { Compute, Comm, Wait, Other, Total };

// Host   — std::chrono around CPU work.
// Device — a cudaEvent_t pair on the work's stream, for GPU kernels.
// Both   — one host + one device record for the same region; right for
//          Python-orchestrated work that also launches async GPU work.
enum class Where : uint8_t { Host, Device, Both };

// Off is the default. Detail is reserved for WAIT barriers, not yet implemented,
// so it currently behaves exactly like Basic; TimerClass::Wait is likewise unused.
enum class Level : uint8_t { Off, Basic, Detail };

// Nesting is *recorded*, not inferred from timestamps: durations include
// children (so time_ns alone double-counts) and t_start_ns is too coarse to
// reconstruct containment. `parent` is the innermost region open when this one
// started (0 = root). Where::Both shares one `id` across its host/device
// records, so a child looks up the nearest ancestor with a record on its own
// clock.
struct Span {
    uint64_t id{ 0 };
    uint64_t parent{ 0 };
};

// Named parent for a region that logically nests under a region open on a
// different thread — e.g. the halo-exchange backward runs on PyTorch's autograd
// worker while `model_forward` is still open on the caller's thread. The owning
// thread captures Timer::current_span_id() and passes it here.
//
// A distinct type rather than a bare integer, to avoid ambiguity with the
// cudaStream_t overload on a literal `0`.
struct ParentSpan {
    uint64_t id{ 0 };
};

#ifdef DYNAMIS_TIMING

// Singleton owning all timing state for one rank; initialized once from
// Simulator.__init__, and every Timer writes into it.
//
// Thread safety: stop() runs on both the main thread and PyTorch's autograd
// worker thread, so mutable state is guarded by `mu`. `io_mu` separately
// serializes init/flush/close, which do blocking CUDA work outside `mu` —
// order is io_mu then mu; see io_mu's declaration.
//
// Records buffer in memory and the file is written once, from close(), so a
// SIGKILL leaves no file; close() also runs from the destructor and from
// Simulator.run()'s `finally`, so exceptions and interrupts still land the data.
class TimingRegistry
{
public:
    static TimingRegistry &instance();

    // Closes any live registry first (draining pending CUDA events) rather than
    // leaking them.
    void init(uint32_t rank, uint32_t local_rank, uint32_t world_size, const std::string &run_id,
              int32_t device_id, const std::string &output_dir);

    void set_level(Level level);
    bool enabled() const;
    void set_step(uint64_t step);
    uint64_t current_step() const;

    // `step` is captured at start(), not read from the registry — otherwise the
    // autograd thread would stamp backward records with the main thread's later step.
    void append(const std::string &key, TimerClass cls, Where where, int64_t t_start_ns,
                int64_t time_ns, uint64_t step, Span span);

    // Emitted with a null time and an `error` field naming the failure, so a
    // broken measurement is visible instead of a fabricated 0.
    void append_error(const std::string &key, TimerClass cls, Where where, uint64_t step, Span span,
                      const char *err);

    // Takes ownership of the event pair and defers the read — reading here would
    // block on the pipeline being measured. Reclaimed once the stop event
    // completes, without blocking or writing to disk.
    void append_device(const std::string &key, TimerClass cls, cudaEvent_t ev_start,
                       cudaEvent_t ev_stop, uint64_t step, Span span);

    // Drains buffered and pending records to disk and re-emits call_order; the
    // file stays open. Blocks on every pending device event, so this isn't on the
    // step loop's path — exposed for tests and quiescent-point checkpoints.
    void flush();

    // flush() + close the file. Also runs from the destructor, so an abnormal
    // exit still lands data.
    //
    // Never throws: this runs from ~TimingRegistry(), an implicitly noexcept
    // context. A full disk is an ordinary failure, not a crash — it's reported on
    // stderr and the file abandoned. flush(), called explicitly, still throws.
    void close();

    // Two events per device measurement; pooled so a long run doesn't churn or
    // exhaust driver event handles.
    cudaEvent_t acquire_event();
    void release_event(cudaEvent_t ev);

    // Nanoseconds on the process-wide monotonic clock every record is stamped on.
    static int64_t now_ns();

    // Public because pybind11's class_<TimingRegistry> instantiates a deallocator
    // for the bound type. Construction stays private — this is a singleton.
    ~TimingRegistry();

private:
    struct PendingDeviceRecord {
        std::string key;
        TimerClass cls;
        uint64_t step;
        Span span;
        cudaEvent_t ev_start;
        cudaEvent_t ev_stop;
    };

    TimingRegistry() = default;
    TimingRegistry(const TimingRegistry &) = delete;
    TimingRegistry &operator=(const TimingRegistry &) = delete;

    // Each requires `io_mu`, not `mu` — they read CUDA events / re-anchor the
    // timebase outside `mu` so the autograd worker can keep appending while they
    // block. `reanchor=false` on the close path: re-anchoring right before
    // destroying the anchor is pointless, and it's the last libtorch call on a
    // path that also runs at static destruction, where libtorch's own statics may
    // already be gone.
    void flush_io_locked(bool reanchor);
    void close_io_locked();
    void reanchor_timebase();

    // All of the below require `mu` to be held.
    void note_key_locked(const std::string &key);
    void write_buffer_locked(const std::string &buf);
    void append_call_order_locked(std::string &out) const;
    void release_event_locked(cudaEvent_t ev);
    void drain_event_pool_locked();
    // Destroys the current anchor. Under `mu` because reap_device_records_locked()
    // reads ref_event from the autograd thread holding only that lock.
    void retire_ref_event_locked();
    // noexcept, and checked: this is the tail of the destructor path.
    void close_locked() noexcept;

    // Moves device records whose stop event has completed into `out_buf`,
    // recycling their events; a record still in flight just stays pending. This
    // bounds the live event count without holding data until exit — see the .cpp
    // for why events, not buffered text, are the actual constraint.
    void reap_device_records_locked();

    // Touches no state beyond the anchor passed in, so it serves both the locked
    // reap path and flush()'s unlocked read path.
    static void format_device_record(std::string &out, const PendingDeviceRecord &rec,
                                     cudaEvent_t ref, int64_t ref_ns, bool ref_ok, const char *err);

    // Serializes init()/flush()/close(), which read device events and re-anchor
    // the timebase outside `mu` (cudaEventSynchronize blocks, and the worker must
    // stay free to append) — without this lock two drains could race on
    // `ref_event` or use-after-free it. Lock order: io_mu before mu, never the
    // reverse.
    mutable std::mutex io_mu;

    mutable std::mutex mu;

    // Atomic rather than mutex-guarded: enabled() and current_step() are on the
    // hottest path there is, and Level::Off must cost a single relaxed load.
    std::atomic<Level> level{ Level::Off };
    std::atomic<uint64_t> step{ 0 };
    bool initialized{ false };

    int fd{ -1 };
    std::string out_buf;
    std::vector<std::string> call_order;
    std::unordered_set<std::string> call_order_seen;
    std::vector<PendingDeviceRecord> device_records;

    std::vector<cudaEvent_t> event_pool;
    int pool_device{ -1 };

    // Common host/device timebase: `ref_event` anchors a quiesced stream to
    // `ref_host_ns`, so a device event's host-timeline position is
    // ref_host_ns + elapsed(ref_event, ev). Anchoring costs a stream sync, so it
    // only happens in init()/flush() — see the .cpp for the resulting drift in
    // t_start_ns (durations are unaffected).
    //
    // Guarded by both locks: `mu` because the reap path reads it from the
    // autograd thread, `io_mu` because flush() reads it after dropping `mu`.
    cudaEvent_t ref_event{ nullptr };
    int64_t ref_host_ns{ 0 };
    bool has_ref{ false };

    uint32_t rank{ 0 };
    uint32_t local_rank{ 0 };
    uint32_t world_size{ 1 };
    int32_t device_id{ -1 };
    std::string run_id;
    std::string output_dir;
    std::string hostname;
    std::string date;
    uint32_t pid{ 0 };
};

// Lightweight measurement instrument. Each instance holds one in-flight start/stop
// pair; stop() appends to the shared TimingRegistry. A single Timer can be reused
// across many start/stop pairs, but not concurrently from two threads.
class Timer
{
public:
    Timer() = default;
    ~Timer();
    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;

    // A view, not `const std::string&`: call sites pass literals that often exceed
    // the small-string threshold, so a string param would heap-allocate a
    // temporary just to have begin() copy it in. begin() assigns the view into
    // the member, reusing its buffer across a reused Timer's start/stop pairs.

    // Host timing, or device timing on the current stream/device.
    //
    // The stream is resolved *inside*, after the level check, so Level::Off costs
    // one relaxed load and an off build has nothing left to elide — passing
    // getCurrentCUDAStream(...) as an argument would force that call even for a
    // no-op. One rank per GPU means the current device is always the right one.
    void start(std::string_view key, TimerClass cls, Where where = Where::Host);

    // Device timing on an explicit `stream`, for a call site that already holds
    // it (the integrator kernels do) — free, since the variable is live either
    // way. No default: the legacy default stream implicitly synchronizes every
    // blocking stream, serializing the pipeline being measured.
    void start(std::string_view key, TimerClass cls, Where where, cudaStream_t stream);

    // As the 3-arg form, but nesting under an explicitly named parent instead of
    // this thread's innermost open region — for a region continued from another
    // thread; see ParentSpan.
    void start(std::string_view key, TimerClass cls, Where where, ParentSpan parent);

    void stop();

    // The innermost region currently open on the calling thread, or 0 if none.
    // Capture this on the owning thread to hand to the ParentSpan overload later.
    static uint64_t current_span_id();

    static void set_level(Level lvl) { TimingRegistry::instance().set_level(lvl); }
    static bool enabled() { return TimingRegistry::instance().enabled(); }
    static void set_step(uint64_t s) { TimingRegistry::instance().set_step(s); }

private:
    void begin(std::string_view key, TimerClass cls, Where where, cudaStream_t stream,
               ParentSpan parent);

    std::string key;
    TimerClass cls{ TimerClass::Other };
    Where where{ Where::Host };
    bool running{ false };
    uint64_t step{ 0 };
    Span span;

    // Host path
    int64_t t0_ns{ 0 };

    // Taken from the registry's pool on start(), returned on stop(), so these are
    // nullptr between pairs.
    cudaEvent_t ev_start{ nullptr };
    cudaEvent_t ev_stop{ nullptr };
    cudaStream_t stream{ nullptr };
};

#else // !DYNAMIS_TIMING

// No-op twins of the classes above: header-only, nothing to link against.
// `Timer t; t.start(...); t.stop();` compiles to nothing, so call sites stay
// free of #ifdefs; the pybind11 API stays exposed but does nothing.
class TimingRegistry
{
public:
    static TimingRegistry &instance()
    {
        static TimingRegistry r;
        return r;
    }

    void init(uint32_t, uint32_t, uint32_t, const std::string &, int32_t, const std::string &) { }
    void set_level(Level) { }
    bool enabled() const { return false; }
    void set_step(uint64_t) { }
    uint64_t current_step() const { return 0; }
    void flush() { }
    void close() { }

private:
    TimingRegistry() = default;
    TimingRegistry(const TimingRegistry &) = delete;
    TimingRegistry &operator=(const TimingRegistry &) = delete;
};

class Timer
{
public:
    Timer() = default;
    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;

    void start(std::string_view, TimerClass, Where = Where::Host) { }
    void start(std::string_view, TimerClass, Where, cudaStream_t) { }
    void start(std::string_view, TimerClass, Where, ParentSpan) { }
    void stop() { }

    static constexpr uint64_t current_span_id() { return 0; }

    static void set_level(Level) { }
    static constexpr bool enabled() { return false; }
    static void set_step(uint64_t) { }
};

// Checks the "costs nothing" claim above rather than trusting it: empty means
// constructing touches no storage, trivially destructible means no code on scope exit.
static_assert(std::is_empty_v<Timer>, "Timer must have no members when DYNAMIS_TIMING is off");
static_assert(std::is_trivially_destructible_v<Timer>,
              "Timer must be trivially destructible when DYNAMIS_TIMING is off");

#endif // DYNAMIS_TIMING

} // namespace dynamis_state
