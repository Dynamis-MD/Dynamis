#include "distnnp_state/timing.hpp"

#ifdef DISTNNP_TIMING

#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace distnnp_state {

namespace {

// Starting size, not a bound — records grow the buffer for the whole run. 64 KiB
// holds ~500 records, enough that short runs never reallocate.
constexpr size_t kBufReserve = 64 * 1024;

// Cap on recycled CUDA events held for reuse; beyond it, events go back to the
// driver instead of being hoarded for the length of the run.
constexpr size_t kMaxPooledEvents = 256;

// How many pending device records pile up before append_device() sweeps the
// completed ones, so a driver query happens roughly every N appends rather
// than on every one.
constexpr size_t kDeviceReapThreshold = 64;

// ---------------------------------------------------------------------------
// Span nesting
// ---------------------------------------------------------------------------
//
// Thread-local so two genuinely parallel regions on different threads never
// become each other's parent by accident; see ParentSpan for the opt-out.
// Ids start at 1 so 0 can mean "root" without a separate flag.
std::atomic<uint64_t> g_next_span_id { 1 };

// Not a strict stack: overlapping regions are legal (`a.start; b.start; a.stop;
// b.stop`), so removal searches for the id instead of assuming it's on top.
thread_local std::vector<uint64_t> g_open_spans;

Span open_span() {
    Span s;
    s.id = g_next_span_id.fetch_add(1, std::memory_order_relaxed);
    s.parent = g_open_spans.empty() ? 0 : g_open_spans.back();
    g_open_spans.push_back(s.id);
    return s;
}

void close_span(uint64_t id) {
    for (size_t i = g_open_spans.size(); i-- > 0;) {
        if (g_open_spans[i] == id) {
            g_open_spans.erase(g_open_spans.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
}

// close() runs from ~TimingRegistry() (implicitly noexcept), so failures are
// reported, not thrown; stderr is guaranteed to exist at static destruction.
void report_failure(const char* what, const char* detail) {
    std::fprintf(stderr, "distnnp_state timing: %s: %s\n", what,
                 detail ? detail : "unknown error");
    std::fflush(stderr);
}

std::string bytes_to_hex(const std::string& bytes) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char c : bytes) {
        out += hex[c >> 4];
        out += hex[c & 0xf];
    }
    return out;
}

// Escape a string for inclusion in a JSON string literal. Keys reach us from Python
// as arbitrary std::string; an unescaped '"' or newline would corrupt the whole
// file's parse, not just one record.
void append_json_escaped(std::string& out, const std::string& s) {
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
}

void append_i64(std::string& out, int64_t v) {
    char buf[24];
    int n = std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v));
    if (n > 0) out.append(buf, static_cast<size_t>(n));
}

void append_u64(std::string& out, uint64_t v) {
    char buf[24];
    int n = std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
    if (n > 0) out.append(buf, static_cast<size_t>(n));
}

std::string utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

const char* timer_class_str(TimerClass cls) {
    switch (cls) {
        case TimerClass::Compute: return "compute";
        case TimerClass::Comm:    return "comm";
        case TimerClass::Wait:    return "wait";
        case TimerClass::Other:   return "other";
        case TimerClass::Total:   return "total";
    }
    return "other";
}

const char* where_str(Where where) {
    // Where::Both never reaches a record: it is expanded into one host and one
    // device record at Timer::stop().
    return where == Where::Device ? "device" : "host";
}

// A record. `t_start_ns` and `time_ns` are nullable: a failed CUDA call must not
// fabricate a plausible-looking 0, so the record is emitted with a null time and an
// `error` field naming the failure instead.
void append_record(std::string& out, uint64_t step, const std::string& key,
                   TimerClass cls, Where where, Span span,
                   const int64_t* t_start_ns, const int64_t* time_ns,
                   const char* err) {
    out += "{\"type\":\"rec\",\"step\":";
    append_u64(out, step);
    out += ",\"key\":\"";
    append_json_escaped(out, key);
    out += "\",\"class\":\"";
    out += timer_class_str(cls);
    out += "\",\"where\":\"";
    out += where_str(where);
    // `id` identifies the region, `parent_id` the region it ran inside — null for a
    // root. Where::Both emits two records under one id, distinguished by `where`.
    out += "\",\"id\":";
    append_u64(out, span.id);
    out += ",\"parent_id\":";
    if (span.parent) append_u64(out, span.parent); else out += "null";
    out += ",\"t_start_ns\":";
    if (t_start_ns) append_i64(out, *t_start_ns); else out += "null";
    out += ",\"time_ns\":";
    if (time_ns) append_i64(out, *time_ns); else out += "null";
    if (err) {
        out += ",\"error\":\"";
        append_json_escaped(out, err);
        out += "\"";
    }
    out += "}\n";
}

// mkdir -p, tolerating a directory that already exists. All ranks on a node race to
// create a shared output_dir; the losers must not die on EEXIST.
void mkdir_p(const std::string& path) {
    if (path.empty()) return;
    std::string acc;
    size_t i = 0;
    if (path[0] == '/') { acc = "/"; i = 1; }
    while (i <= path.size()) {
        size_t j = path.find('/', i);
        if (j == std::string::npos) j = path.size();
        if (j > i) {
            if (!acc.empty() && acc.back() != '/') acc += '/';
            acc.append(path, i, j - i);
            if (::mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST)
                throw std::runtime_error("timing: failed to create output dir '" + acc +
                                         "': " + std::strerror(errno));
        }
        i = j + 1;
    }
}

// Is there a usable CUDA device in this process? Called before anything that would
// throw or abort on a CPU-only build/box.
bool cuda_present() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

cudaStream_t current_stream(int device_index) {
    return device_index < 0
        ? c10::cuda::getCurrentCUDAStream().stream()
        : c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_index)).stream();
}

} // namespace

// ---------------------------------------------------------------------------
// TimingRegistry
// ---------------------------------------------------------------------------

TimingRegistry& TimingRegistry::instance() {
    static TimingRegistry reg;
    return reg;
}

TimingRegistry::~TimingRegistry() {
    // Safety net: lands buffered records even if the run never reached close().
    close();
}

int64_t TimingRegistry::now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void TimingRegistry::init(uint32_t rank_, uint32_t local_rank_, uint32_t world_size_,
                          const std::string& run_id_, int32_t device_id_,
                          const std::string& output_dir_) {
    // Held for the whole of init(): close_io_locked() and reanchor_timebase() both
    // rebuild `ref_event`, and io_mu keeps a concurrent flush()/close() out of that.
    std::lock_guard<std::mutex> io(io_mu);

    // Drain and close any previous registry instead of leaking its pending events.
    close_io_locked();

    std::string meta;
    {
        std::lock_guard<std::mutex> lk(mu);

        rank = rank_;
        local_rank = local_rank_;
        world_size = world_size_;
        run_id = run_id_;
        device_id = device_id_;
        output_dir = output_dir_;
        pid = static_cast<uint32_t>(::getpid());
        date = utc_now();

        char hbuf[256];
        // POSIX does not guarantee null-termination when the name does not fit.
        hbuf[sizeof(hbuf) - 1] = '\0';
        hostname = (::gethostname(hbuf, sizeof(hbuf) - 1) == 0) ? hbuf : "unknown";

        out_buf.clear();
        out_buf.reserve(kBufReserve);
        call_order.clear();
        call_order_seen.clear();
        device_records.clear();

        mkdir_p(output_dir);
        std::string path = output_dir + "/timing_rank" + std::to_string(rank) + ".jsonl";
        // O_CLOEXEC: the fd must not leak into dataloader workers or any subprocess.
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0)
            throw std::runtime_error("timing: failed to open " + path + ": " +
                                     std::strerror(errno));

        // 0.1.0 added `id`/`parent_id` to every record; 0.0.1 readers will not find
        // the nesting they need to compute self time.
        meta += "{\"type\":\"meta\",\"version\":\"0.1.0\"";
        meta += ",\"date\":\"";
        append_json_escaped(meta, date);
        meta += "\",\"run_id\":\"";
        append_json_escaped(meta, bytes_to_hex(run_id));
        meta += "\",\"rank\":";
        append_u64(meta, rank);
        meta += ",\"local_rank\":";
        append_u64(meta, local_rank);
        meta += ",\"world_size\":";
        append_u64(meta, world_size);
        meta += ",\"pid\":";
        append_u64(meta, pid);
        meta += ",\"device_id\":";
        append_i64(meta, device_id);
        meta += ",\"host\":\"";
        append_json_escaped(meta, hostname);
        // steady_clock's epoch is arbitrary but fixed for the process; t0_ns lets
        // records be made relative to it.
        meta += "\",\"clock\":\"steady_clock\",\"time_unit\":\"ns\",\"t0_ns\":";
        append_i64(meta, now_ns());
        meta += "}\n";
        write_buffer_locked(meta);

        pool_device = -1;
        if (cuda_present()) {
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) pool_device = dev;
        }
        initialized = true;
    }

    // Outside the lock: reanchor_timebase() takes `mu` itself and syncs the stream.
    reanchor_timebase();
}

void TimingRegistry::set_level(Level lvl) {
    level.store(lvl, std::memory_order_relaxed);
}

bool TimingRegistry::enabled() const {
    return level.load(std::memory_order_relaxed) != Level::Off;
}

void TimingRegistry::set_step(uint64_t s) {
    step.store(s, std::memory_order_relaxed);
}

uint64_t TimingRegistry::current_step() const {
    return step.load(std::memory_order_relaxed);
}

void TimingRegistry::note_key_locked(const std::string& key) {
    if (call_order_seen.insert(key).second) call_order.push_back(key);
}

void TimingRegistry::append(const std::string& key, TimerClass cls, Where where,
                            int64_t t_start_ns, int64_t time_ns, uint64_t rec_step,
                            Span span) {
    std::lock_guard<std::mutex> lk(mu);
    if (!enabled() || !initialized || fd < 0) return;
    note_key_locked(key);
    append_record(out_buf, rec_step, key, cls, where, span, &t_start_ns, &time_ns,
                  nullptr);
}

void TimingRegistry::append_error(const std::string& key, TimerClass cls, Where where,
                                  uint64_t rec_step, Span span, const char* err) {
    std::lock_guard<std::mutex> lk(mu);
    if (!enabled() || !initialized || fd < 0) return;
    note_key_locked(key);
    append_record(out_buf, rec_step, key, cls, where, span, nullptr, nullptr, err);
}

void TimingRegistry::append_device(const std::string& key, TimerClass cls,
                                   cudaEvent_t ev_start, cudaEvent_t ev_stop,
                                   uint64_t rec_step, Span span) {
    std::lock_guard<std::mutex> lk(mu);
    if (!enabled() || !initialized || fd < 0) {
        // Timing was disabled (or the registry closed) between start() and stop();
        // recycle the events rather than leaking them.
        release_event_locked(ev_start);
        release_event_locked(ev_stop);
        return;
    }
    note_key_locked(key);
    device_records.push_back({key, cls, rec_step, span, ev_start, ev_stop});
    if (device_records.size() >= kDeviceReapThreshold) reap_device_records_locked();
}

// The run holds everything until close(), so what could grow unbounded is the CUDA
// event pairs, not the buffered text (~150 bytes/record vs. two driver handles that
// exhaust the driver at scale). Reaping releases events as soon as their stop event
// completes, moving the record to `out_buf` — the same place a host record sits —
// so "one write, at exit" is unchanged while live events track work in flight.
//
// Emitted in completion order, not start order: every record carries its own step
// and t_start_ns, and post-processing sorts on those.
void TimingRegistry::reap_device_records_locked() {
    size_t keep = 0;
    for (size_t i = 0; i < device_records.size(); ++i) {
        PendingDeviceRecord& rec = device_records[i];
        // Any error other than "still in flight" is left for flush(), which blocks
        // and reports the failure rather than guessing at it from a query.
        if (cudaEventQuery(rec.ev_stop) != cudaSuccess) {
            device_records[keep++] = rec;
            continue;
        }
        format_device_record(out_buf, rec, ref_event, ref_host_ns, has_ref, nullptr);
        release_event_locked(rec.ev_start);
        release_event_locked(rec.ev_stop);
    }
    device_records.resize(keep);
}

// `err`, when non-null, is the failure the caller hit while waiting on ev_stop; the
// record is still emitted, carrying the error instead of a fabricated duration.
void TimingRegistry::format_device_record(std::string& out, const PendingDeviceRecord& rec,
                                          cudaEvent_t ref, int64_t ref_ns, bool ref_ok,
                                          const char* err) {
    int64_t t_start_ns = 0;
    int64_t time_ns = 0;
    bool have_start = false;
    bool have_time = false;

    if (!err) {
        float ms = 0.0f;
        cudaError_t e = cudaEventElapsedTime(&ms, rec.ev_start, rec.ev_stop);
        if (e != cudaSuccess) {
            err = cudaGetErrorName(e);
        } else {
            time_ns = static_cast<int64_t>(std::llround(static_cast<double>(ms) * 1e6));
            have_time = true;
        }
    }

    if (!err && ref_ok) {
        float offset_ms = 0.0f;
        if (cudaEventElapsedTime(&offset_ms, ref, rec.ev_start) == cudaSuccess) {
            t_start_ns = ref_ns +
                static_cast<int64_t>(std::llround(static_cast<double>(offset_ms) * 1e6));
            have_start = true;
        }
        // A failure here costs only the timeline position, not the duration —
        // leave t_start_ns null and keep the measurement.
    }

    append_record(out, rec.step, rec.key, rec.cls, Where::Device, rec.span,
                  have_start ? &t_start_ns : nullptr,
                  have_time ? &time_ns : nullptr,
                  err);
}

// ---------------------------------------------------------------------------
// Event pool
// ---------------------------------------------------------------------------
//
// Assumes one CUDA device per process (one rank per GPU); a stray cross-device
// timer falls back to create/destroy rather than getting a wrong-device event.

cudaEvent_t TimingRegistry::acquire_event() {
    int dev = -1;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (dev >= 0 && dev == pool_device && !event_pool.empty()) {
            cudaEvent_t ev = event_pool.back();
            event_pool.pop_back();
            return ev;
        }
    }
    cudaEvent_t ev = nullptr;
    // cudaEventDefault keeps timing enabled; cudaEventDisableTiming would silently
    // make cudaEventElapsedTime fail.
    if (cudaEventCreateWithFlags(&ev, cudaEventDefault) != cudaSuccess) return nullptr;
    return ev;
}

void TimingRegistry::release_event_locked(cudaEvent_t ev) {
    if (!ev) return;
    if (pool_device >= 0 && event_pool.size() < kMaxPooledEvents) {
        event_pool.push_back(ev);
    } else {
        cudaEventDestroy(ev);
    }
}

void TimingRegistry::release_event(cudaEvent_t ev) {
    std::lock_guard<std::mutex> lk(mu);
    release_event_locked(ev);
}

void TimingRegistry::drain_event_pool_locked() {
    for (cudaEvent_t ev : event_pool) cudaEventDestroy(ev);
    event_pool.clear();
}

// ---------------------------------------------------------------------------
// Timebase
// ---------------------------------------------------------------------------

// Anchors the device timeline to the host clock: records a reference event on a
// quiesced stream and notes the host time at that instant, so a device event's
// host-timeline position is ref_host_ns + elapsed(ref_event, ev).
//
// Costs a cudaStreamSynchronize, so it only happens in init()/flush() — normally
// once, at the start. cudaEventElapsedTime's float precision then means
// t_start_ns drifts (~70us/hr, ~0.4ms after an hour) the longer a run goes without
// a flush; durations (time_ns) are unaffected since start/stop events are always
// close together. flush() at a quiescent point re-anchors and resets the drift.
void TimingRegistry::retire_ref_event_locked() {
    if (ref_event) {
        cudaEventDestroy(ref_event);
        ref_event = nullptr;
    }
    has_ref = false;
}

// Requires io_mu; must NOT be called with mu held.
//
// Built on a local handle, published in one assignment under `mu`: the autograd
// thread reads ref_event holding only `mu` and must never see a half-built event.
// io_mu (held by every caller) keeps this from racing a concurrent flush() reader.
void TimingRegistry::reanchor_timebase() {
    if (!cuda_present()) {
        std::lock_guard<std::mutex> lk(mu);
        retire_ref_event_locked();
        return;
    }

    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDefault) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(mu);
        retire_ref_event_locked();
        return;
    }

    cudaStream_t s = current_stream(-1);
    bool ok = cudaStreamSynchronize(s) == cudaSuccess;
    int64_t before = now_ns();
    ok = ok && cudaEventRecord(ev, s) == cudaSuccess;
    ok = ok && cudaEventSynchronize(ev) == cudaSuccess;
    int64_t after = now_ns();

    std::lock_guard<std::mutex> lk(mu);
    retire_ref_event_locked();
    if (!ok) {
        cudaEventDestroy(ev);
        return;
    }
    ref_event = ev;
    has_ref = true;
    // Event fires somewhere in [before, after]; midpoint estimates the instant.
    ref_host_ns = before + (after - before) / 2;
}

// ---------------------------------------------------------------------------
// Flush / close
// ---------------------------------------------------------------------------

void TimingRegistry::write_buffer_locked(const std::string& buf) {
    if (fd < 0 || buf.empty()) return;
    const char* p = buf.data();
    size_t remaining = buf.size();
    while (remaining > 0) {
        ssize_t n = ::write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("timing write failed: " + std::string(std::strerror(errno)));
        }
        if (n == 0) throw std::runtime_error("timing write failed: write(2) returned 0");
        p += n;
        remaining -= static_cast<size_t>(n);
    }
}

void TimingRegistry::append_call_order_locked(std::string& out) const {
    out += "{\"type\":\"call_order\",\"order\":[";
    for (size_t i = 0; i < call_order.size(); ++i) {
        if (i > 0) out += ",";
        out += "\"";
        append_json_escaped(out, call_order[i]);
        out += "\"";
    }
    out += "]}\n";
}

void TimingRegistry::flush() {
    std::lock_guard<std::mutex> io(io_mu);
    flush_io_locked(/*reanchor=*/true);
}

// Requires io_mu; must NOT be called with mu held.
void TimingRegistry::flush_io_locked(bool reanchor) {
    std::vector<PendingDeviceRecord> pending;
    std::string hostbuf;
    cudaEvent_t ref = nullptr;
    int64_t ref_ns = 0;
    bool ref_ok = false;

    {
        std::lock_guard<std::mutex> lk(mu);
        if (!initialized || fd < 0) return;
        pending.swap(device_records);
        hostbuf.swap(out_buf);
        out_buf.reserve(kBufReserve);
        // Safe outside the lock: only init()/flush()/close() retire ref_event, and
        // all three hold io_mu, which this call already owns.
        ref = ref_event;
        ref_ns = ref_host_ns;
        ref_ok = has_ref;
    }

    // Read the device events outside the lock: cudaEventSynchronize can block, and
    // the autograd worker thread must stay free to append while it does.
    std::string devbuf;
    std::vector<cudaEvent_t> recycled;
    recycled.reserve(pending.size() * 2);

    for (auto& rec : pending) {
        // Unlike the reap path this waits: flush() is the last chance to read these,
        // so a record still in flight must be blocked on rather than skipped.
        cudaError_t e = cudaEventSynchronize(rec.ev_stop);
        format_device_record(devbuf, rec, ref, ref_ns, ref_ok,
                             e == cudaSuccess ? nullptr : cudaGetErrorName(e));

        recycled.push_back(rec.ev_start);
        recycled.push_back(rec.ev_stop);
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        if (fd < 0) {
            for (cudaEvent_t ev : recycled) cudaEventDestroy(ev);
            return;
        }
        std::string buf;
        buf.reserve(hostbuf.size() + devbuf.size() + 256);
        buf += hostbuf;
        buf += devbuf;
        // Written with the records, not as a footer, so a mid-run flush still
        // leaves a complete call_order line — the last one in the file wins.
        append_call_order_locked(buf);
        write_buffer_locked(buf);
        for (cudaEvent_t ev : recycled) release_event_locked(ev);
    }

    if (reanchor) reanchor_timebase();
}

// Reached from ~TimingRegistry() (implicitly noexcept) — nothing here may escape.
// A failed write is ordinary (a full disk) and must not turn exit into an abort.
void TimingRegistry::close() {
    try {
        std::lock_guard<std::mutex> io(io_mu);
        close_io_locked();
    } catch (const std::exception& e) {
        report_failure("close", e.what());
    } catch (...) {
        report_failure("close", nullptr);
    }
}

// Requires io_mu; must NOT be called with mu held.
void TimingRegistry::close_io_locked() {
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!initialized && fd < 0) return;
    }

    // A write failure must not stop the fd from being closed or the pending events
    // from being destroyed below, so the drain is contained rather than propagated.
    try {
        flush_io_locked(/*reanchor=*/false);
    } catch (const std::exception& e) {
        report_failure("flush during close", e.what());
    } catch (...) {
        report_failure("flush during close", nullptr);
    }

    std::lock_guard<std::mutex> lk(mu);
    close_locked();
}

void TimingRegistry::close_locked() noexcept {
    if (fd >= 0) {
        std::string buf;
        buf.swap(out_buf);
        if (!buf.empty()) {
            try {
                write_buffer_locked(buf);
            } catch (const std::exception& e) {
                report_failure("final write", e.what());
            } catch (...) {
                report_failure("final write", nullptr);
            }
        }
        ::close(fd);
        fd = -1;
    }
    // Anything appended between flush() and here was never read; destroy the events
    // rather than leaking them.
    for (auto& rec : device_records) {
        cudaEventDestroy(rec.ev_start);
        cudaEventDestroy(rec.ev_stop);
    }
    device_records.clear();
    drain_event_pool_locked();
    retire_ref_event_locked();
    initialized = false;
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

Timer::~Timer() {
    // start() without a matching stop(): the events were never transferred to the
    // registry, so hand them back here rather than leaking CUDA resources.
    if (ev_start) TimingRegistry::instance().release_event(ev_start);
    if (ev_stop)  TimingRegistry::instance().release_event(ev_stop);
    ev_start = nullptr;
    ev_stop = nullptr;

    // Must also come off the open-span stack, or later regions on this thread
    // are recorded as children of a region that's gone.
    if (running) {
        close_span(span.id);
        running = false;
    }
}

uint64_t Timer::current_span_id() {
    return g_open_spans.empty() ? 0 : g_open_spans.back();
}

void Timer::begin(std::string_view key_, TimerClass cls_, Where where_,
                  cudaStream_t stream_, ParentSpan parent) {
    TimingRegistry& reg = TimingRegistry::instance();
    if (!reg.enabled()) return;

    if (running) {
        // A missing stop() otherwise yields a plausible-looking wrong number and
        // silently leaks the previous event pair — a call-site bug, so hard error.
        throw std::logic_error(
            "distnnp_state::Timer: start(\"" + std::string(key_) +
            "\") called while \"" + key + "\" is still running — a stop() is missing");
    }

    key = key_;
    cls = cls_;
    where = where_;
    step = reg.current_step();
    stream = stream_;
    // Opened before the work runs, so anything launched on this thread nests
    // underneath it; an explicit parent only redirects the upward link.
    span = open_span();
    if (parent.id) span.parent = parent.id;
    running = true;

    if (where == Where::Device || where == Where::Both) {
        ev_start = reg.acquire_event();
        ev_stop = reg.acquire_event();
        if (ev_start) cudaEventRecord(ev_start, stream);
    }
    // Taken last so the host span starts as close to the work as possible.
    if (where == Where::Host || where == Where::Both) t0_ns = TimingRegistry::now_ns();
}

void Timer::start(std::string_view key_, TimerClass cls_, Where where_,
                  cudaStream_t stream_) {
    begin(key_, cls_, where_, stream_, ParentSpan{});
}

void Timer::start(std::string_view key_, TimerClass cls_, Where where_) {
    start(key_, cls_, where_, ParentSpan{});
}

void Timer::start(std::string_view key_, TimerClass cls_, Where where_,
                  ParentSpan parent) {
    // Level check before the stream lookup: that ordering is why the lookup lives
    // here instead of the caller's argument list, keeping Level::Off a single load.
    if (!TimingRegistry::instance().enabled()) return;

    cudaStream_t s = nullptr;
    if (where_ == Where::Device || where_ == Where::Both) {
        if (cuda_present()) {
            s = current_stream(-1);
        } else {
            // No GPU: degrade to a host measurement instead of failing outright.
            where_ = Where::Host;
        }
    }
    begin(key_, cls_, where_, s, parent);
}

void Timer::stop() {
    if (!running) return;
    running = false;
    // Closed before appending: the region is over, so a later region on this thread
    // is its sibling, not its child.
    close_span(span.id);

    TimingRegistry& reg = TimingRegistry::instance();

    if (where == Where::Device || where == Where::Both) {
        if (ev_start && ev_stop) {
            cudaError_t e = cudaEventRecord(ev_stop, stream);
            if (e == cudaSuccess) {
                // Ownership of the pair transfers to the registry for a deferred read.
                reg.append_device(key, cls, ev_start, ev_stop, step, span);
            } else {
                reg.append_error(key, cls, Where::Device, step, span,
                                 cudaGetErrorName(e));
                reg.release_event(ev_start);
                reg.release_event(ev_stop);
            }
        } else {
            // Event creation failed — say so rather than dropping the region silently.
            reg.append_error(key, cls, Where::Device, step, span,
                             "cudaEventCreate failed");
            reg.release_event(ev_start);
            reg.release_event(ev_stop);
        }
        ev_start = nullptr;
        ev_stop = nullptr;
    }

    if (where == Where::Host || where == Where::Both) {
        int64_t t1_ns = TimingRegistry::now_ns();
        reg.append(key, cls, Where::Host, t0_ns, t1_ns - t0_ns, step, span);
    }
}

} // namespace distnnp_state

#endif // DISTNNP_TIMING
