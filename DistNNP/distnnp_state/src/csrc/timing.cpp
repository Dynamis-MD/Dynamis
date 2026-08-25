#include "distnnp_state/timing.hpp"
#include <cerrno>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace distnnp_state {

namespace {

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
    }
    return "other";
}

void write_line(int fd, const std::string& line) {
    const char* p = line.data();
    size_t remaining = line.size();
    while (remaining > 0) {
        ssize_t n = ::write(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("timing write failed: " + std::string(strerror(errno)));
        }
        p += n;
        remaining -= n;
    }
}

} // namespace

TimingRegistry& TimingRegistry::instance() {
    static TimingRegistry reg;
    return reg;
}

void TimingRegistry::init(uint32_t rank_, uint32_t local_rank_, uint32_t world_size_,
                          const std::string& run_id_, int32_t device_id_,
                          const std::string& output_dir_) {
    if (fd >= 0) { close(fd); fd = -1; }
    call_order.clear();
    call_order_seen.clear();
    device_records.clear();

    rank = rank_;
    local_rank = local_rank_;
    world_size = world_size_;
    run_id = run_id_;
    device_id = device_id_;
    output_dir = output_dir_;
    pid = static_cast<uint32_t>(::getpid());
    date = utc_now();

    char hbuf[256];
    hostname = (::gethostname(hbuf, sizeof(hbuf)) == 0) ? hbuf : "unknown";

    initialized = true;

    struct stat st{};
    if (stat(output_dir.c_str(), &st) != 0) {
        if (mkdir(output_dir.c_str(), 0755) != 0)
            throw std::runtime_error("timing: failed to create output dir: " + output_dir);
    }
    std::string path = output_dir + "/timing_rank" + std::to_string(rank) + ".jsonl";
    fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("timing: failed to open " + path + ": " + strerror(errno));

    std::ostringstream meta;
    meta << "{\"type\":\"meta\""
         << ",\"version\":\"0.0.1\""
         << ",\"date\":\"" << date << "\""
         << ",\"run_id\":\"" << bytes_to_hex(run_id) << "\""
         << ",\"rank\":" << rank
         << ",\"local_rank\":" << local_rank
         << ",\"world_size\":" << world_size
         << ",\"pid\":" << pid
         << ",\"device_id\":" << device_id
         << ",\"host\":\"" << hostname << "\""
         << "}\n";
    write_line(fd, meta.str());
}

void TimingRegistry::set_level(Level lvl) { level = lvl; }

bool TimingRegistry::enabled() const { return level != Level::Off; }

void TimingRegistry::set_step(uint64_t s) { step = s; }

void TimingRegistry::append(const std::string& key, TimerClass cls, Where where, double time_us) {
    if (!enabled()) return;

    if (call_order_seen.find(key) == call_order_seen.end()) {
        call_order.push_back(key);
        call_order_seen.insert(key);
    }

    char time_buf[32];
    snprintf(time_buf, sizeof(time_buf), "%.2f", time_us);
    std::ostringstream line;
    line << "{\"type\":\"rec\""
         << ",\"step\":" << step
         << ",\"key\":\"" << key << "\""
         << ",\"class\":\"" << timer_class_str(cls) << "\""
         << ",\"where\":\"host\""
         << ",\"time_us\":" << time_buf
         << "}\n";
    write_line(fd, line.str());
}

void TimingRegistry::append_device(const std::string& key, TimerClass cls,
                                    cudaEvent_t ev_start, cudaEvent_t ev_stop) {
    if (!enabled()) {
        // Timing was disabled between start() and stop(); discard the events.
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        return;
    }

    if (call_order_seen.find(key) == call_order_seen.end()) {
        call_order.push_back(key);
        call_order_seen.insert(key);
    }

    device_records.push_back({key, cls, step, ev_start, ev_stop});
}

void TimingRegistry::flush() {
    if (fd < 0) return;

    // Read all pending device records. Swapping out first means any Timer::stop()
    // calls that race with flush() (unlikely at end-of-run but defensive) don't
    // corrupt the iteration we're about to do.
    std::vector<PendingDeviceRecord> pending;
    pending.swap(device_records);

    for (auto& rec : pending) {
        // Block until the stop event has been reached on the GPU.
        cudaEventSynchronize(rec.ev_stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, rec.ev_start, rec.ev_stop);

        cudaEventDestroy(rec.ev_start);
        cudaEventDestroy(rec.ev_stop);

        char time_buf[32];
        snprintf(time_buf, sizeof(time_buf), "%.2f", static_cast<double>(ms) * 1000.0);
        std::ostringstream line;
        line << "{\"type\":\"rec\""
             << ",\"step\":" << rec.step
             << ",\"key\":\"" << rec.key << "\""
             << ",\"class\":\"" << timer_class_str(rec.cls) << "\""
             << ",\"where\":\"device\""
             << ",\"time_us\":" << time_buf
             << "}\n";
        write_line(fd, line.str());
    }

    // Write call_order footer so post-processing can reconstruct the pipeline.
    std::ostringstream co;
    co << "{\"type\":\"call_order\",\"order\":[";
    for (size_t i = 0; i < call_order.size(); ++i) {
        if (i > 0) co << ",";
        co << "\"" << call_order[i] << "\"";
    }
    co << "]}\n";
    write_line(fd, co.str());
    close(fd);
    fd = -1;
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

Timer::~Timer() {
    // If start() was called but stop() never was, the events were never transferred
    // to the registry — destroy them here to avoid leaking CUDA resources.
    if (ev_start) { cudaEventDestroy(ev_start); ev_start = nullptr; }
    if (ev_stop)  { cudaEventDestroy(ev_stop);  ev_stop  = nullptr; }
}

void Timer::start(const std::string& key_, TimerClass cls_, Where where_, cudaStream_t stream_) {
#ifdef DISTNNP_TIMING
    if (!TimingRegistry::instance().enabled()) return;
    key = key_;
    cls = cls_;
    where = where_;
    running = true;

    if (where == Where::Device) {
        // Create fresh events each measurement so that a reused Timer doesn't
        // stomp events that are still pending in the registry from a prior stop().
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        stream = stream_;
        cudaEventRecord(ev_start, stream);
    } else {
        t0 = std::chrono::steady_clock::now();
    }
#else
    // This is done to silence parameter warnings
    (void)key_; (void)cls_; (void)where_; (void)stream_;
#endif
}

void Timer::stop() {
#ifdef DISTNNP_TIMING
    if (!running) return;
    running = false;

    if (where == Where::Device) {
        cudaEventRecord(ev_stop, stream);
        // Transfer ownership of the event pair to the registry for deferred read.
        // Set pointers to nullptr so the destructor doesn't double-free them.
        TimingRegistry::instance().append_device(key, cls, ev_start, ev_stop);
        ev_start = nullptr;
        ev_stop  = nullptr;
    } else {
        auto t1 = std::chrono::steady_clock::now();
        double time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        TimingRegistry::instance().append(key, cls, Where::Host, time_us);
    }
#endif
}

} // namespace distnnp_state
