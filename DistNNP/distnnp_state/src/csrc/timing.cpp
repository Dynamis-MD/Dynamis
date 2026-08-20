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

const char* where_str(Where where) {
    switch (where) {
        case Where::Host:   return "host";
        case Where::Device: return "device";
    }
    return "host";
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
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
    call_order.clear();
    call_order_seen.clear();

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
    fd_ = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0)
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
    write_line(fd_, meta.str());
}

void TimingRegistry::set_level(Level lvl) {
    level = lvl;
}

bool TimingRegistry::enabled() const {
    return level != Level::Off;
}

void TimingRegistry::set_step(uint64_t s) {
    step = s;
}

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
         << ",\"where\":\"" << where_str(where) << "\""
         << ",\"time_us\":" << time_buf
         << "}\n";
    write_line(fd_, line.str());
}

void TimingRegistry::flush() {
    if (fd_ < 0) return;

    // Write call_order as a footer record so post-processing can reconstruct the pipeline order.
    std::ostringstream co;
    co << "{\"type\":\"call_order\",\"order\":[";
    for (size_t i = 0; i < call_order.size(); ++i) {
        if (i > 0) co << ",";
        co << "\"" << call_order[i] << "\"";
    }
    co << "]}\n";
    write_line(fd_, co.str());
    close(fd_);
    fd_ = -1;
}

void Timer::start(const std::string& key_, TimerClass cls_, Where where_, void* /*stream*/) {
    if (!TimingRegistry::instance().enabled()) return;
    key = key_;
    cls = cls_;
    where = where_;
    running = true;
    t0 = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    if (!running) return;
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    TimingRegistry::instance().append(key, cls, where, time_us);
    running = false;
}

} // namespace distnnp_state
