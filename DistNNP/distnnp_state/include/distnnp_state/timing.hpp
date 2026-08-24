#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <cuda_runtime_api.h>

namespace distnnp_state {

enum class TimerClass : uint8_t { Compute, Comm, Wait, Other };
enum class Where      : uint8_t { Host, Device };
enum class Level      : uint8_t { Off, Basic, Detail };

// Singleton that owns all timing state for one rank.
// Initialized once from Simulator.setup(); every Timer instance writes into it.
class TimingRegistry {
public:
    static TimingRegistry& instance();

    void init(uint32_t rank, uint32_t local_rank, uint32_t world_size,
              const std::string& run_id, int32_t device_id,
              const std::string& output_dir);

    void set_level(Level level);
    bool enabled() const;
    void set_step(uint64_t step);

    // Called by Timer::stop() for host measurements. Writes the record to disk immediately.
    void append(const std::string& key, TimerClass cls, Where where, double time_us);

    // Called by Timer::stop() for device measurements. Stores the event pair for
    // deferred read at flush() — reading cudaEventElapsedTime here would block.
    void append_device(const std::string& key, TimerClass cls,
                       cudaEvent_t ev_start, cudaEvent_t ev_stop);

    // Reads all pending device records (blocking on each stop event), writes them
    // to disk, writes the call_order footer, and closes the file.
    void flush();

private:
    struct PendingDeviceRecord {
        std::string key;
        TimerClass cls;
        uint64_t step;
        cudaEvent_t ev_start;
        cudaEvent_t ev_stop;
    };

    TimingRegistry() = default;
    TimingRegistry(const TimingRegistry&) = delete;
    TimingRegistry& operator=(const TimingRegistry&) = delete;

    Level level { Level::Off };
    uint64_t step { 0 };
    bool initialized { false };

    int fd { -1 };
    std::vector<std::string> call_order;
    std::unordered_set<std::string> call_order_seen;
    std::vector<PendingDeviceRecord> device_records;

    uint32_t rank { 0 };
    uint32_t local_rank { 0 };
    uint32_t world_size { 1 };
    int32_t device_id { -1 };
    std::string run_id;
    std::string output_dir;
    std::string hostname;
    std::string date;
    uint32_t pid { 0 };
};

// Lightweight measurement instrument. Each instance holds one in-flight
// start/stop pair; stop() appends to the shared TimingRegistry.
// A single Timer can be reused across many start/stop pairs.
class Timer {
public:
    Timer() = default;
    ~Timer();

    // `where` selects the measurement mechanism:
    //   Host   — std::chrono; right for CPU-bound or ATen work that syncs internally.
    //   Device — cudaEvent_t pair on `stream`; right for GPU kernels.
    //            Elapsed time is read at flush(), not at stop(), to avoid blocking.
    void start(const std::string& key, TimerClass cls,
               Where where = Where::Host, cudaStream_t stream = nullptr);
    void stop();

    static void set_level(Level lvl) { TimingRegistry::instance().set_level(lvl); }
    static bool enabled()            { return TimingRegistry::instance().enabled(); }
    static void set_step(uint64_t s) { TimingRegistry::instance().set_step(s); }

private:
    std::string key;
    TimerClass cls { TimerClass::Other };
    Where where { Where::Host };
    bool running { false };

    // Host path
    std::chrono::time_point<std::chrono::steady_clock> t0;

    // Device path — fresh events are created on each start() and ownership is
    // transferred to the registry on stop(), so these are nullptr between pairs.
    cudaEvent_t ev_start { nullptr };
    cudaEvent_t ev_stop { nullptr };
    cudaStream_t stream { nullptr };
};

} // namespace distnnp_state
