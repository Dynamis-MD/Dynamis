#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace distnnp_state {

enum class TimerClass : uint8_t { Compute, Comm, Wait, Other };
enum class Where : uint8_t { Host, Device };
enum class Level : uint8_t { Off, Basic, Detail };

// Singleton that owns all timing state for one rank.
// Initialized once from Simulator.setup(); every Timer instance writes into it.
class TimingRegistry {
public:
    static TimingRegistry& instance();

    // Called once from Simulator.setup() after the distributed environment is ready.
    // run_id is the raw UUID bytes shared with TrajectoryWriter.
    void init(uint32_t rank, uint32_t local_rank, uint32_t world_size,
              const std::string& run_id, int32_t device_id,
              const std::string& output_dir);

    void set_level(Level level);
    bool enabled() const;
    void set_step(uint64_t step);

    // Called by Timer::stop(). Writes the record directly to disk; no-op when Off.
    void append(const std::string& key, TimerClass cls, Where where, double time_us);

    // Writes the call_order footer record and closes the output file.
    void flush();

private:
    TimingRegistry() = default;
    TimingRegistry(const TimingRegistry&) = delete;
    TimingRegistry& operator=(const TimingRegistry&) = delete;

    Level level { Level::Off };
    uint64_t step { 0 };
    bool initialized { false };

    // fd_ < 0 means no file is open yet (opened lazily on first append).
    int fd_ { -1 };
    std::vector<std::string> call_order;
    std::unordered_set<std::string> call_order_seen;

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
    void start(const std::string& key, TimerClass cls,
               Where where = Where::Host, void* stream = nullptr);
    void stop();

    static void set_level(Level level) { TimingRegistry::instance().set_level(level); }
    static bool enabled() { return TimingRegistry::instance().enabled(); }
    static void set_step(uint64_t step) { TimingRegistry::instance().set_step(step); }

private:
    std::string key;
    TimerClass cls { TimerClass::Other };
    Where where { Where::Host };
    bool running { false };
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
};

} // namespace distnnp_state
