#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

namespace distnnp_state {

// Timer class which times in nanoseconds.
class Timer {
    public:
        // Start the clock. Always resets, does not lap.
        void start();

        // Stops the clock and records the elapsed time.
        void stop();

        // Return measured times.
        // This is just a placeholder, will likely write or something.
        const std::vector<int64_t>& times_ns() const { return times_; }

        // Clear measured times.
        void clear() { times_.clear(); }

        Timer();

    private:
        const bool is_timing_enabled;
        std::chrono::steady_clock::time_point t0_;
        std::vector<int64_t> times_;

        // Kernel launches return once the work is queued, not once it has run, so the
        // clock has to wait for the stream to drain or it measures the launch instead.
        void wait_for_gpu();
};

} // namespace distnnp_state

// TODO:
// Write timings somewhere