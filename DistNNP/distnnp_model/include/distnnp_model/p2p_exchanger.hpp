#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <vector>
#include "distnnp_state/distnnp_state.hpp"


namespace distnnp_mode
{
    struct P2P_PushTarget{
        float *remote_ptr;      // IPC-opened pointer to peer's buffer
        int64_t remote_offset;  // atom offset in peer's buffer where our data starts
        int64_t send_start;     // start index in send_local_indices for this peer
        int64_t send_count;     // number of atoms to push to this peer
    };


    class IPCExchanger
    {
    public:
        IPCExchanger() = default;
        ~IPCExchanger();

        // Set up IPC handles, allocate 
        void setup_helper(const DistEnvironment& dist_environment,
                          const PartitionMessagePassingInfo& partitino_info);

        void put(const torch::Tensot send_buffer);
        void record_push_complete(cudaStream_t stream);
        void wait_for_pushes(cudaStream_t stream);
        void teardown();

        std::vector<uint8_t> get_recv_forward_buffer_handle();
        std::vector<uint8_t> get_rev_grad_buffer_handle();
    
    private:
        int64_t rank_ = 0;
        int64_t total_send_ = 0;
        int64_t total_recv_ = 0;
        bool is_setup_ = false;
        uint32_t call_counter_ = 0;

        
        void setup_internal_buffer(const size_t recv_forward_size, 
                                   const size_t recv_grad_size);
    };
} // namespace distnnp_model