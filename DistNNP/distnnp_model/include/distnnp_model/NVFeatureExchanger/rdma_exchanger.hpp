#pragma once
#include <iostream>
#include <nccl.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext.h>
#include <vector>

#define FI_CHECK(call, msg)                                                    \
  do {                                                                         \
    int err = (call);                                                          \
    if (err != 0) {                                                            \
      std::cerr << msg << ": " << fi_strerror(-err) << std::endl;              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace distnnp_model {
struct RDMAPeerInfo {
  uint64_t host_hash; // Identifies the physical node
  char ep_name[128];  // Libfabric endpoint
  size_t ep_name_len;
  uint64_t mr_key;               // Libfabric memory key
  void *remote_buf;              // Remote memory address
  cudaIpcMemHandle_t ipc_handle; // CUDA NVLink handle
};

class RDMAExchanger {

public:
  void setup_libfabric_gpu();
  void teardown();

  static torch::Tensor allocate_rdma_buffer(int64_t size, int64_t device_index);

  void setup();

private:
  int64_t rank_0 = 0;
};

} // namespace distnnp_model
