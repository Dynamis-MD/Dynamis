#include "distnnp_model/exchanger.hpp"
#include <torch/extension.h>
#include "distnnp_model/push_buffer.hpp"


PYBIND11_MODULE(distnnp_model_impl, m)
{
  m.def("push_buffer_create", &distnnp_model::push_buffer_create, "Gather rows into buffer by index");
  m.def("supports_fabric_handles", &distnnp_model::supports_fabric_handles,
        "Check if fabric handles (multi-node NVLink / NVL72) are supported",
        py::arg("device_id") = -1);

  py::class_<distnnp_model::IPCExchanger>(m, "IPCExchanger")
    .def(py::init<>())
    .def_static("supports_fabric_handles", &distnnp_model::IPCExchanger::supports_fabric_handles,
         "Check if fabric handles are supported on the device",
         py::arg("device_id") = -1)
    .def_static("allocate_ipc_buffer", &distnnp_model::IPCExchanger::allocate_ipc_buffer,
         py::arg("rows"), py::arg("cols"), py::arg("device_index"))
    .def("export_ipc_handle", &distnnp_model::IPCExchanger::export_ipc_handle,
         py::arg("buffer"))
    .def("setup", &distnnp_model::IPCExchanger::setup,
         py::arg("x_buffer"),
         py::arg("grad_recv_buffer"),
         py::arg("all_x_handles"),
         py::arg("all_grad_handles"),
         py::arg("send_offsets"),
         py::arg("send_local_indices"),
         py::arg("recv_offsets"),
         py::arg("all_recv_offsets"),
         py::arg("all_num_local"),
         py::arg("num_local"),
         py::arg("feature_dim"),
         py::arg("rank"),
         py::arg("world_size"))
    .def("push_forward",
         [](distnnp_model::IPCExchanger &self, torch::Tensor x) { self.push_forward(x); },
         py::arg("x_buffer"))
    .def("push_backward",
         [](distnnp_model::IPCExchanger &self, torch::Tensor g) { self.push_backward(g); },
         py::arg("grad_at_ghost"))
    .def("create_and_export_sync_handle", &distnnp_model::IPCExchanger::create_and_export_sync_handle)
    .def("setup_sync", &distnnp_model::IPCExchanger::setup_sync,
         py::arg("all_sync_handles"))
    .def("record_push_complete",
         [](distnnp_model::IPCExchanger &self) { self.record_push_complete(); })
    .def("wait_for_pushes",
         [](distnnp_model::IPCExchanger &self) { self.wait_for_pushes(); })
    .def("teardown", &distnnp_model::IPCExchanger::teardown)
    .def("is_setup", &distnnp_model::IPCExchanger::is_setup)
    .def("use_fabric", &distnnp_model::IPCExchanger::use_fabric);

  py::class_<distnnp_model::FeatureExchanger>(m, "FeatureExchanger")
    .def(py::init<>())
    // --- shehtab's public interface ---
    .def("feature_push_forward", &distnnp_model::FeatureExchanger::feature_push_forward,
         "Push features forward to neighbors",
         py::arg("local_features"), py::arg("partition_info"))
    .def("feature_push_backward", &distnnp_model::FeatureExchanger::feature_push_backward,
         "Push feature gradients backward to neighbors",
         py::arg("grad_output"), py::arg("grad_input_shape"), py::arg("partition_info"))
    // --- IPC configuration ---
    .def_static("supports_fabric_handles", &distnnp_model::FeatureExchanger::supports_fabric_handles,
         "Check if fabric handles are supported on the device",
         py::arg("device_id") = -1)
    .def_static("allocate_ipc_buffer", &distnnp_model::FeatureExchanger::allocate_ipc_buffer,
         "Allocate a tensor via Fabric VMM or raw cudaMalloc (required for IPC/Fabric)",
         py::arg("rows"), py::arg("cols"), py::arg("device_index"))
    .def("export_ipc_handle", &distnnp_model::FeatureExchanger::export_ipc_handle,
         "Export CUDA IPC or Fabric handle for a buffer as bytes",
         py::arg("buffer"))
    .def("setup_ipc", &distnnp_model::FeatureExchanger::setup_ipc,
         "Open remote IPC/Fabric handles and build push targets",
         py::arg("x_buffer"),
         py::arg("grad_recv_buffer"),
         py::arg("all_x_handles"),
         py::arg("all_grad_handles"),
         py::arg("send_offsets"),
         py::arg("send_local_indices"),
         py::arg("recv_offsets"),
         py::arg("all_recv_offsets"),
         py::arg("all_num_local"),
         py::arg("num_local"),
         py::arg("feature_dim"),
         py::arg("rank"),
         py::arg("world_size"))
    .def("create_and_export_sync_handle", &distnnp_model::FeatureExchanger::create_and_export_sync_handle,
         "Allocate IPC/Fabric sync counter and return its handle bytes")
    .def("setup_sync", &distnnp_model::FeatureExchanger::setup_sync,
         "Open remote IPC/Fabric sync counters for stream-level push synchronization",
         py::arg("all_sync_handles"))
    .def("record_push_complete", &distnnp_model::FeatureExchanger::record_push_complete,
         "Bump sync counter on default stream after push completes")
    .def("wait_for_pushes", &distnnp_model::FeatureExchanger::wait_for_pushes,
         "Wait on default stream until all peers' sync counters reach current step")
    .def("teardown", &distnnp_model::FeatureExchanger::teardown)
    .def("is_ipc_setup", &distnnp_model::FeatureExchanger::is_ipc_setup)
    .def("use_fabric", &distnnp_model::FeatureExchanger::use_fabric)
    .def("set_ce_threshold_bytes", &distnnp_model::FeatureExchanger::set_ce_threshold_bytes,
         py::arg("threshold"))
    .def("ce_threshold_bytes", &distnnp_model::FeatureExchanger::ce_threshold_bytes);
}
