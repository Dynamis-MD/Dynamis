#include "distnnp_model/exchanger.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_model_impl, m)
{
  py::class_<distnnp_model::IPCExchanger>(m, "IPCExchanger")
      .def(py::init<>())
      .def_static("allocate_ipc_buffer", &distnnp_model::IPCExchanger::allocate_ipc_buffer,
           "Allocate a tensor via raw cudaMalloc (required for IPC)",
           py::arg("rows"), py::arg("cols"), py::arg("device_index"))
      .def("export_ipc_handle", &distnnp_model::IPCExchanger::export_ipc_handle,
           "Export CUDA IPC handle for a buffer as bytes",
           py::arg("buffer"))
      .def("setup", &distnnp_model::IPCExchanger::setup,
           "Open remote IPC handles and build push targets",
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
      .def("push_forward", [](distnnp_model::IPCExchanger &self, torch::Tensor x_buffer,
                              int64_t ce_threshold_bytes) {
           self.push_forward(x_buffer, 0, ce_threshold_bytes);
      }, "Forward push: gather + write to peers' ghost regions.\n"
         "ce_threshold_bytes: per-peer data >= this uses Copy Engine (0 = always SM kernel)",
           py::arg("x_buffer"), py::arg("ce_threshold_bytes") = 0)
      .def("push_backward", [](distnnp_model::IPCExchanger &self, torch::Tensor grad) {
           self.push_backward(grad, 0);
      }, "Backward push: gather ghost grads + write to peers' grad buffers",
           py::arg("grad"))
      .def("create_and_export_event", &distnnp_model::IPCExchanger::create_and_export_event,
           "Create interprocess event and return IPC handle (64 bytes)")
      .def("setup_events", &distnnp_model::IPCExchanger::setup_events,
           "Open remote IPC events for stream-level sync",
           py::arg("all_event_handles"))
      .def("record_push_complete", [](distnnp_model::IPCExchanger &self) {
           self.record_push_complete(0);
      }, "Record completion event on default stream")
      .def("wait_for_pushes", [](distnnp_model::IPCExchanger &self) {
           self.wait_for_pushes(0);
      }, "Wait for all remote pushes to complete on default stream")
      .def("teardown", &distnnp_model::IPCExchanger::teardown)
      .def("is_setup", &distnnp_model::IPCExchanger::is_setup)
      .def("feature_dim", &distnnp_model::IPCExchanger::feature_dim)
      .def("num_local", &distnnp_model::IPCExchanger::num_local)
      .def("total_send", &distnnp_model::IPCExchanger::total_send)
      .def("total_recv", &distnnp_model::IPCExchanger::total_recv)
      .def("send_local_indices", &distnnp_model::IPCExchanger::send_local_indices);
}
