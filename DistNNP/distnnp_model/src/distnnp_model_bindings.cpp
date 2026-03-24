#include "distnnp_model/exchanger.hpp"
#include <torch/extension.h>
#include "distnnp_model/push_buffer.hpp"


PYBIND11_MODULE(distnnp_model_impl, m)
{
  m.def("push_buffer_create", &distnnp_model::push_buffer_create, "Gather rows into buffer by index");

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
    .def_static("allocate_ipc_buffer", &distnnp_model::FeatureExchanger::allocate_ipc_buffer,
         "Allocate a tensor via raw cudaMalloc (required for IPC)",
         py::arg("rows"), py::arg("cols"), py::arg("device_index"))
    .def("export_ipc_handle", &distnnp_model::FeatureExchanger::export_ipc_handle,
         "Export CUDA IPC handle for a buffer as bytes",
         py::arg("buffer"))
    .def("setup_ipc", &distnnp_model::FeatureExchanger::setup_ipc,
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
    .def("create_and_export_event", &distnnp_model::FeatureExchanger::create_and_export_event,
         "Create interprocess event and return IPC handle (64 bytes)")
    .def("setup_events", &distnnp_model::FeatureExchanger::setup_events,
         "Open remote IPC events for stream-level sync",
         py::arg("all_event_handles"))
    .def("record_push_complete", &distnnp_model::FeatureExchanger::record_push_complete,
         "Record completion event on default stream")
    .def("wait_for_pushes", &distnnp_model::FeatureExchanger::wait_for_pushes,
         "Wait for all remote pushes to complete on default stream")
    .def("teardown", &distnnp_model::FeatureExchanger::teardown)
    .def("is_ipc_setup", &distnnp_model::FeatureExchanger::is_ipc_setup)
    .def("set_ce_threshold_bytes", &distnnp_model::FeatureExchanger::set_ce_threshold_bytes,
         py::arg("threshold"))
    .def("ce_threshold_bytes", &distnnp_model::FeatureExchanger::ce_threshold_bytes);
}
