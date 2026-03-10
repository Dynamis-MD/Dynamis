#include "distnnp_state/distnnp_state.hpp"
#include "distnnp_state/partitioner.hpp"
#include "distnnp_state/radius_pbc.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(distnnp_state_impl, m)
{
    py::class_<distnnp_state::MolecularInfo>(m, "MolecularInfo")
        .def(py::init<>())
        .def_readwrite("num_atoms", &distnnp_state::MolecularInfo::num_atoms)
        .def_readwrite("atom_indices", &distnnp_state::MolecularInfo::atom_indices)
        .def_readwrite("atom_positions", &distnnp_state::MolecularInfo::atom_positions)
        .def_readwrite("atom_velocities", &distnnp_state::MolecularInfo::atom_velocities);

    py::class_<distnnp_state::DistEnvironment>(m, "DistEnvironment")
        .def(py::init<>())
        .def_readwrite("global_rank", &distnnp_state::DistEnvironment::global_rank)
        .def_readwrite("world_size", &distnnp_state::DistEnvironment::world_size)
        .def_readwrite("gpus_per_node", &distnnp_state::DistEnvironment::gpus_per_node)
        .def_readwrite("num_nodes", &distnnp_state::DistEnvironment::num_nodes)
        .def_readwrite("local_rank", &distnnp_state::DistEnvironment::local_rank);

    py::class_<distnnp_state::SimulationConfig>(m, "SimulationConfig")
        .def(py::init<>())
        .def_readwrite("cutoff", &distnnp_state::SimulationConfig::cutoff)
        .def_readwrite("ghost_cutoff_buffer", &distnnp_state::SimulationConfig::ghost_cutoff_buffer)
        .def_readwrite("cell", &distnnp_state::SimulationConfig::cell)
        .def_readwrite("pbc", &distnnp_state::SimulationConfig::pbc);

    py::class_<distnnp_state::PartitionMessagePassingInfo>(m, "PartitionMessagePassingInfo")
        .def(py::init<>())
        .def_readwrite("partition_id", &distnnp_state::PartitionMessagePassingInfo::partition_id)
        .def_readwrite("global_send_recv_matrix", &distnnp_state::PartitionMessagePassingInfo::global_send_recv_matrix)
        .def_readwrite("send_offsets", &distnnp_state::PartitionMessagePassingInfo::send_offsets)
        .def_readwrite("send_local_indices", &distnnp_state::PartitionMessagePassingInfo::send_local_indices)
        .def_readwrite("recv_offsets", &distnnp_state::PartitionMessagePassingInfo::recv_offsets)
        .def_readwrite("recv_ghost_indices", &distnnp_state::PartitionMessagePassingInfo::recv_ghost_indices)
        .def_readwrite("interior_mask", &distnnp_state::PartitionMessagePassingInfo::interior_mask);

    py::class_<distnnp_state::DistNNPState>(m, "DistNNPState")
        .def(py::init<>())
        .def_readwrite("dist_env", &distnnp_state::DistNNPState::dist_env)
        .def_readwrite("global_molecular_info", &distnnp_state::DistNNPState::global_molecular_info)
        .def_readwrite("local_partition_info", &distnnp_state::DistNNPState::local_partition_info);

    py::class_<distnnp_state::KDTreePartitioner>(m, "KDTreePartitioner")
        .def(py::init<>())
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &>(
                 &distnnp_state::KDTreePartitioner::partition),
             "Partition: unweighted (default)")
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &,
                               const torch::Tensor weights &>(
                 &distnnp_state::KDTreePartitioner::partition),
             "Partition: user-weighted")
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &,
                               const distnnp_state::SimulationConfig &>(
                 &distnnp_state::KDTreePartitioner::partition),
             "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
        .def("build_halo_maps", &distnnp_state::KDTreePartitioner::build_halo_maps);

    py::class_<distnnp_state::SimpleSpatialPartitioner>(m, "SimpleSpatialPartitioner")
        .def(py::init<>())
        .def("partition", &distnnp_state::SimpleSpatialPartitioner::partition)
        .def("build_halo_maps", &distnnp_state::SimpleSpatialPartitioner::build_halo_maps);

    py::enum_<distnnp_state::SFCCurveType>(m, "SFCCurveType")
        .value("Morton", distnnp_state::SFCCurveType::Morton)
        .value("Hilbert", distnnp_state::SFCCurveType::Hilbert);

    py::class_<distnnp_state::SFCPartitioner>(m, "SFCPartitioner")
        .def(py::init<distnnp_state::SFCCurveType>(),
             py::arg("curve_type") = distnnp_state::SFCCurveType::Morton)
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &>(
                 &distnnp_state::SFCPartitioner::partition),
             "Partition: unweighted")
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &,
                               const torch::Tensor weights &>(
                 &distnnp_state::SFCPartitioner::partition),
             "Partition: user-weighted")
        .def("partition",
             py::overload_cast<const distnnp_state::MolecularInfo &,
                               const distnnp_state::DistEnvironment &,
                               const distnnp_state::SimulationConfig &>(
                 &distnnp_state::SFCPartitioner::partition),
             "Partition: auto-compute edge weights via radius_graph_pbc (CUDA)")
        .def("build_halo_maps", &distnnp_state::SFCPartitioner::build_halo_maps);

    m.def("radius_graph_pbc", &radius_graph_pbc,
          "CUDA cell-list radius graph with selective PBC tiling",
          py::arg("pos"),
          py::arg("cell"),
          py::arg("pbc"),
          py::arg("radius"),
          py::arg("max_num_neighbors"),
          py::arg("enforce_max_strictly"));
}
