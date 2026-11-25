"""Regression tests for the UMA periodic cell-offset sign.

radius_graph_pbc_cuda returns cell offsets in fairchem's convention
    vec = pos[edge_index[0]] - pos[edge_index[1]] + cell_offsets @ cell
and UMAModel.__call__ must hand them to AtomicData unchanged. With the old ``-cell_offsets`` every periodic-image
edge got a 6-100 A vector, the envelope zeroed it, and atoms near the box faces saw vacuum: the energy was no longer
invariant under rigid translations of the periodic system (hundreds of eV at 2k CH4) and NVE heated 298 -> ~500 K.

* test_radius_graph_offsets_convention: model-free, a random periodic box built exactly like UMAModel.__call__
  builds its graph; all edges <= cutoff with +offsets (and, as a sensitivity control, the -offsets vectors of the
  periodic edges are all longer than the cutoff).
* test_uma_translation_invariance[stock|fused]: Dynamis' UMAModel on CH4 2k (eager TF32, world size 1): the edge
  lengths the backbone computes (captured from _generate_graph) are <= cutoff, and a rigid translation of the whole
  periodic system (atoms wrapped back into the cell) leaves energy and forces unchanged.
Skipped without CUDA (fused additionally needs sm_100) or without the CH4 2k structure.
    pytest dynamis_model/tests/test_uma_pbc_offsets.py -s
"""
import os
import socket

import pytest
import torch

DATA = os.environ.get(
    "DYNAMIS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "../../experiments/exp_datasets/baselines"),
)
XYZ = os.path.join(DATA, "ch4_2k_splited.xyz")
CUTOFF = 6.0

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build_graph(pos, cell, pbc, cutoff):
    """Exactly the call UMAModel.__call__ makes."""
    from dynamis_state import radius_graph_pbc_cuda

    data = type("Data", (object,), {"pos": pos, "cell": cell, "natoms": [pos.shape[0]], "pbc": pbc})()
    with torch.no_grad():
        ei, off, _ = radius_graph_pbc_cuda(data=data, radius=cutoff, max_num_neighbors_threshold=0,
                                           enforce_max_neighbors_strictly=False)
    return ei, off


def _edge_vec(pos, ei, off, cell):
    """fairchem's convention (escn_md._generate_graph / graph.compute.get_pbc_distances)."""
    return pos[ei[0]] - pos[ei[1]] + off.to(cell.dtype) @ cell.reshape(3, 3)


@cuda
def test_radius_graph_offsets_convention():
    g = torch.Generator(device="cpu").manual_seed(0)
    L = 14.0
    pos = (torch.rand(600, 3, generator=g) * L).cuda()
    cell = (torch.eye(3) * L).unsqueeze(0).cuda()
    pbc = torch.ones(1, 3, dtype=torch.bool, device="cuda")
    ei, off = _build_graph(pos, cell, pbc, CUTOFF)
    periodic = off.abs().sum(1) > 0
    assert periodic.sum() > 1000          # many periodic-image edges in a 14 A box with a 6 A cutoff
    d = _edge_vec(pos, ei, off, cell).norm(dim=1)
    assert d.max().item() <= CUTOFF + 1e-4
    # sensitivity control: the old (wrong) sign puts every periodic-image edge outside the cutoff
    d_wrong = _edge_vec(pos, ei, -off, cell).norm(dim=1)
    assert d_wrong[periodic].min().item() > CUTOFF
    assert torch.equal(d_wrong[~periodic], d[~periodic])


def _port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@cuda
@pytest.mark.skipif(not os.path.exists(XYZ), reason="needs the CH4 2k structure ($DYNAMIS_DATA_DIR)")
@pytest.mark.parametrize("fused", [False], ids=["stock"])  # the fused path adds "fused"
def test_uma_translation_invariance(fused, monkeypatch):
    if fused and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("fused UMA edge blocks need sm_100")
    from ase.io import read
    from dynamis import Simulator, SimulatorConfig
    from dynamis_integrator import Verlet
    from dynamis_model.nnp import UMAModel
    from dynamis_model.nnp.uma import DistributedESCNMDBackbone
    from dynamis_model.units import EV_PER_A_TO_AMU_A_PER_FS2 as U
    from dynamis_state import TorchDistEnv
    from fairchem.core.units.mlip_unit import InferenceSettings

    # capture the edge lengths the backbone actually uses
    seen = []
    orig = DistributedESCNMDBackbone._generate_graph

    def _gen(self, data_dict):
        gd = orig(self, data_dict)
        seen.append(gd["edge_distance"].detach().max().item())
        return gd

    monkeypatch.setattr(DistributedESCNMDBackbone, "_generate_graph", _gen)

    tde = TorchDistEnv(is_distributed=True, rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{_port()}")
    try:
        settings = InferenceSettings(tf32=True, merge_mole=True, compile=False, activation_checkpointing=False,
                                     external_graph_gen=True)
        model = UMAModel("uma-s-1p2", dataset="omol", device="cuda", inference_settings=settings,
                         **({"fused": True} if fused else {}))
        L = float(read(XYZ).cell.array[0, 0])
        cfg = SimulatorConfig(timestep_fs=0.5, total_steps=0, initial_temperature_K=298.0, side_length_angstrom=L, cutoff_angstrom=CUTOFF,
                              rebalance_interval=0, traj_interval=0, log_interval=0,
                              output_dir=os.environ.get("TMPDIR", "/tmp") + "/uma_pbc_test",
                              enable_drift_watchdog=False)
        sim = Simulator(model, Verlet(dt=0.5), tde.dist_env, cfg, tde.device)
        sim.setup(XYZ)
        st = sim.state._state
        mi = st.local_molecular_info
        x0 = mi.atom_positions.clone()
        cell = st.sim_config.cell.reshape(3, 3).to(x0.dtype)
        assert torch.allclose(cell, torch.eye(3, device=cell.device, dtype=cell.dtype) * L)

        res = []
        for shift in ((0.0, 0.0, 0.0), (1.7, -3.1, 5.3), (L / 2 + 0.37, L / 3, -L / 5)):
            xs = x0 + torch.tensor(shift, device=x0.device, dtype=x0.dtype)
            mi.atom_positions.copy_(xs - torch.floor(xs / L) * L)
            e, f = model(st)
            res.append((e.double().item() / U, f.double() / U))
        mi.atom_positions.copy_(x0)

        e0, f0 = res[0]
        print(f"\n[{'fused' if fused else 'stock'}] max edge length per call {['%.3f' % d for d in seen]} A; E "
              + ", ".join(f"{e:.4f} (dE {e - e0:+.4f}, F rel {((f - f0).norm() / f0.norm()).item():.2e})"
                          for e, f in res))
        assert seen and max(seen) <= CUTOFF + 1e-4, seen
        for e, f in res[1:]:
            # measured (TF32, 2k): |dE| ~0.1 eV on |E| ~2e6 eV, forces rel ~1e-3 (run-to-run noise ~6e-4);
            # the old -cell_offsets gives |dE| of tens to hundreds of eV and forces rel O(1)
            assert abs(e - e0) / abs(e0) < 1e-6, (e, e0)
            assert ((f - f0).norm() / f0.norm()).item() < 5e-3
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
