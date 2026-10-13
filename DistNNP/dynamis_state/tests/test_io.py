"""Tests for io integration"""

import pathlib
import struct
import uuid

import torch
import torch.multiprocessing as mp
import pytest

from dynamis_state import MolecularInfo, TrajectoryReader, TrajectoryWriter, translate_dynamis_to_xyz
from dynamis_state.masses import TORCH_ATOMIC_MASSES


# ─── TrajectoryReader.read() ───────────────────────────────────────────────


def test_xyz_extended_io():
    """Extended xyz test"""
    env, cell, pbc = TrajectoryReader.read("tests/extended_xyz.xyz")
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape(6, 3)
    velocities = torch.concat((values, values), dim=0)
    velocities = velocities.reshape(6, 3)

    assert env.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env.atom_types)
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(velocities, env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env.atom_vertex_numbers)

    assert torch.equal(torch.diag(torch.tensor([15.0, 30.0, 30.0])), cell)
    assert torch.equal(torch.ones(3, dtype=torch.uint8), pbc)


def _two_atom_xyz(path, column):
    path.write_text(
        "2\n"
        f'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3:{column}:R:3 pbc="T T T"\n'
        "C 0.0 1.0 2.0 0.5 1.5 2.5\n"
        "H 3.0 4.0 5.0 3.5 4.5 5.5\n"
    )
    return path


@pytest.mark.parametrize("column", ["vel", "velo", "velocities"])
def test_xyz_velocity_column_aliases(tmp_path, column):
    """Every velocity spelling is read straight into atom_velocities."""
    xyz = _two_atom_xyz(tmp_path / f"{column}.xyz", column)

    env, _, _ = TrajectoryReader.read(str(xyz))

    assert torch.equal(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), env.atom_positions)
    assert torch.equal(torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]), env.atom_velocities)


def test_xyz_momenta_column_divided_by_mass(tmp_path):
    """A momenta column holds p = m*v, so it has to be divided back through by
    each atom's mass to land in atom_velocities."""
    xyz = _two_atom_xyz(tmp_path / "momenta.xyz", "momenta")

    env, _, _ = TrajectoryReader.read(str(xyz))

    carbon, hydrogen = TORCH_ATOMIC_MASSES[6], TORCH_ATOMIC_MASSES[1]
    expected = torch.tensor([
        [0.5 / carbon, 1.5 / carbon, 2.5 / carbon],
        [3.5 / hydrogen, 4.5 / hydrogen, 5.5 / hydrogen],
    ])

    assert torch.equal(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), env.atom_positions)
    assert torch.allclose(expected, env.atom_velocities)


def test_xyz_io():
    """Regular xyz test"""
    env, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape([6, 3])
    assert env.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env.atom_types)
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(torch.zeros([6, 3]), env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env.atom_vertex_numbers)

    assert torch.equal(torch.zeros(3, 3), cell)
    assert torch.equal(torch.zeros(3, dtype=torch.uint8), pbc)


# multi_frame.xyz: 3 frames, 3 atoms each (Ca, Br, H), each frame's positions
# offset by 100 * frame_index; frame 1 has Lattice=/pbc=/velocities, frame 2
# has Lattice=/pbc= but no velocities, frame 0 has none of the above.
_MULTI_FRAME_TYPES = torch.tensor([20, 35, 1], dtype=torch.long)


def _multi_frame_positions(step):
    return torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]) + step * 100.0


def test_xyz_multi_frame_explicit_indices():
    for step in (0, 1, 2):
        env, cell, pbc = TrajectoryReader.read("tests/multi_frame.xyz", step)
        assert torch.equal(_MULTI_FRAME_TYPES, env.atom_types)
        assert torch.equal(_multi_frame_positions(step), env.atom_positions)

    env0, cell0, pbc0 = TrajectoryReader.read("tests/multi_frame.xyz", 0)
    assert torch.equal(torch.zeros([3, 3]), env0.atom_velocities)
    assert torch.equal(torch.zeros(3, 3), cell0)
    assert torch.equal(torch.zeros(3, dtype=torch.uint8), pbc0)

    env1, cell1, pbc1 = TrajectoryReader.read("tests/multi_frame.xyz", 1)
    assert torch.equal(torch.ones([3, 3]), env1.atom_velocities)
    assert torch.equal(torch.eye(3) * 10.0, cell1)
    assert torch.equal(torch.ones(3, dtype=torch.uint8), pbc1)

    env2, cell2, pbc2 = TrajectoryReader.read("tests/multi_frame.xyz", 2)
    assert torch.equal(torch.zeros([3, 3]), env2.atom_velocities)
    assert torch.equal(torch.eye(3) * 20.0, cell2)
    assert torch.equal(torch.tensor([0, 0, 1], dtype=torch.uint8), pbc2)


def test_xyz_multi_frame_default_is_last():
    env, cell, pbc = TrajectoryReader.read("tests/multi_frame.xyz")
    assert torch.equal(_multi_frame_positions(2), env.atom_positions)
    assert torch.equal(torch.eye(3) * 20.0, cell)


def test_xyz_multi_frame_negative_index():
    env, cell, pbc = TrajectoryReader.read("tests/multi_frame.xyz", -2)
    assert torch.equal(_multi_frame_positions(1), env.atom_positions)
    assert torch.equal(torch.eye(3) * 10.0, cell)


def test_xyz_frame_index_out_of_range_raises():
    with pytest.raises(RuntimeError):
        TrajectoryReader.read("tests/multi_frame.xyz", 3)
    with pytest.raises(RuntimeError):
        TrajectoryReader.read("tests/multi_frame.xyz", -4)


def test_xyz_natoms_mismatch_raises(tmp_path):
    bad_file = tmp_path / "natoms_mismatch.xyz"
    bad_file.write_text(
        "3\n\n"
        "Ca 0.0 1.0 2.0\n"
        "Br 3.0 4.0 5.0\n"
        "H 6.0 7.0 8.0\n"
        "2\n\n"
        "Ca 0.0 1.0 2.0\n"
        "Br 3.0 4.0 5.0\n"
    )
    with pytest.raises(RuntimeError):
        TrajectoryReader.read(str(bad_file))


def test_xyz_malformed_atom_line_raises(tmp_path):
    bad_file = tmp_path / "malformed.xyz"
    bad_file.write_text("2\n\nCa 0.0 1.0 2.0\nnotanatom\n")
    with pytest.raises(RuntimeError):
        TrajectoryReader.read(str(bad_file))


def test_dynamis_bad_magic_raises(tmp_path):
    bad_file = tmp_path / "bad_magic.dynamis"
    bad_file.write_bytes(b"NOTAMAGIC" + b"\x00" * 100)
    with pytest.raises(RuntimeError):
        TrajectoryReader.read(str(bad_file))


def test_dynamis_unsupported_version_raises(tmp_path):
    bad_file = tmp_path / "bad_version.dynamis"
    fixed = struct.pack(
        "<8sI16s64sQfIII",
        b"DYNAMIS\x00", 2,
        b"1234".ljust(16, b"\x00"),
        b"author".ljust(64, b"\x00"),
        1, 1.0, 1, 12, 0,
    )
    bad_file.write_bytes(fixed + b"\x00" * 32)
    with pytest.raises(RuntimeError):
        TrajectoryReader.read(str(bad_file))


def test_read_missing_position_field_raises(tmp_path):
    """A hand-crafted file whose only field is "velocity" (no "position") must
    fail to load, since positions are mandatory on read."""

    fixed = struct.pack(
        "<8sI16s64sQfIII",
        b"DYNAMIS\x00", 1,
        b"1234".ljust(16, b"\x00"),
        b"author".ljust(64, b"\x00"),
        2, 1.0, 1, 12, 1,
    )
    field_descriptor = struct.pack("<32s", b"velocity")
    front_matter_size = struct.pack("<Q", 168)
    header_padding = b"\x00" * 4
    atom_types = bytes([6, 1])
    atom_type_padding = b"\x00" * 6
    velocities = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    pbc_cell = b"\x00" * 40

    bad_file = tmp_path / "missing_position.dynamis"
    bad_file.write_bytes(fixed + field_descriptor + front_matter_size + header_padding
                          + atom_types + atom_type_padding + pbc_cell + velocities)

    with pytest.raises(RuntimeError):
        TrajectoryReader.read(str(bad_file))


def test_reader_missing_file_raises():
    with pytest.raises(RuntimeError):
        TrajectoryReader.read("./does_not_exist.dynamis")


def test_reader_reads_xyz_directly():
    """read() should transparently handle a .xyz source, not just .dynamis."""
    result, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")

    assert result.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), result.atom_types)


def test_reader_accepts_pathlike():
    result, cell, pbc = TrajectoryReader.read(pathlib.Path("tests/xyz.xyz"))
    assert result.num_atoms == 6


# ─── TrajectoryWriter ───────────────────────────────────────────────────────


def test_write_front_matter():
    source, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    TrajectoryWriter(0, b"1234", "author", "./", "trajectory.dynamis", source, cell, pbc, 1.0, 1).flush()

    fixed_format = "<8sI16s64sQfIII32s32s32s32sQ4x6B2x"
    fixed_size = struct.calcsize(fixed_format)

    with open("./trajectory.dynamis", "rb") as f:
        raw_fixed = f.read(fixed_size)
        if len(raw_fixed) < fixed_size:
            raise ValueError(f"File too small to contain a valid header. got {len(raw_fixed)} expected {fixed_size}")

        vals = struct.unpack(fixed_format, raw_fixed)

        h = {
            "magic":           vals[0],
            "version":         vals[1],
            "run_id":          uuid.UUID(bytes=vals[2]),
            "author":          vals[3].decode('utf-8').rstrip('\x00'),
            "total_atoms":     vals[4],
            "time_per_step":   vals[5],
            "steps_per_write": vals[6],
            "bytes_per_atom":  vals[7],
            "field_count":     vals[8],
            "pbc_cell":        vals[9].decode('utf-8').rstrip('\x00'),
            "id":              vals[10].decode('utf-8').rstrip('\x00'),
            "position":        vals[11].decode('utf-8').rstrip('\x00'),
            "velocity":        vals[12].decode('utf-8').rstrip('\x00'),
            "front_matter_size": vals[13],
            "atom_types":      vals[14:20]
        }

        assert h["magic"] == b"DYNAMIS\x00"
        assert h["version"] == 1
        assert h["run_id"] == uuid.UUID(bytes=b"1234".ljust(16, b'\x00'))
        assert h["author"] == "author"
        assert h["total_atoms"] == 6
        assert h["time_per_step"] == 1.0
        assert h["steps_per_write"] == 1
        assert h["bytes_per_atom"] == 32
        assert h["field_count"] == 4
        assert h["pbc_cell"] == "pbc_cell"
        assert h["id"] == "id"
        assert h["position"] == "position"
        assert h["velocity"] == "velocity"
        assert h["front_matter_size"] == 264
        assert h["atom_types"] == (20, 35, 1, 6, 1, 1)


def test_write_frame():
    """Testing 2 objects writing to the same file"""
    env1, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    env2, _, _ = TrajectoryReader.read("tests/xyz.xyz")

    base_writer = TrajectoryWriter(0, b"1234", "author", "./", "trajectory.dynamis", env1, cell, pbc, 1.0, 1)
    writer1 = TrajectoryWriter(1, b"1234", "author", "./", "trajectory.dynamis", env2, cell, pbc, 1.0, 1)

    # split up the envs
    # 6 atoms
    env1.num_atoms = 3
    env1.atom_vertex_numbers = torch.split(env1.atom_vertex_numbers, 3)[0]
    env1.atom_types = torch.split(env1.atom_types, 3)[0]
    env1.atom_positions = torch.split(env1.atom_positions, 3)[0]
    env1.atom_velocities = torch.split(env1.atom_velocities, 3)[0]

    env2.num_atoms = 3
    env2.atom_vertex_numbers = torch.split(env2.atom_vertex_numbers, 3)[1]
    env2.atom_types = torch.split(env2.atom_types, 3)[1]
    env2.atom_positions = torch.split(env2.atom_positions, 3)[1]
    env2.atom_velocities = torch.split(env2.atom_velocities, 3)[1]

    base_writer.update_atom_counts(torch.tensor([3, 3], dtype=torch.uint32))
    writer1.update_atom_counts(torch.tensor([3, 3], dtype=torch.uint32))

    # we are assuming the front matter is fine, just reading the frame
    base_writer.submit(0, env1)
    writer1.submit(0, env2)

    # making sure the file is fully written to
    base_writer.flush()
    writer1.flush()

    # front matter size is 264; each frame starts with a 40-byte pbc_cell
    # block (9 floats + 3 bytes pbc + 1 padding byte) before the atom data
    format = "<264x 9f3B1x 6Q 18f 18f"
    size = struct.calcsize(format)

    with open("./trajectory.dynamis", "rb") as f:
        raw = f.read(size)
        if len(raw) < size:
            raise ValueError(f"File too small to contain a valid header. got {len(raw)} expected {size}")

        vals = struct.unpack(format, raw)

        h = {
            "cell":         vals[0:9],
            "pbc":          vals[9:12],
            "indices":      vals[12:18],
            "positions":    vals[18:36],
            "velocities":   vals[36:54],
        }

        # neither writer passed cell/pbc to submit(), and both were
        # constructed with tests/xyz.xyz's cell/pbc (plain xyz, no Lattice=),
        # so zero defaults apply
        assert h["cell"] == (0.0,) * 9
        assert h["pbc"] == (0, 0, 0)
        assert h["indices"] == (0, 1, 2, 3, 4, 5)
        assert h["positions"] == (0.0, 1.0, 2.0
                                  , 3.0, 4.0, 5.0
                                  , 6.0, 7.0, 8.0
                                  , 0.0, 1.0, 2.0
                                  , 3.0, 4.0, 5.0
                                  , 6.0, 7.0, 8.0)
        assert h["velocities"] == (0.0,) * 18


def test_submit_cell_pbc_roundtrip_and_carry_forward():
    """cell/pbc passed to submit() are persisted per-frame and carried
    forward (without re-passing) on subsequent submit() calls."""
    env, source_cell, source_pbc = TrajectoryReader.read("tests/xyz.xyz")
    with TrajectoryWriter(0, b"1234", "author", "./", "trajectory.dynamis", env, source_cell, source_pbc, 1.0, 1) as writer:
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))

        cell = torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
        pbc = torch.tensor([True, True, False])

        writer.submit(0, env, cell, pbc)  # frame 0: explicit
        writer.submit(1, env)             # frame 1: carried forward, no re-pass

        assert torch.equal(writer.get_cell(), cell.to(torch.float32))
        assert torch.equal(writer.get_pbc(), pbc.to(torch.uint8))

    frame_header_fmt = "<9f3B1x"
    header_size = struct.calcsize(frame_header_fmt)  # 40
    atom_block = 6 * 32  # bytes_per_atom * num_atoms
    frame_size = header_size + atom_block
    front_matter_size = 264  # for the 6-atom xyz.xyz fixture

    with open("./trajectory.dynamis", "rb") as f:
        for step in range(2):
            f.seek(front_matter_size + step * frame_size)
            vals = struct.unpack(frame_header_fmt, f.read(header_size))
            got_cell = torch.tensor(vals[:9]).reshape(3, 3)
            got_pbc = torch.tensor(vals[9:12], dtype=torch.uint8)
            assert torch.equal(got_cell, cell.to(torch.float32))
            assert torch.equal(got_pbc, pbc.to(torch.uint8))


def test_submit_distinct_cells_per_frame():
    """Two frames written with different cells must not collide."""
    env, source_cell, source_pbc = TrajectoryReader.read("tests/xyz.xyz")
    with TrajectoryWriter(0, b"1234", "author", "./", "trajectory.dynamis", env, source_cell, source_pbc, 1.0, 1) as writer:
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))

        cell_a = torch.eye(3) * 5.0
        cell_b = torch.eye(3) * 50.0
        pbc_a = torch.tensor([True, False, False])
        pbc_b = torch.tensor([False, False, True])

        writer.submit(0, env, cell_a, pbc_a)
        writer.submit(1, env, cell_b, pbc_b)

    frame_header_fmt = "<9f3B1x"
    header_size = struct.calcsize(frame_header_fmt)
    atom_block = 6 * 32
    frame_size = header_size + atom_block
    front_matter_size = 264

    with open("./trajectory.dynamis", "rb") as f:
        f.seek(front_matter_size)
        vals_a = struct.unpack(frame_header_fmt, f.read(header_size))
        f.seek(front_matter_size + frame_size)
        vals_b = struct.unpack(frame_header_fmt, f.read(header_size))

    assert torch.tensor(vals_a[:9]).reshape(3, 3).equal(cell_a)
    assert torch.tensor(vals_b[:9]).reshape(3, 3).equal(cell_b)
    assert tuple(vals_a[9:12]) == (1, 0, 0)
    assert tuple(vals_b[9:12]) == (0, 0, 1)


def test_opts_reduced_fields_round_trip():
    """Writing with opts=["positions", "indices"] should drop velocities from the file,
    shrink the header, and read back with velocities zero-filled."""

    values = torch.arange(0, 9, 1).to(torch.float32)
    positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    env, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    with TrajectoryWriter(0, b"1234", "author", "./", "opts_reduced.dynamis", env, cell, pbc, 1.0, 1,
                           ["positions", "indices"]) as writer:
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, env)

    # header should reflect only 2 fields (id + position), bytes_per_atom = 8 + 12 = 20
    fixed_format = "<8sI16s64sQfIII"
    fixed_size = struct.calcsize(fixed_format)
    with open("./opts_reduced.dynamis", "rb") as f:
        vals = struct.unpack(fixed_format, f.read(fixed_size))
        assert vals[7] == 20  # bytes_per_atom
        assert vals[8] == 3   # field_count (pbc_cell + id + position)

    env2, _, _ = TrajectoryReader.read("./opts_reduced.dynamis")

    assert env2.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env2.atom_types)
    assert torch.equal(positions, env2.atom_positions)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env2.atom_vertex_numbers)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), env2.atom_velocities)


def test_opts_missing_indices_assumes_sequential_order():
    """Writing without "indices" should read back as sequential [0, num_atoms)."""

    env, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    with TrajectoryWriter(0, b"1234", "author", "./", "opts_no_indices.dynamis", env, cell, pbc, 1.0, 1,
                           ["positions", "velocities"]) as writer:
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, env)

    env2, _, _ = TrajectoryReader.read("./opts_no_indices.dynamis")

    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env2.atom_vertex_numbers)


def test_opts_positions_required_on_write():
    """opts that exclude "positions" should be rejected."""
    env, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    with pytest.raises(RuntimeError):
        TrajectoryWriter(0, b"1234", "author", "./", "opts_invalid.dynamis", env, cell, pbc, 1.0, 1,
                          ["indices", "velocities"])


def test_opts_unknown_option_rejected():
    env, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")
    with pytest.raises(RuntimeError):
        TrajectoryWriter(0, b"1234", "author", "./", "opts_unknown.dynamis", env, cell, pbc, 1.0, 1,
                          ["positions", "not_a_real_field"])


def test_writer_reader_round_trip():
    """Writing then reading a MolecularInfo through a .dynamis file should
    round-trip all fields intact."""
    source, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")

    with TrajectoryWriter(0, b"1234", "author", "./", "io_round_trip.dynamis", source, cell, pbc, 1.0, 1) as writer:
        writer.update_atom_counts(torch.tensor([source.num_atoms], dtype=torch.uint32))
        writer.submit(0, source)

    result, result_cell, result_pbc = TrajectoryReader.read("./io_round_trip.dynamis")

    assert result.num_atoms == source.num_atoms
    assert torch.equal(result.atom_types, source.atom_types)
    assert torch.equal(result.atom_positions, source.atom_positions)
    assert torch.equal(result.atom_velocities, source.atom_velocities)
    assert torch.equal(result.atom_vertex_numbers, source.atom_vertex_numbers)


def test_writer_reader_round_trip_with_opts():
    """opts=["positions"] should drop indices/velocities from the write; reader
    should synthesize sequential indices and zero-filled velocities."""
    source, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")

    with TrajectoryWriter(0, b"1234", "author", "./", "io_round_trip_opts.dynamis", source, cell, pbc, 1.0, 1,
                           ["positions"]) as writer:
        writer.update_atom_counts(torch.tensor([source.num_atoms], dtype=torch.uint32))
        writer.submit(0, source)

    result, _, _ = TrajectoryReader.read("./io_round_trip_opts.dynamis")

    assert torch.equal(result.atom_positions, source.atom_positions)
    assert torch.equal(result.atom_vertex_numbers, torch.arange(0, source.num_atoms, dtype=torch.int64))
    assert torch.equal(result.atom_velocities, torch.zeros_like(source.atom_velocities))


def test_writer_accepts_pathlike():
    """Both output_dir and output_file should accept pathlib.Path."""
    source, cell, pbc = TrajectoryReader.read("tests/xyz.xyz")

    out_dir = pathlib.Path(".")
    out_file = pathlib.Path("io_pathlike.dynamis")
    with TrajectoryWriter(0, b"1234", "author", out_dir, out_file, source, cell, pbc, 1.0, 1) as writer:
        writer.update_atom_counts(torch.tensor([source.num_atoms], dtype=torch.uint32))
        writer.submit(0, source)

    result, _, _ = TrajectoryReader.read(pathlib.Path("io_pathlike.dynamis"))
    assert torch.equal(result.atom_positions, source.atom_positions)


# ─── distributed multi-rank round trip (real multi-process, no CUDA needed) ───
#
# Mirrors the multi-process pattern used by test_ipc_exchanger.py's
# TestIPCExchanger, except here "distributed" only means separate OS processes
# each writing their own local slice of atoms to a shared .dynamis file --
# exactly how distnnp/simulator.py drives one TrajectoryWriter per rank. No
# extra locking is needed beyond a single barrier: rank 0's O_TRUNC happens in
# the constructor, and every rank's submit() writes into a disjoint byte range
# of the same file, so as long as the truncate has already happened no two
# ranks ever touch the same bytes.
#
# Each rank also writes several frames (not just one), since submit()'s offset
# is step * total_atoms * bytes_per_atom -- a single-frame test can't tell a
# correct per-step offset apart from one that always lands on frame 0.

FRAME_POSITION_STEP = 100.0  # added per frame index so frames are distinguishable


def _dist_write_worker(rank, output_dir, filename, source_file, atom_counts, opts, num_frames, barrier, results):
    try:
        env, cell, pbc = TrajectoryReader.read(source_file)
        writer = TrajectoryWriter(rank, b"1234", "author", output_dir, filename, env, cell, pbc, 1.0, 1, opts)
        barrier.wait()  # guarantee rank 0's O_TRUNC happened before anyone writes

        offsets = [0]
        for c in atom_counts:
            offsets.append(offsets[-1] + c)
        lo, hi = offsets[rank], offsets[rank + 1]

        writer.update_atom_counts(torch.tensor(atom_counts, dtype=torch.uint32))

        local = MolecularInfo()
        local.num_atoms = hi - lo
        local.atom_vertex_numbers = env.atom_vertex_numbers[lo:hi].clone()
        local.atom_types = env.atom_types[lo:hi].clone()
        local.atom_velocities = env.atom_velocities[lo:hi].clone()

        base_positions = env.atom_positions[lo:hi].clone()
        for step in range(num_frames):
            local.atom_positions = base_positions + step * FRAME_POSITION_STEP
            writer.submit(step, local)

        writer.flush()
        results[rank] = None
    except Exception as e:
        results[rank] = f"{type(e).__name__}: {e}"


def _run_distributed_write(world_size, output_dir, filename, atom_counts, opts=None, num_frames=1):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results = manager.dict()
    barrier = mp.Barrier(world_size)

    processes = []
    for rank in range(world_size):
        proc = mp.Process(
            target=_dist_write_worker,
            args=(rank, output_dir, filename, "tests/xyz.xyz", atom_counts, opts or [], num_frames, barrier, results),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join(timeout=60)

    for rank in range(world_size):
        assert rank in results, f"Rank {rank} did not report a result (crashed?)"
        assert results[rank] is None, f"Rank {rank} failed: {results[rank]}"


def _read_raw_frame(path, frame_index, num_atoms, has_indices, has_velocities):
    """Manually parses the .dynamis binary layout (mirrors
    read_dynamis_frame in trajectory_reader.cpp) to pull back one specific
    frame by index, as a cross-check independent of TrajectoryReader.read()."""
    with open(path, "rb") as f:
        f.seek(112)
        field_count, = struct.unpack("<I", f.read(4))
        f.seek(116 + field_count * 32)
        front_matter_size, = struct.unpack("<Q", f.read(8))

    pbc_cell_block_size = 40
    bytes_per_atom = (8 if has_indices else 0) + 12 + (12 if has_velocities else 0)
    frame_size = pbc_cell_block_size + num_atoms * bytes_per_atom
    cursor = front_matter_size + frame_index * frame_size + pbc_cell_block_size

    with open(path, "rb") as f:
        ids = None
        if has_indices:
            f.seek(cursor)
            ids = torch.tensor(struct.unpack(f"<{num_atoms}q", f.read(num_atoms * 8)), dtype=torch.int64)
            cursor += num_atoms * 8

        f.seek(cursor)
        positions = torch.tensor(
            struct.unpack(f"<{num_atoms * 3}f", f.read(num_atoms * 3 * 4)), dtype=torch.float32
        ).reshape(num_atoms, 3)
        cursor += num_atoms * 3 * 4

        velocities = None
        if has_velocities:
            f.seek(cursor)
            velocities = torch.tensor(
                struct.unpack(f"<{num_atoms * 3}f", f.read(num_atoms * 3 * 4)), dtype=torch.float32
            ).reshape(num_atoms, 3)

    return ids, positions, velocities


def test_distributed_multi_rank_write_read_round_trip():
    """3 real OS processes, each owning one rank's TrajectoryWriter pointed at
    the same shared file, write several frames, submitting only their own slice
    of atoms each time. Every frame -- not just the last -- should be
    individually addressable via TrajectoryReader.read(path, frame_index)."""
    filename = "dist_multirank_full.dynamis"
    num_frames = 3

    _run_distributed_write(
        world_size=3, output_dir="./", filename=filename, atom_counts=[2, 2, 2], num_frames=num_frames,
    )

    values = torch.arange(0, 9, 1).to(torch.float32)
    base_positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    result, _, _ = TrajectoryReader.read(f"./{filename}")  # default: last frame

    assert result.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), result.atom_types)
    assert torch.equal(base_positions + (num_frames - 1) * FRAME_POSITION_STEP, result.atom_positions)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), result.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), result.atom_vertex_numbers)

    # every individual frame is addressable and distinct, not just the last
    for step in range(num_frames):
        frame_result, _, _ = TrajectoryReader.read(f"./{filename}", step)
        assert torch.equal(base_positions + step * FRAME_POSITION_STEP, frame_result.atom_positions)
        assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), frame_result.atom_vertex_numbers)
        assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), frame_result.atom_velocities)

        # cross-checked against a from-scratch parse of the raw bytes
        ids, positions, velocities = _read_raw_frame(
            f"./{filename}", step, num_atoms=6, has_indices=True, has_velocities=True,
        )
        assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), ids)
        assert torch.equal(base_positions + step * FRAME_POSITION_STEP, positions)
        assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), velocities)


def test_distributed_multi_rank_write_read_round_trip_reduced_opts():
    """Same multi-frame distributed setup, but every rank writes with
    opts=["positions", "indices"]: velocities should be dropped from the file
    entirely, every frame should still read back distinctly, and the last
    frame should zero-fill velocities on read."""
    filename = "dist_multirank_opts.dynamis"
    num_frames = 2

    _run_distributed_write(
        world_size=2, output_dir="./", filename=filename, atom_counts=[3, 3],
        opts=["positions", "indices"], num_frames=num_frames,
    )

    values = torch.arange(0, 9, 1).to(torch.float32)
    base_positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    result, _, _ = TrajectoryReader.read(f"./{filename}")

    assert result.num_atoms == 6
    assert torch.equal(base_positions + (num_frames - 1) * FRAME_POSITION_STEP, result.atom_positions)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), result.atom_vertex_numbers)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), result.atom_velocities)

    for step in range(num_frames):
        frame_result, _, _ = TrajectoryReader.read(f"./{filename}", step)
        assert torch.equal(base_positions + step * FRAME_POSITION_STEP, frame_result.atom_positions)

        ids, positions, velocities = _read_raw_frame(
            f"./{filename}", step, num_atoms=6, has_indices=True, has_velocities=False,
        )
        assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), ids)
        assert torch.equal(base_positions + step * FRAME_POSITION_STEP, positions)
        assert velocities is None


# ─── translate_dynamis_to_xyz ──────────────────────────────────────────────


def _write_multiframe_dynamis(path, num_frames=3, opts=None):
    source, _, _ = TrajectoryReader.read("tests/xyz.xyz")
    cell = torch.eye(3) * 12.0
    pbc = torch.tensor([True, False, True])

    directory, filename = "./", path
    with TrajectoryWriter(0, b"1234", "author", directory, filename, source, cell, pbc, 1.0, 1, opts) as writer:
        writer.update_atom_counts(torch.tensor([source.num_atoms], dtype=torch.uint32))
        base_positions = source.atom_positions.clone()
        for step in range(num_frames):
            frame = MolecularInfo()
            frame.num_atoms = source.num_atoms
            frame.atom_vertex_numbers = source.atom_vertex_numbers
            frame.atom_types = source.atom_types
            frame.atom_positions = base_positions + step * FRAME_POSITION_STEP
            frame.atom_velocities = source.atom_velocities + step
            writer.submit(step, frame, cell * (step + 1), pbc)

    return source


def test_translate_dynamis_to_xyz_multi_frame():
    num_frames = 3
    source = _write_multiframe_dynamis("translate_source.dynamis", num_frames=num_frames)

    translate_dynamis_to_xyz("./translate_source.dynamis", "./translate_out.xyz")

    with open("./translate_out.xyz") as f:
        lines = f.readlines()

    num_atoms = int(source.num_atoms)
    lines_per_frame = num_atoms + 2
    assert len(lines) == lines_per_frame * num_frames

    for step in range(num_frames):
        block = lines[step * lines_per_frame:(step + 1) * lines_per_frame]
        assert int(block[0].strip()) == num_atoms
        # vel, not momenta: MolecularInfo carries velocities (A/fs)
        assert "Properties=species:S:1:pos:R:3:vel:R:3 " in block[1]
        for line in block[2:]:
            assert len(line.split()) == 7  # species + 3 pos + 3 vel

        info, cell, pbc = TrajectoryReader.read("./translate_out.xyz", step)
        assert torch.equal(source.atom_types, info.atom_types)
        assert torch.allclose(source.atom_positions + step * FRAME_POSITION_STEP, info.atom_positions)
        assert torch.allclose(source.atom_velocities + step, info.atom_velocities)
        assert torch.equal(torch.eye(3) * 12.0 * (step + 1), cell)
        assert torch.equal(torch.tensor([1, 0, 1], dtype=torch.uint8), pbc)


def test_translate_dynamis_to_xyz_opts_drops_velocities():
    _write_multiframe_dynamis("translate_opts.dynamis", num_frames=2)

    translate_dynamis_to_xyz("./translate_opts.dynamis", "./translate_opts_out.xyz", ["positions"])

    with open("./translate_opts_out.xyz") as f:
        lines = f.readlines()

    num_atoms = 6
    lines_per_frame = num_atoms + 2
    assert len(lines) == lines_per_frame * 2

    for step in range(2):
        block = lines[step * lines_per_frame:(step + 1) * lines_per_frame]
        assert block[1].strip().startswith("Lattice=")
        assert "Properties=species:S:1:pos:R:3 " in block[1]
        for line in block[2:]:
            assert len(line.split()) == 4  # species + 3 pos, no velocities


def test_translate_dynamis_to_xyz_round_trip():
    """translate -> re-read every frame from the xyz -> compare against
    TrajectoryReader.read() on the original .dynamis at the same frame_index."""
    num_frames = 3
    _write_multiframe_dynamis("translate_roundtrip.dynamis", num_frames=num_frames)

    translate_dynamis_to_xyz("./translate_roundtrip.dynamis", "./translate_roundtrip.xyz")

    for step in range(num_frames):
        dynamis_info, dynamis_cell, dynamis_pbc = TrajectoryReader.read("./translate_roundtrip.dynamis", step)
        xyz_info, xyz_cell, xyz_pbc = TrajectoryReader.read("./translate_roundtrip.xyz", step)

        assert torch.equal(dynamis_info.atom_types, xyz_info.atom_types)
        assert torch.allclose(dynamis_info.atom_positions, xyz_info.atom_positions)
        assert torch.allclose(dynamis_info.atom_velocities, xyz_info.atom_velocities)
        assert torch.equal(dynamis_cell, xyz_cell)
        assert torch.equal(dynamis_pbc, xyz_pbc)
