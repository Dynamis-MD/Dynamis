"""Tests for io integration"""

import pathlib
import struct
import uuid

import torch
import torch.multiprocessing as mp
import pytest

from distnnp_state import (MolecularInfo, TrajectoryWriter)
from distnnp_state.io import MolecularInfoReader, MolecularInfoWriter

def test_xyz_extended_io():
    """Extended xyz test"""

    # the 1 makes sure it only loads in the file and doesn't do front_matter stuff
    writer = TrajectoryWriter(1, b"1234", "author", "./")

    env = writer.initialize("tests/extended_xyz.xyz", 1.0, 1)
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

    assert torch.equal(torch.diag(torch.tensor([15.0, 30.0, 30.0])), writer.get_cell())
    assert torch.equal(torch.ones(3, dtype=torch.uint8), writer.get_pbc())

def test_xyz_io():
    """Regular xyz test"""

    # the 1 makes sure it only loads in the file and doesn't do front_matter stuff
    writer = TrajectoryWriter(1, b"1234", "author", "./")

    env = writer.initialize("tests/xyz.xyz", 1.0, 1)
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape([6, 3])
    assert env.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env.atom_types)
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(torch.zeros([6, 3]), env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env.atom_vertex_numbers)

    assert torch.equal(torch.zeros(3, 3), writer.get_cell())
    assert torch.equal(torch.zeros(3, dtype=torch.uint8), writer.get_pbc())

def test_distnnp_io():
    """Distnnp test"""

    # test setup
    values = torch.arange(0, 9, 1).to(torch.float32)
    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape([6, 3]).to(torch.float32)

    with TrajectoryWriter(0, b"1234", "author", "./") as writer:
        env = writer.initialize("tests/xyz.xyz", 1.0, 1)

        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, env)
        writer.flush()

    writer2 = TrajectoryWriter(1, b"1234", "author", "./")
    env2 = writer2.initialize("./trajectory.distnnp", 1.0, 1)

    assert env2.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env2.atom_types)
    assert torch.equal(positions, env2.atom_positions)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), env2.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env2.atom_vertex_numbers)


def test_write_front_matter():
    writer = TrajectoryWriter(0, b"1234", "author", "./")

    # since it is rank 0 it writes to the file
    env = writer.initialize("tests/xyz.xyz", 1.0, 1)

    fixed_format = "<8sI16s64sQfIII32s32s32s32sQ4x6B2x"
    fixed_size = struct.calcsize(fixed_format)

    with open("./trajectory.distnnp", "rb") as f:
        raw_fixed = f.read(fixed_size)
        if len(raw_fixed) < fixed_size:
            raise ValueError(f"File too small to contain a valid header. got {len(raw_fixed)} expected {fixed_size}")

        # Unpack the raw bytes into a tuple
        vals = struct.unpack(fixed_format, raw_fixed)

        # Mapping tuple to a dictionary for readability
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

        # Check to see if it matches expected
        assert h["magic"] == b"DISTNNP\x00"
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
    base_writer = TrajectoryWriter(0, b"1234", "author", "./")
    writer1 = TrajectoryWriter(1, b"1234", "author", "./")

    env1 = base_writer.initialize("tests/xyz.xyz", 1.0, 1)
    env2 = writer1.initialize("tests/xyz.xyz", 1.0, 1)

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

    with open("./trajectory.distnnp", "rb") as f:
        raw = f.read(size)
        if len(raw) < size:
            raise ValueError(f"File too small to contain a valid header. got {len(raw)} expected {size}")

        # Unpack the raw bytes into a tuple
        vals = struct.unpack(format, raw)

        # Mapping tuple to a dictionary for readability
        h = {
            "cell":         vals[0:9],
            "pbc":          vals[9:12],
            "indices":      vals[12:18],
            "positions":    vals[18:36],
            "velocities":   vals[36:54],
        }

        # neither writer passed cell/pbc to submit(), and both loaded from
        # tests/xyz.xyz (plain xyz, no Lattice=), so defaults apply
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
    with TrajectoryWriter(0, b"1234", "author", "./") as writer:
        env = writer.initialize("tests/xyz.xyz", 1.0, 1)
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

    with open("./trajectory.distnnp", "rb") as f:
        for step in range(2):
            f.seek(front_matter_size + step * frame_size)
            vals = struct.unpack(frame_header_fmt, f.read(header_size))
            got_cell = torch.tensor(vals[:9]).reshape(3, 3)
            got_pbc = torch.tensor(vals[9:12], dtype=torch.uint8)
            assert torch.equal(got_cell, cell.to(torch.float32))
            assert torch.equal(got_pbc, pbc.to(torch.uint8))

def test_submit_distinct_cells_per_frame():
    """Two frames written with different cells must not collide."""
    with TrajectoryWriter(0, b"1234", "author", "./") as writer:
        env = writer.initialize("tests/xyz.xyz", 1.0, 1)
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

    with open("./trajectory.distnnp", "rb") as f:
        f.seek(front_matter_size)
        vals_a = struct.unpack(frame_header_fmt, f.read(header_size))
        f.seek(front_matter_size + frame_size)
        vals_b = struct.unpack(frame_header_fmt, f.read(header_size))

    assert torch.tensor(vals_a[:9]).reshape(3, 3).equal(cell_a)
    assert torch.tensor(vals_b[:9]).reshape(3, 3).equal(cell_b)
    assert tuple(vals_a[9:12]) == (1, 0, 0)
    assert tuple(vals_b[9:12]) == (0, 0, 1)

def test_write_xyz():
    writer = TrajectoryWriter(1, b"1234", "author", "./")

    env = writer.initialize("tests/extended_xyz.xyz", 1.0, 1)

    writer.write_to_xyz(env, "output.xyz");

    with open("tests/extended_xyz.xyz", 'r') as orig:
        with open("output.xyz", 'r') as fin:
            orig_lines = orig.readlines()
            fin_lines = fin.readlines()

            # num atoms
            orig_atoms = int(orig_lines[0].strip())
            fin_atoms = int(fin_lines[0].strip())
            assert(orig_atoms == fin_atoms)

            orig_data = [line.split() for line in orig_lines[2:2+orig_atoms]]
            fin_data = [line.split() for line in fin_lines[2:2+fin_atoms]]

            orig_syms = [row[0] for row in orig_data]
            fin_syms = [row[0] for row in fin_data]
            assert(orig_syms == fin_syms)

            orig_attribs = torch.tensor([[float(y) for y in x] for x in [row[1:7] for row in orig_data]], dtype=torch.float32)
            fin_attribs= torch.tensor([[float(y) for y in x] for x in [row[1:7] for row in fin_data]], dtype=torch.float32)
            assert torch.equal(orig_attribs, fin_attribs)


def test_opts_reduced_fields_round_trip():
    """Writing with opts=["positions", "indices"] should drop velocities from the file,
    shrink the header, and read back with velocities zero-filled."""

    values = torch.arange(0, 9, 1).to(torch.float32)
    positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    with TrajectoryWriter(0, b"1234", "author", "./", "opts_reduced.distnnp") as writer:
        env = writer.initialize("tests/xyz.xyz", 1.0, 1, ["positions", "indices"])
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, env)
        writer.flush()

    # header should reflect only 2 fields (id + position), bytes_per_atom = 8 + 12 = 20
    fixed_format = "<8sI16s64sQfIII"
    fixed_size = struct.calcsize(fixed_format)
    with open("./opts_reduced.distnnp", "rb") as f:
        vals = struct.unpack(fixed_format, f.read(fixed_size))
        assert vals[7] == 20  # bytes_per_atom
        assert vals[8] == 3   # field_count (pbc_cell + id + position)

    writer2 = TrajectoryWriter(1, b"1234", "author", "./")
    env2 = writer2.initialize("./opts_reduced.distnnp", 1.0, 1)

    assert env2.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), env2.atom_types)
    assert torch.equal(positions, env2.atom_positions)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env2.atom_vertex_numbers)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), env2.atom_velocities)


def test_opts_missing_indices_assumes_sequential_order():
    """Writing without "indices" should read back as sequential [0, num_atoms)."""

    with TrajectoryWriter(0, b"1234", "author", "./", "opts_no_indices.distnnp") as writer:
        env = writer.initialize("tests/xyz.xyz", 1.0, 1, ["positions", "velocities"])
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, env)
        writer.flush()

    writer2 = TrajectoryWriter(1, b"1234", "author", "./")
    env2 = writer2.initialize("./opts_no_indices.distnnp", 1.0, 1)

    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), env2.atom_vertex_numbers)


def test_opts_positions_required_on_write():
    """opts that exclude "positions" should be rejected."""

    writer = TrajectoryWriter(0, b"1234", "author", "./", "opts_invalid.distnnp")
    with pytest.raises(RuntimeError):
        writer.initialize("tests/xyz.xyz", 1.0, 1, ["indices", "velocities"])


def test_opts_unknown_option_rejected():
    writer = TrajectoryWriter(0, b"1234", "author", "./", "opts_unknown.distnnp")
    with pytest.raises(RuntimeError):
        writer.initialize("tests/xyz.xyz", 1.0, 1, ["positions", "not_a_real_field"])


def test_read_missing_position_field_raises():
    """A hand-crafted file whose only field is "velocity" (no "position") must
    fail to load, since positions are mandatory on read."""

    fixed = struct.pack(
        "<8sI16s64sQfIII",
        b"DISTNNP\x00", 1,
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

    with open("./missing_position.distnnp", "wb") as f:
        f.write(fixed + field_descriptor + front_matter_size + header_padding
                 + atom_types + atom_type_padding + velocities)

    writer = TrajectoryWriter(1, b"1234", "author", "./")
    with pytest.raises(RuntimeError):
        writer.initialize("./missing_position.distnnp", 1.0, 1)


def test_write_to_xyz_opts_drops_velocities():
    """write_to_xyz with opts=["positions"] should omit the momenta column/header."""

    writer = TrajectoryWriter(1, b"1234", "author", "./")
    env = writer.initialize("tests/extended_xyz.xyz", 1.0, 1)

    writer.write_to_xyz(env, "output_opts.xyz", ["positions"])

    with open("output_opts.xyz", "r") as f:
        lines = f.readlines()

    assert lines[1].strip() == "Properties=species:S:1:pos:R:3"
    num_atoms = int(lines[0].strip())
    for line in lines[2:2 + num_atoms]:
        # species + 3 position columns, no velocity columns
        assert len(line.split()) == 4


# ─── distnnp_state.io: MolecularInfoWriter / MolecularInfoReader ──────────────


def test_molecular_info_writer_reader_round_trip():
    """write_struct + read_struct should round-trip a MolecularInfo through a
    .distnnp file in a single call each, with all fields intact."""
    loader = TrajectoryWriter(1, b"1234", "author", "./")
    source = loader.initialize("tests/xyz.xyz", 1.0, 1)

    MolecularInfoWriter.write_struct(source, "./io_round_trip.distnnp")
    result = MolecularInfoReader.read_struct("./io_round_trip.distnnp")

    assert result.num_atoms == source.num_atoms
    assert torch.equal(result.atom_types, source.atom_types)
    assert torch.equal(result.atom_positions, source.atom_positions)
    assert torch.equal(result.atom_velocities, source.atom_velocities)
    assert torch.equal(result.atom_vertex_numbers, source.atom_vertex_numbers)


def test_molecular_info_writer_reader_round_trip_with_opts():
    """opts=["positions"] should drop indices/velocities from the write; reader
    should synthesize sequential indices and zero-filled velocities."""
    loader = TrajectoryWriter(1, b"1234", "author", "./")
    source = loader.initialize("tests/xyz.xyz", 1.0, 1)

    MolecularInfoWriter.write_struct(source, "./io_round_trip_opts.distnnp", opts=["positions"])
    result = MolecularInfoReader.read_struct("./io_round_trip_opts.distnnp")

    assert torch.equal(result.atom_positions, source.atom_positions)
    assert torch.equal(result.atom_vertex_numbers, torch.arange(0, source.num_atoms, dtype=torch.int64))
    assert torch.equal(result.atom_velocities, torch.zeros_like(source.atom_velocities))


def test_molecular_info_reader_reads_xyz_directly():
    """read_struct should transparently handle a .xyz source, not just .distnnp."""
    result = MolecularInfoReader.read_struct("tests/xyz.xyz")

    assert result.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), result.atom_types)


def test_molecular_info_reader_missing_file_raises():
    with pytest.raises(AssertionError):
        MolecularInfoReader.read_struct("./does_not_exist.distnnp")


def test_molecular_info_writer_reader_accepts_pathlike():
    """Both halves should accept pathlib.Path, not just str (write_struct via
    os.path.split, read_struct via os.fspath before crossing into C++)."""
    loader = TrajectoryWriter(1, b"1234", "author", "./")
    source = loader.initialize("tests/xyz.xyz", 1.0, 1)

    out_path = pathlib.Path("./io_pathlike.distnnp")
    MolecularInfoWriter.write_struct(source, out_path)
    result = MolecularInfoReader.read_struct(out_path)

    assert torch.equal(result.atom_positions, source.atom_positions)


# ─── distributed multi-rank round trip (real multi-process, no CUDA needed) ───
#
# Mirrors the multi-process pattern used by test_ipc_exchanger.py's
# TestIPCExchanger, except here "distributed" only means separate OS processes
# each writing their own local slice of atoms to a shared .distnnp file --
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
        writer = TrajectoryWriter(rank, b"1234", "author", output_dir, filename)
        barrier.wait()  # guarantee rank 0's O_TRUNC happened before anyone writes

        env = writer.initialize(source_file, 1.0, 1, opts)

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
    """Manually parses the .distnnp binary layout (mirrors
    TrajectoryWriter::load_distnnp_file in trajectory_writer.cpp) to pull back
    one specific frame by index. TrajectoryWriter.initialize()/MolecularInfoReader
    only ever return the *last* frame in the file (they seek from EOF), so this
    is the only way to verify that earlier frames were written correctly too."""
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
    of atoms each time. Every frame -- not just the last -- should read back
    correctly, proving each rank's writes land in the right per-step offset."""
    filename = "dist_multirank_full.distnnp"
    num_frames = 3

    _run_distributed_write(
        world_size=3, output_dir="./", filename=filename, atom_counts=[2, 2, 2], num_frames=num_frames,
    )

    values = torch.arange(0, 9, 1).to(torch.float32)
    base_positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    # MolecularInfoReader (and TrajectoryWriter.initialize on an existing file)
    # always reads back the last frame written.
    result = MolecularInfoReader.read_struct(f"./{filename}")

    assert result.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.long), result.atom_types)
    assert torch.equal(base_positions + (num_frames - 1) * FRAME_POSITION_STEP, result.atom_positions)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), result.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), result.atom_vertex_numbers)

    # every earlier frame must still be intact and distinguishable too
    for step in range(num_frames):
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
    filename = "dist_multirank_opts.distnnp"
    num_frames = 2

    _run_distributed_write(
        world_size=2, output_dir="./", filename=filename, atom_counts=[3, 3],
        opts=["positions", "indices"], num_frames=num_frames,
    )

    values = torch.arange(0, 9, 1).to(torch.float32)
    base_positions = torch.concat((values, values), dim=0).reshape([6, 3]).to(torch.float32)

    result = MolecularInfoReader.read_struct(f"./{filename}")

    assert result.num_atoms == 6
    assert torch.equal(base_positions + (num_frames - 1) * FRAME_POSITION_STEP, result.atom_positions)
    assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), result.atom_vertex_numbers)
    assert torch.equal(torch.zeros([6, 3], dtype=torch.float32), result.atom_velocities)

    for step in range(num_frames):
        ids, positions, velocities = _read_raw_frame(
            f"./{filename}", step, num_atoms=6, has_indices=True, has_velocities=False,
        )
        assert torch.equal(torch.arange(0, 6, 1).to(torch.int64), ids)
        assert torch.equal(base_positions + step * FRAME_POSITION_STEP, positions)
        assert velocities is None

