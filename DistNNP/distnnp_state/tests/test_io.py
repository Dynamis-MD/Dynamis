"""Tests for io integration"""

import torch
import pytest
import struct
import uuid

from distnnp_state import (MolecularInfo, TrajectoryWriter)

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

