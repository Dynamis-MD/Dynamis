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
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.uint32), env.atom_types)
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(velocities, env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1), env.atom_vertex_numbers)

def test_xyz_io():
    """Regular xyz test"""

    # the 1 makes sure it only loads in the file and doesn't do front_matter stuff
    writer = TrajectoryWriter(1, b"1234", "author", "./")

    env = writer.initialize("tests/xyz.xyz", 1.0, 1)
    values = torch.arange(0, 9, 1)

    positions = torch.concat((values, values), dim=0)
    positions = positions.reshape([6, 3])
    assert env.num_atoms == 6
    assert torch.equal(torch.tensor([20, 35, 1, 6, 1, 1], dtype=torch.uint32), env.atom_types)
    assert torch.equal(positions, env.atom_positions)
    assert torch.equal(torch.zeros([6, 3]), env.atom_velocities)
    assert torch.equal(torch.arange(0, 6, 1), env.atom_vertex_numbers)

def test_write_front_matter():
    writer = TrajectoryWriter(0, b"1234", "author", "./")

    # since it is rank 0 it writes to the file
    env = writer.initialize("tests/xyz.xyz", 1.0, 1)

    fixed_format = "<8sI16s64sQfIII32s32s32sQ36x6I40x"
    fixed_size = struct.calcsize(fixed_format)

    with open("./trajectory.xyzbin", "rb") as f:
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
            "id":              vals[9].decode('utf-8').rstrip('\x00'),
            "position":        vals[10].decode('utf-8').rstrip('\x00'),
            "velocity":        vals[11].decode('utf-8').rstrip('\x00'),
            "front_matter_size": vals[12],
            "atom_types":      vals[13:19]
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
        assert h["field_count"] == 3
        assert h["id"] == "id"
        assert h["position"] == "position"
        assert h["velocity"] == "velocity"
        assert h["front_matter_size"] == 320
        assert h["atom_types"] == (20, 35, 1, 6, 1, 1)
