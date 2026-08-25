"""Tests for Timer and TimingRegistry (distnnp_state.timing)."""

import json
import os
import time

import pytest
import torch

from distnnp_state import Timer, TimingRegistry, TimerClass, Where, Level, TIMING_ENABLED

pytestmark = pytest.mark.skipif(
    not TIMING_ENABLED,
    reason="distnnp_state built with -DDISTNNP_TIMING=OFF",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

RUN_ID = b"\x01" * 16


def init_registry(tmp_path, rank=0):
    TimingRegistry.instance().init(
        rank, rank, 1, RUN_ID, -1, str(tmp_path)
    )
    Timer.set_level(Level.Basic)


def flush_and_read(tmp_path, rank=0):
    TimingRegistry.instance().flush()
    path = tmp_path / f"timing_rank{rank}.jsonl"
    lines = path.read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


# ── level Off ────────────────────────────────────────────────────────────────

def test_level_off_produces_no_records(tmp_path):
    init_registry(tmp_path)
    Timer.set_level(Level.Off)

    t = Timer()
    t.start("iteration", TimerClass.Other)
    t.stop()

    records = flush_and_read(tmp_path)
    assert all(r["type"] != "rec" for r in records)


# ── basic host timer ──────────────────────────────────────────────────────────

def test_host_timer_records_elapsed(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Other)
    time.sleep(0.01)
    t.stop()

    records = flush_and_read(tmp_path)
    assert records[0]["type"] == "meta"
    rec = records[1]
    assert rec["type"] == "rec"
    assert rec["key"] == "iteration"
    assert rec["class"] == "other"
    assert rec["where"] == "host"
    assert rec["time_us"] >= 10_000.0  # at least 10 ms


def test_elapsed_time_is_positive(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Other)
    t.stop()

    records = flush_and_read(tmp_path)
    assert records[1]["time_us"] > 0.0


# ── reuse across steps ────────────────────────────────────────────────────────

def test_timer_reuse_produces_one_record_per_stop(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in range(5):
        Timer.set_step(step)
        t.start("iteration", TimerClass.Other)
        t.stop()

    records = flush_and_read(tmp_path)
    recs = [r for r in records if r["type"] == "rec"]
    assert len(recs) == 5
    for i, rec in enumerate(recs):
        assert rec["step"] == i


# ── set_step stamps records ───────────────────────────────────────────────────

def test_set_step_stamps_records(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in [0, 10, 99]:
        Timer.set_step(step)
        t.start("iteration", TimerClass.Other)
        t.stop()

    records = flush_and_read(tmp_path)
    steps = [r["step"] for r in records if r["type"] == "rec"]
    assert steps == [0, 10, 99]


# ── call_order in header ──────────────────────────────────────────────────────

def test_call_order_reflects_first_seen_order(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for key in ["iteration", "build_graph", "iteration", "halo_exchange"]:
        t.start(key, TimerClass.Other)
        t.stop()

    records = flush_and_read(tmp_path)
    co = records[-1]
    assert co["type"] == "call_order"
    assert co["order"] == ["iteration", "build_graph", "halo_exchange"]


# ── metadata header ───────────────────────────────────────────────────────────

def test_header_fields(tmp_path):
    init_registry(tmp_path, rank=3)
    Timer.set_level(Level.Basic)

    t = Timer()
    t.start("iteration", TimerClass.Other)
    t.stop()

    records = flush_and_read(tmp_path, rank=3)
    hdr = records[0]
    assert hdr["type"] == "meta"
    assert hdr["rank"] == 3
    assert hdr["local_rank"] == 3
    assert hdr["world_size"] == 1
    assert hdr["device_id"] == -1
    assert hdr["run_id"] == "01" * 16
    assert "date" in hdr
    assert "host" in hdr
    assert "pid" in hdr
    assert "version" in hdr


# ── per-rank file naming ──────────────────────────────────────────────────────

def test_output_file_named_by_rank(tmp_path):
    for rank in [0, 2, 7]:
        TimingRegistry.instance().init(rank, rank, 8, RUN_ID, -1, str(tmp_path))
        Timer.set_level(Level.Basic)
        t = Timer()
        t.start("iteration", TimerClass.Other)
        t.stop()
        TimingRegistry.instance().flush()
        assert (tmp_path / f"timing_rank{rank}.jsonl").exists()


# ── TimerClass values round-trip through the file ────────────────────────────

@pytest.mark.parametrize("cls,expected", [
    (TimerClass.Compute, "compute"),
    (TimerClass.Comm,    "comm"),
    (TimerClass.Wait,    "wait"),
    (TimerClass.Other,   "other"),
])
def test_timer_class_written_correctly(tmp_path, cls, expected):
    init_registry(tmp_path)

    t = Timer()
    t.start("region", cls)
    t.stop()

    records = flush_and_read(tmp_path)
    assert records[1]["class"] == expected


# ── device timers ─────────────────────────────────────────────────────────────

@requires_cuda
def test_device_timer_produces_device_record(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
    # Give the GPU a trivial workload so there is something to time.
    torch.cuda.synchronize(0)
    t.stop()

    records = flush_and_read(tmp_path)
    recs = [r for r in records if r["type"] == "rec"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["key"] == "gpu_region"
    assert rec["where"] == "device"
    assert rec["class"] == "compute"
    assert rec["time_us"] >= 0.0


@requires_cuda
def test_device_timer_elapsed_is_positive(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
    # A small CUDA kernel to ensure the GPU does real work.
    _ = torch.zeros(1024, device="cuda") + 1.0
    torch.cuda.synchronize(0)
    t.stop()

    records = flush_and_read(tmp_path)
    rec = next(r for r in records if r["type"] == "rec")
    assert rec["time_us"] > 0.0


@requires_cuda
def test_device_timer_deferred_until_flush(tmp_path):
    # Records from device timers are written only at flush(), not at stop().
    init_registry(tmp_path)
    path = tmp_path / "timing_rank0.jsonl"

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
    t.stop()

    # File should only have the meta line before flush.
    lines_before = [l for l in path.read_text().splitlines() if l.strip()]
    recs_before = [json.loads(l) for l in lines_before if json.loads(l)["type"] == "rec"]
    assert len(recs_before) == 0

    TimingRegistry.instance().flush()

    lines_after = [l for l in path.read_text().splitlines() if l.strip()]
    recs_after = [json.loads(l) for l in lines_after if json.loads(l)["type"] == "rec"]
    assert len(recs_after) == 1


@requires_cuda
def test_device_timer_reuse_produces_one_record_per_stop(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in range(3):
        Timer.set_step(step)
        t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
        _ = torch.zeros(1, device="cuda")
        t.stop()

    records = flush_and_read(tmp_path)
    recs = [r for r in records if r["type"] == "rec"]
    assert len(recs) == 3
    for i, rec in enumerate(recs):
        assert rec["step"] == i
        assert rec["where"] == "device"


@requires_cuda
def test_level_off_suppresses_device_records(tmp_path):
    init_registry(tmp_path)
    Timer.set_level(Level.Off)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
    t.stop()

    records = flush_and_read(tmp_path)
    assert all(r["type"] != "rec" for r in records)


@requires_cuda
def test_device_timer_stop_without_start_is_safe(tmp_path):
    # stop() before start() must not crash (running flag guards it).
    init_registry(tmp_path)
    t = Timer()
    t.stop()  # should be a no-op
    records = flush_and_read(tmp_path)
    assert all(r["type"] != "rec" for r in records)


@requires_cuda
def test_device_timer_destructor_without_stop_does_not_leak(tmp_path):
    # If stop() is never called the destructor must release the CUDA events.
    init_registry(tmp_path)
    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device, 0)
    del t  # destructor fires here — must not crash or leak events
