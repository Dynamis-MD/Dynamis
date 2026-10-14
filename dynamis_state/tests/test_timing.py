"""Tests for Timer and TimingRegistry (dynamis_state.timing)."""

import json
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

from dynamis_state import (
    Timer,
    TimingRegistry,
    TimerClass,
    Where,
    Level,
    TIMING_ENABLED,
    TrajectoryReader,
    TrajectoryWriter,
)

pytestmark = pytest.mark.skipif(
    not TIMING_ENABLED,
    reason="dynamis_state built with -DDISTNNP_TIMING=OFF",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

RUN_ID = b"\x01" * 16


@pytest.fixture(autouse=True)
def isolated_registry():
    """The registry is a process-wide singleton, so tests must not inherit each
    other's level, open fd or buffered records. Without this, the suite passes only
    because of collection order."""
    TimingRegistry.instance().close()
    Timer.set_level(Level.Off)
    yield
    TimingRegistry.instance().close()
    Timer.set_level(Level.Off)


def init_registry(tmp_path, rank=0, device_id=-1):
    TimingRegistry.instance().init(rank, rank, 1, RUN_ID, device_id, str(tmp_path))
    Timer.set_level(Level.Basic)


def read_records(tmp_path, rank=0):
    path = tmp_path / f"timing_rank{rank}.jsonl"
    lines = path.read_text().splitlines()
    # json.loads on every line is the splice check: a torn record fails to parse.
    return [json.loads(l) for l in lines if l.strip()]


def flush_and_read(tmp_path, rank=0):
    TimingRegistry.instance().flush()
    return read_records(tmp_path, rank)


def only_recs(records):
    return [r for r in records if r["type"] == "rec"]


# ── off by default ────────────────────────────────────────────────────────────


def test_level_is_off_before_anything_sets_it():
    """A fresh process records nothing until asked to.

    Runs in a subprocess: every other test here has already touched the
    process-wide level, so the default can't be observed any other way.
    """
    script = (
        "from dynamis_state import Timer, TimerClass\n"
        "assert Timer.enabled() is False, 'timing is on before set_level'\n"
        "t = Timer()\n"
        "t.start('iteration', TimerClass.Total)\n"
        "t.stop()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_level_off_produces_no_records(tmp_path):
    init_registry(tmp_path)
    Timer.set_level(Level.Off)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()

    assert only_recs(flush_and_read(tmp_path)) == []


def test_level_detail_behaves_as_basic(tmp_path):
    # Detail is reserved for the WAIT barriers, which are not implemented. Until
    # they are, it must at least not silently record less than Basic.
    init_registry(tmp_path)
    Timer.set_level(Level.Detail)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()

    assert len(only_recs(flush_and_read(tmp_path))) == 1


def test_level_off_costs_little_per_call():
    """Regression trip-wire, not a benchmark: the bound is loose, but a per-call
    cost near a syscall or stream lookup blows through it."""
    Timer.set_level(Level.Off)
    t = Timer()

    n = 20_000
    start = time.perf_counter()
    for _ in range(n):
        t.start("iteration", TimerClass.Total)
        t.stop()
    elapsed = time.perf_counter() - start

    per_call_us = elapsed / n * 1e6
    assert per_call_us < 50.0, f"Level.Off costs {per_call_us:.1f} us per start/stop"


# ── basic host timer ──────────────────────────────────────────────────────────


def test_host_timer_records_elapsed(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    time.sleep(0.01)
    t.stop()

    records = flush_and_read(tmp_path)
    assert records[0]["type"] == "meta"
    rec = records[1]
    assert rec["type"] == "rec"
    assert rec["key"] == "iteration"
    assert rec["class"] == "total"
    assert rec["where"] == "host"
    assert rec["time_ns"] >= 10_000_000  # at least 10 ms


def test_durations_are_integer_nanoseconds(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    time.sleep(0.001)
    t.stop()

    rec = only_recs(flush_and_read(tmp_path))[0]
    assert isinstance(rec["time_ns"], int)
    assert isinstance(rec["t_start_ns"], int)
    assert rec["time_ns"] > 0


def test_host_records_carry_a_monotonic_start_timestamp(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for _ in range(3):
        t.start("iteration", TimerClass.Total)
        time.sleep(0.001)
        t.stop()

    starts = [r["t_start_ns"] for r in only_recs(flush_and_read(tmp_path))]
    assert starts == sorted(starts)
    assert starts[0] != starts[-1]


# ── nesting ───────────────────────────────────────────────────────────────────


def test_host_nesting_is_derivable_from_timestamps(tmp_path):
    """The whole point of t_start_ns: post-processing must be able to tell that
    `halo_exchange` happened *inside* `model_forward` and subtract it."""
    init_registry(tmp_path)

    outer = Timer()
    inner = Timer()
    outer.start("model_forward", TimerClass.Compute)
    time.sleep(0.002)
    inner.start("halo_exchange", TimerClass.Comm)
    time.sleep(0.002)
    inner.stop()
    time.sleep(0.002)
    outer.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    o, i = recs["model_forward"], recs["halo_exchange"]
    assert o["t_start_ns"] <= i["t_start_ns"]
    assert i["t_start_ns"] + i["time_ns"] <= o["t_start_ns"] + o["time_ns"]
    assert i["time_ns"] < o["time_ns"]


@requires_cuda
def test_device_nesting_is_derivable_from_timestamps(tmp_path):
    init_registry(tmp_path, device_id=0)

    a = torch.randn(512, 512, device="cuda")
    outer = Timer()
    inner = Timer()
    outer.start("model_forward", TimerClass.Compute, Where.Device)
    for _ in range(4):
        a = a @ a.T
    inner.start("halo_exchange", TimerClass.Comm, Where.Device)
    for _ in range(4):
        a = a @ a.T
    inner.stop()
    for _ in range(4):
        a = a @ a.T
    outer.stop()
    torch.cuda.synchronize(0)

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    o, i = recs["model_forward"], recs["halo_exchange"]
    assert o["t_start_ns"] is not None and i["t_start_ns"] is not None
    assert o["t_start_ns"] <= i["t_start_ns"]
    assert i["t_start_ns"] + i["time_ns"] <= o["t_start_ns"] + o["time_ns"]


@requires_cuda
def test_host_and_device_records_share_a_timebase(tmp_path):
    """Where.Both emits one host and one device record for the same region. If the
    reference-event anchoring were wrong, the two t_start_ns values would be on
    unrelated clocks and could differ by the process's whole uptime."""
    init_registry(tmp_path, device_id=0)

    a = torch.randn(512, 512, device="cuda")
    t = Timer()
    t.start("iteration", TimerClass.Total, Where.Both)
    for _ in range(4):
        a = a @ a.T
    t.stop()
    torch.cuda.synchronize(0)

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == 2
    by_where = {r["where"]: r for r in recs}
    assert set(by_where) == {"host", "device"}
    assert by_where["host"]["key"] == by_where["device"]["key"] == "iteration"
    # The device span starts when the first launched work runs, which is at most a
    # few milliseconds after the host entered the region.
    delta_ns = abs(by_where["device"]["t_start_ns"] - by_where["host"]["t_start_ns"])
    assert delta_ns < 100_000_000, f"host/device timebases differ by {delta_ns} ns"


# ── span identity and nesting ─────────────────────────────────────────────────


def test_records_carry_span_identity(tmp_path):
    """Nesting is recorded, not inferred — t_start_ns is too imprecise and
    ambiguous to reconstruct containment from timestamps alone."""
    init_registry(tmp_path)

    outer, inner = Timer(), Timer()
    outer.start("model_forward", TimerClass.Compute)
    inner.start("halo_exchange", TimerClass.Comm)
    inner.stop()
    outer.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    o, i = recs["model_forward"], recs["halo_exchange"]
    assert o["id"] != i["id"]
    assert o["parent_id"] is None, "an outermost region is a root"
    assert i["parent_id"] == o["id"], "the inner region must name its parent"


def test_span_ids_are_unique_across_reuse(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in range(20):
        Timer.set_step(step)
        t.start("iteration", TimerClass.Total)
        t.stop()

    ids = [r["id"] for r in only_recs(flush_and_read(tmp_path))]
    assert len(ids) == 20
    assert len(set(ids)) == 20, "a reused Timer must not reuse its span id"


def test_siblings_are_not_nested(tmp_path):
    # `repartition` and `iteration` are separate regions on one Timer, not a nest.
    init_registry(tmp_path)

    t = Timer()
    t.start("repartition", TimerClass.Other)
    t.stop()
    t.start("iteration", TimerClass.Total)
    t.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert recs["repartition"]["parent_id"] is None
    assert recs["iteration"]["parent_id"] is None


def test_three_deep_nesting_is_exact(tmp_path):
    """The real shape: iteration > integrator_step > model_forward > halo_exchange.

    With the chain recorded, self time is a subtraction rather than a guess.
    """
    init_registry(tmp_path)

    keys = ["iteration", "integrator_step", "model_forward", "halo_exchange"]
    timers = [Timer() for _ in keys]
    for timer, key in zip(timers, keys):
        timer.start(key, TimerClass.Compute)
    for timer in reversed(timers):
        timer.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert recs["iteration"]["parent_id"] is None
    for child, parent in zip(keys[1:], keys):
        assert recs[child]["parent_id"] == recs[parent]["id"], child


def test_overlapping_regions_do_not_corrupt_the_stack(tmp_path):
    # The API permits a.start; b.start; a.stop; b.stop. Closing `a` out of order must
    # not strand `b`, and the region opened afterwards must still be a root.
    init_registry(tmp_path)

    a, b = Timer(), Timer()
    a.start("partition", TimerClass.Compute)
    b.start("build_graph", TimerClass.Compute)
    a.stop()
    b.stop()

    after = Timer()
    after.start("iteration", TimerClass.Total)
    after.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert recs["partition"]["parent_id"] is None
    assert recs["build_graph"]["parent_id"] == recs["partition"]["id"]
    assert recs["iteration"]["parent_id"] is None


def test_timer_destroyed_while_running_releases_its_span(tmp_path):
    # Otherwise its id stays on the thread's open-span stack and every later region
    # on that thread is recorded as a child of a region that has gone.
    init_registry(tmp_path)

    leaked = Timer()
    leaked.start("build_graph", TimerClass.Compute)
    del leaked

    after = Timer()
    after.start("iteration", TimerClass.Total)
    after.stop()

    rec = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}["iteration"]
    assert rec["parent_id"] is None


def test_nesting_is_per_thread(tmp_path):
    """Two parallel regions must not become each other's parent just because one
    started first — which is why the open-span stack is thread-local."""
    init_registry(tmp_path)

    outer = Timer()
    outer.start("model_forward", TimerClass.Compute)

    def worker():
        t = Timer()
        t.start("traj_write", TimerClass.Other)
        t.stop()

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    outer.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert recs["traj_write"]["parent_id"] is None


def test_explicit_parent_crosses_a_thread_boundary(tmp_path):
    """How the halo-exchange backward stays inside model_forward despite running
    on a different thread — see feature_push.py's forward()/backward()."""
    init_registry(tmp_path)

    outer = Timer()
    outer.start("model_forward", TimerClass.Compute)
    captured = Timer.current_span_id()
    assert captured != 0

    def worker():
        t = Timer()
        t.start("halo_exchange", TimerClass.Comm, parent_id=captured)
        t.stop()

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    outer.stop()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert recs["halo_exchange"]["parent_id"] == recs["model_forward"]["id"]


def test_current_span_id_is_zero_outside_any_region():
    # Also a hygiene check: the open-span stack is process state no fixture resets.
    assert (
        Timer.current_span_id() == 0
    ), "an earlier test leaked a running Timer onto this thread's open-span stack"


# ── reuse across steps ────────────────────────────────────────────────────────


def test_timer_reuse_produces_one_record_per_stop(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in range(5):
        Timer.set_step(step)
        t.start("iteration", TimerClass.Total)
        t.stop()

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == 5
    for i, rec in enumerate(recs):
        assert rec["step"] == i


@requires_cuda
def test_timer_reused_across_host_then_device(tmp_path):
    init_registry(tmp_path, device_id=0)

    t = Timer()
    t.start("region", TimerClass.Compute, Where.Host)
    time.sleep(0.001)
    t.stop()
    t.start("region", TimerClass.Compute, Where.Device)
    _ = torch.zeros(1024, device="cuda") + 1.0
    t.stop()

    recs = only_recs(flush_and_read(tmp_path))
    assert sorted(r["where"] for r in recs) == ["device", "host"]


def test_double_start_is_a_hard_error(tmp_path):
    # A misplaced stop() would otherwise produce a plausible-looking wrong number
    # and silently leak the first event pair.
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    with pytest.raises(RuntimeError):
        t.start("iteration", TimerClass.Total)


def test_stop_without_start_is_safe(tmp_path):
    init_registry(tmp_path)
    t = Timer()
    t.stop()
    assert only_recs(flush_and_read(tmp_path)) == []


# ── set_step stamps records ───────────────────────────────────────────────────


def test_set_step_stamps_records(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for step in [0, 10, 99]:
        Timer.set_step(step)
        t.start("iteration", TimerClass.Total)
        t.stop()

    steps = [r["step"] for r in only_recs(flush_and_read(tmp_path))]
    assert steps == [0, 10, 99]


def test_step_is_captured_at_start_not_at_stop(tmp_path):
    # A region that spans a step boundary belongs to the step it started in —
    # otherwise the autograd backward is attributed to whatever step the main
    # thread has since reached.
    init_registry(tmp_path)

    t = Timer()
    Timer.set_step(7)
    t.start("halo_exchange", TimerClass.Comm)
    Timer.set_step(8)
    t.stop()

    assert only_recs(flush_and_read(tmp_path))[0]["step"] == 7


# ── thread safety ─────────────────────────────────────────────────────────────


def test_concurrent_timers_produce_well_formed_lines(tmp_path):
    """The halo-exchange backward runs on a PyTorch autograd worker thread, so the
    registry is mutated from at least two threads. Every line must still parse, and
    no record may be lost."""
    init_registry(tmp_path)

    n_threads, per_thread = 8, 250
    errors = []

    def worker(tid):
        try:
            t = Timer()
            for _ in range(per_thread):
                t.start(f"key_{tid}", TimerClass.Comm)
                t.stop()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == n_threads * per_thread
    for tid in range(n_threads):
        assert sum(1 for r in recs if r["key"] == f"key_{tid}") == per_thread


def test_concurrent_timers_and_flushes(tmp_path):
    # Same as above, but with the main thread flushing underneath the workers
    init_registry(tmp_path)

    n_threads, per_thread = 4, 5_000
    errors = []
    done = threading.Event()

    def worker(tid):
        try:
            t = Timer()
            for _ in range(per_thread):
                t.start(f"key_{tid}", TimerClass.Comm)
                t.stop()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    while not done.is_set():
        TimingRegistry.instance().flush()
        time.sleep(0.002)
        if not any(th.is_alive() for th in threads):
            done.set()
    for th in threads:
        th.join()

    assert not errors, errors
    records = flush_and_read(tmp_path)  # parses every line; a splice raises here
    recs = only_recs(records)
    assert len(recs) == n_threads * per_thread


# ── call_order ────────────────────────────────────────────────────────────────


def test_call_order_reflects_first_seen_order(tmp_path):
    init_registry(tmp_path)

    t = Timer()
    for key in ["iteration", "build_graph", "iteration", "halo_exchange"]:
        t.start(key, TimerClass.Other)
        t.stop()

    co = flush_and_read(tmp_path)[-1]
    assert co["type"] == "call_order"
    assert co["order"] == ["iteration", "build_graph", "halo_exchange"]


def test_call_order_is_re_emitted_on_every_flush(tmp_path):
    """A killed job must still have a call_order line — so it cannot be a footer
    written once at close()."""
    init_registry(tmp_path)

    t = Timer()
    t.start("iteration", TimerClass.Other)
    t.stop()
    TimingRegistry.instance().flush()

    t.start("build_graph", TimerClass.Compute)
    t.stop()
    TimingRegistry.instance().flush()

    orders = [r["order"] for r in read_records(tmp_path) if r["type"] == "call_order"]
    assert len(orders) == 2
    assert orders[0] == ["iteration"]
    assert orders[1] == ["iteration", "build_graph"]


# ── metadata header ───────────────────────────────────────────────────────────


def test_header_fields(tmp_path):
    init_registry(tmp_path, rank=3)

    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()

    hdr = flush_and_read(tmp_path, rank=3)[0]
    assert hdr["type"] == "meta"
    assert hdr["rank"] == 3
    assert hdr["local_rank"] == 3
    assert hdr["world_size"] == 1
    assert hdr["device_id"] == -1
    assert hdr["run_id"] == "01" * 16
    assert hdr["time_unit"] == "ns"
    assert hdr["clock"] == "steady_clock"
    for field in ("date", "host", "pid", "version", "t0_ns"):
        assert field in hdr


# ── per-rank file naming ──────────────────────────────────────────────────────


def test_output_file_named_by_rank(tmp_path):
    for rank in [0, 2, 7]:
        TimingRegistry.instance().init(rank, rank, 8, RUN_ID, -1, str(tmp_path))
        Timer.set_level(Level.Basic)
        t = Timer()
        t.start("iteration", TimerClass.Total)
        t.stop()
        TimingRegistry.instance().flush()
        assert (tmp_path / f"timing_rank{rank}.jsonl").exists()


def test_init_creates_nested_output_dir(tmp_path):
    # mkdir must be recursive, and must tolerate a directory another rank on the
    # same node created a moment earlier.
    nested = tmp_path / "runs" / "job42" / "timing"
    TimingRegistry.instance().init(0, 0, 1, RUN_ID, -1, str(nested))
    Timer.set_level(Level.Basic)
    TimingRegistry.instance().flush()
    assert (nested / "timing_rank0.jsonl").exists()

    # Second rank hitting the same existing directory must not raise.
    TimingRegistry.instance().init(1, 1, 2, RUN_ID, -1, str(nested))
    Timer.set_level(Level.Basic)
    TimingRegistry.instance().flush()
    assert (nested / "timing_rank1.jsonl").exists()


# ── robustness against misuse ─────────────────────────────────────────────────


def test_timing_without_init_does_not_crash():
    # set_level(Basic) with no init() used to write to fd -1 and throw EBADF.
    Timer.set_level(Level.Basic)
    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()
    TimingRegistry.instance().flush()


def test_records_after_close_do_not_crash(tmp_path):
    init_registry(tmp_path)
    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()
    TimingRegistry.instance().close()

    before = read_records(tmp_path)
    t.start("iteration", TimerClass.Total)
    t.stop()
    TimingRegistry.instance().flush()
    assert read_records(tmp_path) == before


# ── failure handling ──────────────────────────────────────────────────────────


def test_close_survives_a_write_failure(tmp_path):
    """close() must not throw, because ~TimingRegistry() calls it and an escaping
    exception there is std::terminate. RLIMIT_FSIZE + SIGXFSZ ignored simulates a
    full-disk write failure. Runs in a subprocess since the rlimit is process-wide.
    """
    script = f"""
import resource, signal
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (2048, 2048))

from dynamis_state import Timer, TimerClass, TimingRegistry, Level
TimingRegistry.instance().init(0, 0, 1, {RUN_ID!r}, -1, {str(tmp_path)!r})
Timer.set_level(Level.Basic)

t = Timer()
for i in range(500):
    t.start("a_reasonably_long_region_key", TimerClass.Compute)
    t.stop()

try:
    TimingRegistry.instance().close()
except BaseException as exc:
    print("close raised:", type(exc).__name__, exc)
print("survived")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert (
        out.returncode == 0
    ), f"rc={out.returncode}\nstdout={out.stdout}\nstderr={out.stderr}"
    assert "survived" in out.stdout
    assert "close raised" not in out.stdout, out.stdout


def test_data_lands_when_the_process_exits_without_closing(tmp_path):
    """The write-once-at-exit contract, end to end: close() must land records even
    running from ~TimingRegistry() at static destruction, interpreter torn down."""
    script = f"""
from dynamis_state import Timer, TimerClass, TimingRegistry, Level
TimingRegistry.instance().init(0, 0, 1, {RUN_ID!r}, -1, {str(tmp_path)!r})
Timer.set_level(Level.Basic)
t = Timer()
for step in range(10):
    Timer.set_step(step)
    t.start("iteration", TimerClass.Total)
    t.stop()
# No close(), no flush() — exit and let the destructor land it.
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    recs = only_recs(read_records(tmp_path))
    assert len(recs) == 10
    assert sorted(r["step"] for r in recs) == list(range(10))


def test_close_is_idempotent(tmp_path):
    init_registry(tmp_path)
    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()

    TimingRegistry.instance().close()
    before = read_records(tmp_path)
    TimingRegistry.instance().close()
    TimingRegistry.instance().close()
    assert read_records(tmp_path) == before


def test_concurrent_flush_and_close_do_not_lose_records(tmp_path):
    """A smoke test, not a real race check — the GIL serializes these calls from
    Python anyway; TSan on a C++ driver is what actually verifies io_mu."""
    init_registry(tmp_path)

    t = Timer()
    for step in range(200):
        Timer.set_step(step)
        t.start("iteration", TimerClass.Total)
        t.stop()

    errors = []

    def flusher():
        try:
            for _ in range(20):
                TimingRegistry.instance().flush()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=flusher) for _ in range(4)]
    for th in threads:
        th.start()
    TimingRegistry.instance().close()
    for th in threads:
        th.join()

    assert not errors, errors
    assert len(only_recs(read_records(tmp_path))) == 200


# ── write-once-at-exit ────────────────────────────────────────────────────────


def test_nothing_is_written_before_close(tmp_path):
    """The run buffers everything and writes once. Without an explicit flush, a
    record must not reach the file just because it was recorded."""
    init_registry(tmp_path)
    t = Timer()
    for step in range(50):
        Timer.set_step(step)
        t.start("iteration", TimerClass.Total)
        t.stop()

    # The header line is written by init(); the records are not.
    assert only_recs(read_records(tmp_path)) == []

    TimingRegistry.instance().close()
    assert len(only_recs(read_records(tmp_path))) == 50


def test_close_writes_records_without_any_flush(tmp_path):
    init_registry(tmp_path)
    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()
    t.start("repartition", TimerClass.Other)
    t.stop()

    TimingRegistry.instance().close()

    recs = only_recs(read_records(tmp_path))
    assert [r["key"] for r in recs] == ["iteration", "repartition"]
    # close() still leaves an authoritative call_order line behind.
    co = read_records(tmp_path)[-1]
    assert co["type"] == "call_order"
    assert co["order"] == ["iteration", "repartition"]


@requires_cuda
def test_device_records_survive_reaping(tmp_path):
    """Reaping recycles CUDA events well before close(); every record must still
    land exactly once."""
    init_registry(tmp_path, device_id=0)

    a = torch.randn(64, 64, device="cuda")
    t = Timer()
    n = 300  # comfortably past the reap threshold
    for step in range(n):
        Timer.set_step(step)
        t.start("build_graph", TimerClass.Compute, Where.Device)
        a = a @ a.T
        t.stop()
        # Let the stop events complete so the next append actually reaps them.
        if step % 50 == 0:
            torch.cuda.synchronize(0)

    torch.cuda.synchronize(0)
    TimingRegistry.instance().close()

    recs = only_recs(read_records(tmp_path))
    assert len(recs) == n
    assert sorted(r["step"] for r in recs) == list(range(n))
    assert {r["where"] for r in recs} == {"device"}
    # Reaping must not fabricate or drop durations.
    assert all(r.get("time_ns") is not None and r["time_ns"] >= 0 for r in recs)


@requires_cuda
def test_reaping_bounds_live_cuda_events(tmp_path):
    """Reaping must bound live events, not accumulate one pair per measurement
    over the whole run."""
    init_registry(tmp_path, device_id=0)

    a = torch.randn(32, 32, device="cuda")
    t = Timer()
    n = 5000
    for step in range(n):
        Timer.set_step(step)
        t.start("build_graph", TimerClass.Compute, Where.Device)
        a = a @ a.T
        t.stop()

    torch.cuda.synchronize(0)
    TimingRegistry.instance().close()

    recs = only_recs(read_records(tmp_path))
    assert len(recs) == n
    assert sorted(r["step"] for r in recs) == list(range(n))


def test_reinit_closes_previous_file(tmp_path):
    init_registry(tmp_path, rank=0)
    t = Timer()
    t.start("iteration", TimerClass.Total)
    t.stop()

    # Re-init must drain the live registry rather than dropping its records.
    init_registry(tmp_path, rank=1)
    assert only_recs(read_records(tmp_path, rank=0))


@pytest.mark.parametrize(
    "key",
    [
        'quote"inside',
        "back\\slash",
        "new\nline",
        "tab\there",
        "ctrl\x01char",
    ],
)
def test_keys_are_json_escaped(tmp_path, key):
    # Timer.start takes an arbitrary std::string from Python. An unescaped quote or
    # newline breaks the parse of the entire file, not just one record.
    init_registry(tmp_path)

    t = Timer()
    t.start(key, TimerClass.Other)
    t.stop()

    records = flush_and_read(tmp_path)
    rec = only_recs(records)[0]
    assert rec["key"] == key
    assert records[-1]["order"] == [key]


# ── TimerClass values round-trip through the file ────────────────────────────


@pytest.mark.parametrize(
    "cls,expected",
    [
        (TimerClass.Compute, "compute"),
        (TimerClass.Comm, "comm"),
        (TimerClass.Wait, "wait"),
        (TimerClass.Other, "other"),
        (TimerClass.Total, "total"),
    ],
)
def test_timer_class_written_correctly(tmp_path, cls, expected):
    init_registry(tmp_path)

    t = Timer()
    t.start("region", cls)
    t.stop()

    assert only_recs(flush_and_read(tmp_path))[0]["class"] == expected


# ── trajectory writer ─────────────────────────────────────────────────────────


def test_trajectory_writer_emits_records(tmp_path):
    """The writer is instrumented in C++ (TrajectoryWriter::submit / ::flush), so
    any caller is measured — not only Simulator.run()."""
    init_registry(tmp_path)

    xyz = os.path.join(os.path.dirname(__file__), "xyz.xyz")
    state, cell, pbc = TrajectoryReader.read(xyz)
    with TrajectoryWriter(
        0,
        RUN_ID,
        "author",
        str(tmp_path),
        "trajectory.dynamis",
        state,
        cell,
        pbc,
        1.0,
        1,
    ) as writer:
        writer.update_atom_counts(torch.tensor([6], dtype=torch.uint32))
        writer.submit(0, state)
        writer.flush()

    recs = {r["key"]: r for r in only_recs(flush_and_read(tmp_path))}
    assert "traj_write" in recs
    assert "traj_flush" in recs
    for rec in recs.values():
        # Writing is host work: a device timer here would measure nothing, since
        # submit() launches no asynchronous CUDA work.
        assert rec["where"] == "host"
        assert rec["class"] == "other"
        assert rec["time_ns"] > 0


# ── device timers ─────────────────────────────────────────────────────────────


@requires_cuda
def test_device_timer_produces_device_record(tmp_path):
    init_registry(tmp_path, device_id=0)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device)
    # A real kernel, not a synchronize() — the region must have something to time.
    a = torch.randn(512, 512, device="cuda")
    for _ in range(4):
        a = a @ a.T
    t.stop()
    torch.cuda.synchronize(0)

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == 1
    rec = recs[0]
    assert rec["key"] == "gpu_region"
    assert rec["where"] == "device"
    assert rec["class"] == "compute"
    assert rec["time_ns"] > 0
    assert rec.get("error") is None


@requires_cuda
def test_device_time_agrees_with_torch_cuda_event(tmp_path):
    """Accuracy check against an independent mechanism. Our event pair sits strictly
    inside the reference pair, so ours must be slightly smaller — never larger, and
    never off by an order of magnitude."""
    init_registry(tmp_path, device_id=0)

    a = torch.randn(1024, 1024, device="cuda")
    for _ in range(3):  # warm up autotuning / allocator
        a @ a
    torch.cuda.synchronize(0)

    ref_start = torch.cuda.Event(enable_timing=True)
    ref_stop = torch.cuda.Event(enable_timing=True)

    t = Timer()
    ref_start.record()
    t.start("matmul", TimerClass.Compute, Where.Device)
    for _ in range(20):
        a @ a
    t.stop()
    ref_stop.record()
    torch.cuda.synchronize(0)

    ref_us = ref_start.elapsed_time(ref_stop) * 1000.0
    ours_us = only_recs(flush_and_read(tmp_path))[0]["time_ns"] / 1000.0

    assert ref_us > 100.0, "workload too small to compare meaningfully"
    assert ours_us <= ref_us * 1.05
    assert abs(ours_us - ref_us) / ref_us < 0.25


@requires_cuda
def test_device_timer_deferred_until_flush(tmp_path):
    # cudaEventElapsedTime blocks, so device durations must be read at the flush
    # boundary and never at stop().
    init_registry(tmp_path, device_id=0)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device)
    t.stop()

    assert only_recs(read_records(tmp_path)) == []
    TimingRegistry.instance().flush()
    assert len(only_recs(read_records(tmp_path))) == 1


@requires_cuda
def test_device_timer_reuse_produces_one_record_per_stop(tmp_path):
    init_registry(tmp_path, device_id=0)

    t = Timer()
    for step in range(3):
        Timer.set_step(step)
        t.start("gpu_region", TimerClass.Compute, Where.Device)
        _ = torch.zeros(1, device="cuda")
        t.stop()

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == 3
    for i, rec in enumerate(recs):
        assert rec["step"] == i
        assert rec["where"] == "device"


@requires_cuda
def test_repeated_flushes_do_not_exhaust_cuda_events(tmp_path):
    """Reap (append_device) and explicit flush both recycle events; checks they
    don't interfere with each other."""
    init_registry(tmp_path, device_id=0)

    t = Timer()
    total = 0
    for step in range(50):
        Timer.set_step(step)
        for _ in range(20):
            t.start("gpu_region", TimerClass.Compute, Where.Device)
            _ = torch.zeros(1, device="cuda")
            t.stop()
            total += 1
        TimingRegistry.instance().flush()

    recs = only_recs(read_records(tmp_path))
    assert len(recs) == total
    assert all(r.get("error") is None for r in recs)


@requires_cuda
def test_where_both_records_share_one_span_id(tmp_path):
    # One region measured on two clocks: a child looks up the nearest ancestor that
    # has a record on its own `where`, so the two must not be separate spans.
    init_registry(tmp_path, device_id=0)

    a = torch.randn(256, 256, device="cuda")
    t = Timer()
    t.start("model_forward", TimerClass.Compute, Where.Both)
    a = a @ a.T
    t.stop()
    torch.cuda.synchronize(0)

    recs = only_recs(flush_and_read(tmp_path))
    assert len(recs) == 2
    by_where = {r["where"]: r for r in recs}
    assert by_where["host"]["id"] == by_where["device"]["id"]


@requires_cuda
def test_device_nesting_survives_repeated_reanchoring(tmp_path):
    """Every flush() retires the timebase anchor and builds a new one. Records taken
    across those boundaries must stay on one timeline and keep their parent links."""
    init_registry(tmp_path, device_id=0)

    a = torch.randn(128, 128, device="cuda")
    for step in range(10):
        Timer.set_step(step)
        outer, inner = Timer(), Timer()
        outer.start("model_forward", TimerClass.Compute, Where.Device)
        a = a @ a.T
        inner.start("halo_exchange", TimerClass.Comm, Where.Device)
        a = a @ a.T
        inner.stop()
        outer.stop()
        TimingRegistry.instance().flush()

    torch.cuda.synchronize(0)
    TimingRegistry.instance().close()

    recs = only_recs(read_records(tmp_path))
    assert len(recs) == 20
    by_id = {r["id"]: r for r in recs}
    children = [r for r in recs if r["key"] == "halo_exchange"]
    assert len(children) == 10
    for child in children:
        parent = by_id[child["parent_id"]]
        assert parent["key"] == "model_forward"
        assert parent["step"] == child["step"]
        assert child["time_ns"] is not None and child["time_ns"] >= 0


@requires_cuda
def test_level_off_suppresses_device_records(tmp_path):
    init_registry(tmp_path, device_id=0)
    Timer.set_level(Level.Off)

    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device)
    t.stop()

    assert only_recs(flush_and_read(tmp_path)) == []


@requires_cuda
def test_device_timer_destructor_without_stop_does_not_leak(tmp_path):
    # If stop() is never called the destructor must return the events to the pool.
    init_registry(tmp_path, device_id=0)
    t = Timer()
    t.start("gpu_region", TimerClass.Compute, Where.Device)
    del t

    TimingRegistry.instance().flush()
    assert only_recs(read_records(tmp_path)) == []
