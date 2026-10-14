"""Process-wide drift-watchdog flags carried by the IPC push-sync handshake.

Every IPC push already ends in an all-rank handshake: each rank writes a generation
counter into its own sync word and waits until every other rank's counter reaches the
same generation (ipc_exchanger.cpp, record_push_complete / wait_for_pushes). When flags
are enabled for a device, every GPUDirectExchanger on that device carries them on that
handshake at no extra communication cost:

  * before its counter write, each rank copies ``local`` into its sync word;
  * after the wait, each rank ORs (bitwise) every rank's copy into ``global_``.

The flag is a small bit field (the Simulator uses bit 0 = rebuild the halo before the
next step, bit 1 = this step's graph was built past the displacement guarantee).

All ranks run the same sequence of pushes, so after any given push every rank holds
the same ``global_``. That agreement is what lets every rank decide identically, and
without a collective, whether to repartition.

The registry is empty by default, and exchangers only carry flags that are registered
here, so nothing changes unless a caller (the Simulator's drift watchdog) enables it.
Models do not need to know about it: their exchangers pick the flags up on their next
push.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class DriftFlags:
    local: (
        torch.Tensor
    )  # int32 [1]: this rank's flag word, written by the caller on device
    global_: (
        torch.Tensor
    )  # int32 [1]: bitwise OR over all ranks, rewritten after every push
    n_wired: int = (
        0  # exchangers that have picked these flags up (0 => none carry them)
    )


_REGISTRY: Dict[torch.device, DriftFlags] = {}


def _key(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def enable(device) -> DriftFlags:
    """Install a fresh, zeroed set of flags for ``device`` and return it. Exchangers on
    that device switch to it at their next push, so ``n_wired`` counts only the
    exchangers that carry *these* flags; state left by an earlier caller (a previous
    Simulator in the same process) cannot leak in."""
    key = _key(device)
    _REGISTRY[key] = DriftFlags(
        local=torch.zeros(1, dtype=torch.int32, device=key),
        global_=torch.zeros(1, dtype=torch.int32, device=key),
    )
    return _REGISTRY[key]


def get(device) -> Optional[DriftFlags]:
    """The flags for ``device``, or None if the watchdog is not enabled there."""
    return _REGISTRY.get(_key(device))


def disable(device=None) -> None:
    """Stop carrying flags on ``device`` (all devices if None)."""
    if device is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(_key(device), None)
