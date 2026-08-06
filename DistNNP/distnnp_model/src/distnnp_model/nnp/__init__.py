"""Backend models, imported lazily.

Importing this package must NOT pull in every backend: the extras are optional
(``distnnp_model[uma]`` / ``[orb]``), so an eager ``from .uma import UMAModel``
here would force every ORB user to install fairchem-core and vice versa.

PEP 562 module ``__getattr__`` keeps both call styles working while importing
only the backend actually requested:

    from distnnp_model.nnp import ORBModel        # imports orb only
    from distnnp_model.nnp.orb import ORBModel    # imports orb only
"""

from typing import TYPE_CHECKING

_LAZY = {
    "UMAModel": "distnnp_model.nnp.uma",
    "ORBModel": "distnnp_model.nnp.orb",
}

if TYPE_CHECKING:  # static analysers only; never executed at runtime
    from distnnp_model.nnp.uma import UMAModel
    from distnnp_model.nnp.orb import ORBModel


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "UMAModel",
    "ORBModel"
]
