# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Interop adapters between ``cuda.stf._experimental`` and external runtimes.

Each submodule is opt-in: importing :mod:`cuda.stf._experimental` itself does
not pull in Numba, PyTorch, or any other optional dependency. Users explicitly
import the adapter they need, for example::

    from cuda.stf._experimental.interop.numba import numba_task
    from cuda.stf._experimental.interop.pytorch import pytorch_task

The optional runtime is imported lazily inside the adapter functions; a missing
dependency raises a clear ``ImportError`` at first call.
"""


_SUBMODULES = frozenset({"numba", "pytorch"})


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
