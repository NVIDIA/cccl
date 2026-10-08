# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative primitives for the CUTLASS CuTe DSL compiler.

Importing this namespace validates the optional runtime and registers its
active compiler environment with the common API. Calls inside that environment
use these implementations. The namespace provides block DIRECT Load and Store
plus per-thread register payloads.
"""

from .._core.api import ThreadDataLike
from ._compiler._activation import register_trace_context
from ._group_load_store import load, store
from ._thread_data import ThreadData
from ._thread_group import Hierarchy, ThreadGroup, ThreadHierarchy, this_block

__all__ = [
    "Hierarchy",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "load",
    "store",
    "this_block",
]


def __dir__():
    """List the qualified public API for interactive completion."""

    return sorted(__all__)


register_trace_context()
del register_trace_context
