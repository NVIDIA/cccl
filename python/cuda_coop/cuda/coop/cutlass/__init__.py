# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS CuTe DSL backend for block-wide cooperative primitives."""

from cuda.coop._core import ThreadGroup, this_block
from cuda.coop._core.root_api import _register_qualified_backend

from ._compiler import register_trace_context
from ._load_store import load, store
from ._runtime import validate_cutlass_runtime
from ._thread_data import ThreadData

validate_cutlass_runtime()
register_trace_context()

__all__ = ["ThreadData", "ThreadGroup", "this_block", "load", "store"]


_register_qualified_backend(__name__)
