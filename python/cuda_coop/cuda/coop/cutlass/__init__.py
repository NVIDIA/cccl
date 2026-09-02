# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS CuTe DSL backend for block-wide cooperative primitives."""

from cuda.coop._core import ThreadGroup, this_block
from cuda.coop._core.root_api import _register_qualified_backend

from ._load_store import load, store
from ._runtime import is_current_cutlass_environment, validate_cutlass_runtime
from ._thread_data import ThreadData

validate_cutlass_runtime()

__all__ = ["ThreadData", "ThreadGroup", "this_block", "load", "store"]


_register_qualified_backend(__name__, is_active=is_current_cutlass_environment)
