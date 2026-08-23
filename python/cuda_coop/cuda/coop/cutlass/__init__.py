# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS-backed group-first cooperative primitives."""

import importlib as _importlib

from cuda.coop._core.api import (
    _register_qualified_backend as _register_qualified_backend,
)

from ._compiler import register_trace_context as _register_trace_context
from ._compiler._runtime import (
    validate_cutlass_runtime as _validate_cutlass_runtime,
)
from ._group_adjacent_difference import adjacent_difference
from ._group_discontinuity import discontinuity
from ._group_exchange import exchange
from ._group_histogram import histogram
from ._group_load_store import load, store
from ._group_merge_sort import merge_sort_keys, merge_sort_pairs
from ._group_radix import radix_rank, radix_sort_keys, radix_sort_pairs
from ._group_reduce import reduce, sum
from ._group_run_length_decode import run_length_decode
from ._group_scan import (
    exclusive_scan,
    exclusive_sum,
    inclusive_scan,
    inclusive_sum,
    scan,
)
from ._group_shuffle import shuffle
from ._group_topk import (
    topk_max_keys,
    topk_max_pairs,
    topk_min_keys,
    topk_min_pairs,
)
from ._thread_data import (
    ThreadData,
    ThreadDataLoadSource,
    ThreadDataSource,
    ThreadDataTensorMetadata,
)
from ._thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)

_validate_cutlass_runtime()
_register_trace_context()

__all__ = [
    "Hierarchy",
    "TempStorage",
    "ThreadData",
    "ThreadDataLoadSource",
    "ThreadDataSource",
    "ThreadDataTensorMetadata",
    "ThreadGroup",
    "ThreadHierarchy",
    "adjacent_difference",
    "aot",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "scan",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]


def __getattr__(name: str):
    if name == "TempStorage":
        module = _importlib.import_module(f"{__name__}._temp_storage")
        value = module.TempStorage
        globals()[name] = value
        return value
    if name == "aot":
        module = _importlib.import_module(f"{__name__}.aot")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


_register_qualified_backend(__name__)
