# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile cooperative kernels through Numba typing instead of the planner.

Setting ``CUDA_COOP_NUMBA_MLIR_PIPELINE=typed`` before the backend activates
selects this pipeline in place of ``CoopWholeFunctionPlanner``. That planner
is not registered. The pipeline has four stages:

1. native inference constraints and typing templates on cooperative Numba
   types (``_inference``, ``_types``, ``_templates``),
   which give each cooperative call a request that knows nothing of the
   launch;
2. one after-inference call resolution per kernel (``_resolve``), which
   turns every request into a concrete provider call and whose decisions
   are plain data (``_plan``);
3. lowering of each resolved call to a provider call (``_lower``);
4. link items backed by the resolution's one compile (``_link``).

Every Numba-CUDA-MLIR interface that
is not public is reached through ``_bridges``, and every change to
Numba-CUDA-MLIR's own behaviour is a pinned patch in ``_ncm_patches``,
applied only by this pipeline. Operation families register their supported
typing and lowering together; unsupported cooperative calls fail explicitly.
"""

from __future__ import annotations

_activated = False

__all__ = ["_activate"]


def _activate() -> None:
    """Register the pipeline's typing, rewrite and lowering hooks once.

    The backend's activation calls this under its activation lock, in place
    of registering the planner.
    """

    global _activated
    if _activated:
        return
    from . import (
        _bridges,
        _exchange,
        _groups,
        _lower,
        _merge_sort,
        _ncm_patches,
        _neighbors,
        _radix,
        _reductions,
        _resolve,
        _scan,
        _temp_storage,
        _templates,
        _topk,
        _types,
    )

    _ncm_patches.apply()
    _types.register()
    _groups.register()
    _temp_storage.register()
    _templates.register()
    _exchange.register()
    _reductions.register()
    _scan.register()
    _merge_sort.register()
    _radix.register()
    _topk.register()
    _neighbors.register()
    _lower.register()
    _resolve.register()
    _bridges.refresh_registries()
    _activated = True
