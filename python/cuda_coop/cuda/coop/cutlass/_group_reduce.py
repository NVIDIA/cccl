# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first built-in Reduce and Sum entry points."""

from enum import Enum
from numbers import Integral

from cuda.coop._core import ArgumentBinding
from cuda.coop._core.thread_group import ThreadGroup

from ._compiler._launch import current_kernel_launch_facts
from ._group_load_store import _is_boolean
from ._thread_data import _coerce_thread_payload
from ._thread_group import _require_complete_warp_partition

_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset({"raking_commutative_only", "raking", "warp_reductions"})


def _classify_valid_items(value):
    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.reduce valid_items must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_SCOPE}.reduce valid_items must be an integer")


def _normalize_algorithm(algorithm):
    if algorithm is None:
        return None
    if not isinstance(algorithm, str) or isinstance(algorithm, Enum):
        raise TypeError(f"{_SCOPE}.reduce algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token not in _ALGORITHMS:
        raise ValueError(
            f"{_SCOPE}.reduce algorithm must be one of: "
            + ", ".join(sorted(_ALGORITHMS))
        )
    return token


def reduce(
    group, value, /, *, binary_op=None, broadcast=True, valid_items=None, algorithm=None
):
    """Reduce a scalar or per-thread payload to one scalar.

    Full built-in reductions use the hierarchy-aware CUDAX implementation.
    Prefix counts and explicit block algorithms select direct CUB and require
    ``broadcast=False``. Every group member must call; only group rank zero may
    consume a nonbroadcast result. ``valid_items`` counts contributing members,
    requires scalar input, and must be uniform and between one and group size.
    Mapped groups of warps require every parent block thread to reach the call,
    including threads excluded by a non-exhaustive mapping.
    Inputs are preserved. Qualified register tensors and TensorSSA are adapted
    to per-thread payloads. Custom callbacks are unsupported.
    """

    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_SCOPE}.reduce broadcast must be a bool")
    op = normalize_operator(binary_op)
    algorithm = _normalize_algorithm(algorithm)
    valid_binding = _classify_valid_items(valid_items)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="reduce", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._reduce import provider_reduce

    return provider_reduce(
        group=group,
        launch=launch,
        value=value,
        op=op,
        broadcast=broadcast,
        valid_items=valid_items,
        valid_items_binding=valid_binding,
        algorithm=algorithm,
    )


def sum(group, value, /, *, broadcast=True, valid_items=None, algorithm=None):
    """Sum all per-thread contributions using the Reduce participation contract."""
    return reduce(
        group, value, broadcast=broadcast, valid_items=valid_items, algorithm=algorithm
    )


__all__ = ["reduce", "sum"]
