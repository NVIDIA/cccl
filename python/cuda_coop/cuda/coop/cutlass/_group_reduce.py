# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first reduction entrypoint."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
)
from cuda.coop._core.block import BlockReduceAlgorithm

from ._thread_data import _coerce_thread_payload
from ._thread_group import ThreadGroup

_SCOPE = __name__.rsplit(".", 1)[0]

_REDUCE_OPERATOR_CPP = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}

_BLOCK_REDUCE_ALGORITHM_ALIASES = {
    "raking_commutative_only": BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
    "raking": BlockReduceAlgorithm.RAKING,
    "warp_reductions": BlockReduceAlgorithm.WARP_REDUCTIONS,
    "warp_reductions_nondeterministic": (
        BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC
    ),
}


def _normalize_reduce_op(binary_op: Any) -> str:
    from ._compiler._types import normalize_reduce_op

    return normalize_reduce_op(binary_op, scope=_SCOPE)


def _is_boolean_control(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    try:
        from cutlass.base_dsl.typing import Boolean
    except ImportError:
        return False
    return isinstance(value, Boolean)


def _classify_valid_items(valid_items: Any) -> ArgumentBinding:
    if valid_items is None:
        return ArgumentBinding.omitted()
    if _is_boolean_control(valid_items):
        raise TypeError(f"{_SCOPE}.reduce valid_items must be an integer")
    if isinstance(valid_items, Integral):
        return ArgumentBinding.static(int(valid_items))
    from cutlass.base_dsl.typing import Integer

    if isinstance(valid_items, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_SCOPE}.reduce valid_items must be an integer")


def _normalize_block_reduce_algorithm(algorithm: Any) -> Any:
    if _is_boolean_control(algorithm):
        raise TypeError(f"{_SCOPE}.reduce algorithm must not be boolean")
    if isinstance(algorithm, str):
        return _BLOCK_REDUCE_ALGORITHM_ALIASES.get(algorithm, algorithm)
    return algorithm


def _validate_group_for_reduce(group: ThreadGroup) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")
    if group.kind == "grid":
        raise NotImplementedError(f"{_SCOPE}.reduce does not support grid groups")


def _reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a CUDA thread group with CUDAX or direct CUB.

    ``group`` always names the participating threads explicitly.
    Full-group reductions use CUDAX and may be broadcast to every member or
    returned only at rank zero. ``valid_items`` and explicit block ``algorithm``
    selectors use direct CUB and therefore require ``broadcast=False``.

    With ``broadcast=False``, only group rank zero has a defined result; other
    members receive an implementation placeholder and must not consume it.
    Every group member must still invoke the collective in converged control
    flow; root-only result ownership never means root-only participation.
    ``valid_items`` counts the first contributing group members, not the total
    items inside ``ThreadData``, and is therefore supported only for scalar
    operands. It changes which members contribute, not which members call the
    collective. Its valid range is ``1 <= valid_items <= group.static_size``.
    Static counts are checked while tracing. A runtime count is a uniform caller
    precondition and is passed directly to CUB without a device-side range
    check.

    Register-resident CuTe tensor fragments and ``TensorSSA`` values are
    adapted to ``ThreadData`` directly without another global/shared-memory
    load or bulk copy. Values that are already thread-local do not require an
    intervening ``coop.load`` call.

    Supported block algorithm names are ``raking_commutative_only``, ``raking``,
    and ``warp_reductions``. The nondeterministic CUB specialization is excluded
    because its current implementation is addition-specific. All direct CUB
    routes return a value defined only at the group root.
    """

    from ._compiler._launch import infer_launch_facts

    _validate_group_for_reduce(group)
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_SCOPE}.reduce broadcast must be a bool")
    op = _normalize_reduce_op(binary_op)
    valid_items_binding = _classify_valid_items(valid_items)
    algorithm = _normalize_block_reduce_algorithm(algorithm)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )
    launch = infer_launch_facts(
        {},
        scope=_SCOPE,
        primitive_name="reduce",
    )
    from ._lowering import _reduce as _provider

    return _provider.provider_reduce(
        group=group,
        launch=launch,
        value=value,
        op=op,
        broadcast=broadcast,
        valid_items=valid_items,
        valid_items_binding=valid_items_binding,
        algorithm=algorithm,
    )


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a CUDA thread group with CUDAX or direct CUB."""

    return _reduce(
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a CUDA thread group."""

    return _reduce(
        group,
        value,
        binary_op=None,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = ["reduce", "sum"]
