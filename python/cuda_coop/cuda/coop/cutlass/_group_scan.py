# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first scan entrypoint."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
    ScanMode,
)
from cuda.coop._core.block import BlockScanAlgorithm

from ._thread_data import _coerce_thread_payload
from ._thread_group import (
    ThreadGroup,
    _require_complete_warp_partition,
    _resolve_collective_group_from_launch,
)

_SCOPE = __name__.rsplit(".", 1)[0]

_SCAN_OPERATOR_CPP = {
    "sum": "::cuda::std::plus<T>",
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}

_BLOCK_SCAN_ALGORITHM_ALIASES = {
    "raking": BlockScanAlgorithm.RAKING,
    "raking_memoize": BlockScanAlgorithm.RAKING_MEMOIZE,
    "warp_scans": BlockScanAlgorithm.WARP_SCANS,
}


def _normalize_scan_op(scan_op: Any) -> str:
    from ._compiler._types import normalize_scan_op

    return normalize_scan_op(scan_op, scope=_SCOPE)


def _normalize_scan_mode(mode: Any) -> str:
    from ._group_reduce import _is_boolean_control

    if _is_boolean_control(mode):
        raise TypeError(f"{_SCOPE}.scan mode must not be boolean")
    try:
        return ScanMode(mode).value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_SCOPE}.scan mode must be 'exclusive' or 'inclusive'"
        ) from exc


def _normalize_block_scan_algorithm(algorithm: Any) -> Any:
    from ._group_reduce import _is_boolean_control

    if _is_boolean_control(algorithm):
        raise TypeError(f"{_SCOPE}.scan algorithm must not be boolean")
    if isinstance(algorithm, str):
        return _BLOCK_SCAN_ALGORITHM_ALIASES.get(algorithm, algorithm)
    return algorithm


def _validate_group_for_scan(group: ThreadGroup) -> None:
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.scan currently lowers CUB scans only for this_block "
            "and physical or logical warp groups"
        )


def _classify_valid_items(valid_items: Any) -> ArgumentBinding:
    if valid_items is None:
        return ArgumentBinding.omitted()
    from ._group_reduce import _is_boolean_control

    if _is_boolean_control(valid_items):
        raise TypeError(f"{_SCOPE}.scan valid_items must be an integer")
    if isinstance(valid_items, Integral):
        return ArgumentBinding.static(int(valid_items))
    from cutlass.base_dsl.typing import Integer

    if isinstance(valid_items, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_SCOPE}.scan valid_items must be an integer")


def _scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    aggregate_output: Any = None,
    valid_items: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Internal implementation for the public group-first scan entrypoints."""

    from ._compiler._launch import infer_launch_facts

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")

    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )

    mode = _normalize_scan_mode(mode)
    op = _normalize_scan_op(scan_op)
    algorithm = _normalize_block_scan_algorithm(algorithm)
    _classify_valid_items(valid_items)
    if mode == ScanMode.INCLUSIVE.value and initial_value is not None:
        raise ValueError(
            f"{_SCOPE}.scan initial_value is not supported for inclusive scans"
        )
    if mode == ScanMode.EXCLUSIVE.value and op != "sum" and initial_value is None:
        raise ValueError(
            f"{_SCOPE}.scan requires initial_value for non-default exclusive scans"
        )

    _validate_group_for_scan(group)
    launch = infer_launch_facts(
        {},
        scope=_SCOPE,
        primitive_name="scan",
    )
    validated_group = _resolve_collective_group_from_launch(
        group,
        launch,
        feature="scan",
    )
    assert validated_group.hierarchy is not None
    _require_complete_warp_partition(
        validated_group,
        feature="scan",
        exact_block_dim=validated_group.hierarchy.block_dim,
    )

    from ._lowering import _scan as _provider

    return _provider.provider_scan(
        group=group,
        launch=launch,
        value=value,
        mode=mode,
        op=op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Scan values across a complete CUDA block, physical warp, or logical warp.

    ``group`` always names the participating threads explicitly.
    Block scans accept scalars, ``ThreadData``, rmem tensors, and ``TensorSSA``.
    Physical- and logical-warp scans are scalar-per-lane. The default operator is sum;
    non-default exclusive scans require ``initial_value``. Inclusive scans do
    not accept an initial value. Explicit ``algorithm`` selectors apply only
    to block scans and accept ``raking``, ``raking_memoize``, or ``warp_scans``.
    ``warp_scans`` requires a block size that is a multiple of 32 threads.
    A runtime ``initial_value`` must be uniform across every group member.
    ``valid_items`` selects a leading partial logical warp and is accepted only
    for physical or logical warp groups. ``aggregate_output`` receives the
    reduction of the input values and excludes ``initial_value``.

    Every group member must invoke the collective in converged control flow.
    ``coop.TempStorage()`` requests inferred caller-owned scratch for block
    scans. Fixed-capacity storage uses the supplied allocation and validates
    its capacity and alignment in the generated CUB shim. Shared storage
    inserts the reuse barrier by default; ``auto_sync=False`` leaves
    synchronization to the caller. Warp scans use implementation-owned scratch.
    """

    return _scan(
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive sum across a block, physical warp, or logical warp."""

    return _scan(
        group,
        value,
        mode="exclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive sum across a block, physical warp, or logical warp."""

    return _scan(
        group,
        value,
        mode="inclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an exclusive scan using a supported built-in operator."""

    return _scan(
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
) -> Any:
    """Return an inclusive scan using a supported built-in operator."""

    return _scan(
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=None,
        algorithm=algorithm,
        aggregate_output=aggregate_output,
        valid_items=valid_items,
        temp_storage=temp_storage,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
