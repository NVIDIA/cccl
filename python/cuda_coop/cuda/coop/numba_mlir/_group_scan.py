# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Scan and Sum markers for Numba-CUDA-MLIR."""

from __future__ import annotations

from typing import Any

from ._compiler._operations import group_operation
from ._group_marker import group_primitive_marker
from ._thread_group import ThreadGroup

_FAMILY_MODULE = "cuda.coop.numba_mlir._compiler._group_scan"


@group_operation("scan", family_module=_FAMILY_MODULE)
def scan(
    group: ThreadGroup,
    value: Any,
    prefix_state: Any = None,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
    prefix_op: Any = None,
) -> Any:
    """Scan values across a block or warp group.

    Prefix callbacks are block-only. A ``StatefulFunction`` callback requires
    its one-item running state as the third positional argument.
    """

    return group_primitive_marker(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
        prefix_state=prefix_state,
        prefix_op=prefix_op,
    )


@group_operation("exclusive_scan", family_module=_FAMILY_MODULE)
def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    prefix_state: Any = None,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
    prefix_op: Any = None,
) -> Any:
    """Return an exclusive scan across a block or warp group."""

    return group_primitive_marker(
        "exclusive_scan",
        group,
        value,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
        prefix_state=prefix_state,
        prefix_op=prefix_op,
    )


@group_operation("inclusive_scan", family_module=_FAMILY_MODULE)
def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    prefix_state: Any = None,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
    prefix_op: Any = None,
) -> Any:
    """Return an inclusive scan across a block or warp group."""

    return group_primitive_marker(
        "inclusive_scan",
        group,
        value,
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
        prefix_state=prefix_state,
        prefix_op=prefix_op,
    )


@group_operation("exclusive_sum", family_module=_FAMILY_MODULE)
def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    prefix_state: Any = None,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
    prefix_op: Any = None,
) -> Any:
    """Return an exclusive prefix sum across a block or warp group."""

    return group_primitive_marker(
        "exclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
        prefix_state=prefix_state,
        prefix_op=prefix_op,
    )


@group_operation("inclusive_sum", family_module=_FAMILY_MODULE)
def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    prefix_state: Any = None,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
    valid_items: Any = None,
    aggregate_output: Any = None,
    prefix_op: Any = None,
) -> Any:
    """Return an inclusive prefix sum across a block or warp group."""

    return group_primitive_marker(
        "inclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
        prefix_state=prefix_state,
        prefix_op=prefix_op,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
