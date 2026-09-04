# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose Numba-qualified Scan markers and their user-facing contracts.

These functions are recognized inside kernels. Their markers let the group
planner choose a CUB provider from the launch, payload, and selectors. They
extend the common API with local arrays, device operators, lane-prefix
counts, and aggregate outputs; they do not execute a host-side scan.
"""

from __future__ import annotations

from typing import Any

from .._compiler._operations import group_operation
from .._thread_group import ThreadGroup
from ._marker import group_primitive_marker

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
    block_prefix_callback_op: Any = None,
) -> Any:
    """Scan values across a block or warp group.

    Extends :func:`cuda.coop.scan` with the options below. Group requirements,
    modes, algorithms, and temporary storage follow the common function.

    Parameters
    ----------
    value : numeric scalar, ThreadData, or local array
        Blocks also accept a fixed-size one-dimensional local array, with
        the same dtype and extent across threads. Warps accept scalars only.
    scan_op : str or device callable, optional
        Also accepts ``operator``/NumPy aliases for the built-in strings or a
        stateless device function ``op(left, right)``. The operator is fixed
        at compile time and must be associative and return the input dtype.
        ``None`` selects sum. Non-sum exclusive scans require
        ``initial_value`` unless ``prefix_op`` supplies the seed. Stateful
        binary operators are unsupported.
    valid_items : int or integer scalar, optional
        Warp-only count of contributing lanes, from one through the group
        size, uniform within the group. ``None`` includes all lanes. Every
        lane participates, but only ranks below this count have defined scan
        results. The enclosing block must still contain complete physical
        warps. Blocks require ``None``.
    aggregate_output : ThreadData or local array, optional
        Writable one-item payload with the input dtype. Receives the input
        aggregate on every member, excluding ``initial_value`` and lanes
        beyond ``valid_items``. Requires ``None`` with ``prefix_op``.
    prefix_state : ThreadData or local array, optional
        One-item state passed as the third positional argument for a
        ``StatefulFunction`` prefix callback. Its dtype must match the
        callback descriptor.
    prefix_op : device callable or StatefulFunction, optional
        Block-only callback that receives the tile aggregate and supplies
        its prefix. A stateful callback also receives ``prefix_state`` first.
        Requires ``initial_value=None`` and ``aggregate_output=None``.

    Returns
    -------
    numeric scalar or per-thread payload
        This thread's prefixes, with the input dtype and item count. A
        payload input produces a fresh payload and remains unchanged. With
        ``valid_items``, read only valid lanes. ``aggregate_output`` is
        written separately.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ Scan, Sum, and aggregate overloads.

    Examples
    --------
    See :func:`~cuda.coop.numba_mlir.inclusive_scan` for a device operator,
    :func:`~cuda.coop.numba_mlir.exclusive_scan` for a partial warp and
    aggregate output.
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
        block_prefix_callback_op=block_prefix_callback_op,
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
    block_prefix_callback_op: Any = None,
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
        block_prefix_callback_op=block_prefix_callback_op,
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
    block_prefix_callback_op: Any = None,
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
        block_prefix_callback_op=block_prefix_callback_op,
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
    block_prefix_callback_op: Any = None,
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
        block_prefix_callback_op=block_prefix_callback_op,
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
    block_prefix_callback_op: Any = None,
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
        block_prefix_callback_op=block_prefix_callback_op,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
