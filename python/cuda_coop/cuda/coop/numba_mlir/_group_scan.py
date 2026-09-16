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
    """Scan a group with Numba-CUDA-MLIR operators and prefix callbacks.

    This qualified API extends :func:`cuda.coop.scan` with local-array
    inputs, device operators, partial Warp scans, aggregate outputs, and
    Block prefix callbacks. See :ref:`coop-scans` for prefix semantics.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp; see :ref:`coop-thread-groups`.
        Warp scans require an enclosing block size divisible by 32.
        Every member must participate, including lanes beyond ``valid_items``;
        see :ref:`coop-participation`.
    value : numeric scalar, ThreadData, or local array
        Each thread's input. Blocks accept a scalar or a fixed-size
        one-dimensional local payload. Warps accept one scalar per lane.
        The numeric dtype and item count must match across threads.
        Payloads use :ref:`blocked order <coop-data-layouts>` and remain
        unchanged; see :ref:`coop-thread-data`.
    prefix_state : ThreadData or local array, optional
        Writable one-item state, passed as the third positional argument
        when ``prefix_op`` is a :class:`~cuda.coop.numba_mlir.StatefulFunction`.
        Initialize every thread's copy identically before the first call.
        Its dtype must match the descriptor's dtype, which may differ from
        the input dtype. Read the final state from block thread zero.
        Stateless prefix callbacks and calls without a prefix callback
        require ``None``.
    mode : {"exclusive", "inclusive"}, optional
        Compile-time choice, default ``"exclusive"``. Exclusive prefixes
        exclude the current input; inclusive prefixes include it.
    scan_op : str or device callable, optional
        Compile-time binary operator. Accepts the built-in strings from
        :func:`cuda.coop.scan`, their ``operator``/NumPy callable aliases,
        or a stateless device function ``op(left, right)``. The function
        must be associative and return the input dtype. ``None`` selects
        sum. Stateful binary operators are unsupported.
    initial_value : numeric scalar, optional
        Exclusive-scan seed, uniform across the group. Defaults to zero
        for sum. Other exclusive operators require a seed unless
        ``prefix_op`` supplies it. Typed scalars must match the input dtype;
        Python literals must be finite and in range. Inclusive scans and
        calls with ``prefix_op`` require ``None``.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``. ``None`` selects ``"raking"`` for blocks.
        ``"warp_scans"`` requires a block size divisible by 32. Warp groups
        require ``None``; see :func:`cuda.coop.scan` for algorithm details.
    temp_storage : cuda.coop.TempStorageLike, optional
        Block :ref:`scratch descriptor <coop-temp-storage>`. ``None`` uses
        automatic scratch and reuse synchronization. Warp groups require
        ``None``. Keep automatic synchronization enabled when carrying a
        prefix across successive calls unless the kernel provides the
        required block barriers.
    valid_items : int or integer scalar, optional
        Warp-only count of contributing lanes, from one through the group
        size, uniform within that group. ``None`` includes the full group.
        All lanes execute the call, but only lanes whose group rank is
        below this count have defined scan results. It does not enable
        partial Block scans or permit a partially launched physical warp.
    aggregate_output : ThreadData or local array, optional
        Writable one-item payload with the input dtype. Receives the
        aggregate of the input values on every member, excluding any
        ``initial_value``. For a partial Warp scan, the aggregate includes
        only the valid prefix. Must be ``None`` when ``prefix_op`` is used.
    prefix_op : device callable or StatefulFunction, optional
        Block-only callback supplying the prefix preceding this tile.
        A stateless callback takes ``tile_aggregate``. A ``StatefulFunction``
        takes ``(state, tile_aggregate)`` and may update its one-item state.
        Both return a prefix in the input dtype. CUB invokes the callback
        in the block's first warp and applies lane zero's returned prefix.
        Requires ``initial_value=None`` and ``aggregate_output=None``.
        See :ref:`coop-prefix-callbacks` for ownership and repeated calls.

    Returns
    -------
    numeric scalar or per-thread payload
        This thread's prefixes, with the input dtype and item count. A
        payload input produces a fresh payload. With ``valid_items``, read
        only the valid lanes' scan results. ``aggregate_output`` and
        ``prefix_state`` are separate output/state arguments, not parts
        of the return value.

    Notes
    -----
    Prefixes restart for each group unless a Block prefix callback carries
    state across successive tiles processed by that block. A wider state
    dtype does not widen the scan outputs. Arithmetic has the same integer
    overflow and floating-point regrouping limits as :func:`cuda.coop.scan`.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing Scan, Sum, aggregate, and prefix
        callback overloads.

    Examples
    --------
    The :func:`~cuda.coop.numba_mlir.inclusive_scan` example supplies a
    stateless binary device function. The
    :func:`~cuda.coop.numba_mlir.exclusive_scan` example combines a valid
    Warp prefix, initial value, and aggregate output. Carrying a stateful
    prefix across tiles is shown under
    :func:`~cuda.coop.numba_mlir.exclusive_sum`.
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
    """Return an exclusive prefix using a built-in or device operator.

    Extends :func:`cuda.coop.exclusive_scan` with the qualified options
    documented in :func:`cuda.coop.numba_mlir.scan`. The mode is fixed to
    exclusive. Each output combines the seed with preceding input values.
    Non-sum operators require ``initial_value`` or a Block ``prefix_op``.

    Blocks accept scalar or fixed-size :ref:`per-thread payloads
    <coop-thread-data>`; warps accept scalars. The return has the input
    dtype and item count and preserves the input. All members must
    participate; with Warp ``valid_items``, only the valid prefix has
    defined scan results. ``aggregate_output`` receives the input aggregate
    on every member and excludes ``initial_value``. Prefix callbacks cannot
    be combined with an initial value or aggregate output.

    See :func:`cuda.coop.numba_mlir.scan` for the full parameter list,
    callback state rules, algorithms, and :ref:`scratch <coop-temp-storage>`.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``ExclusiveScan``.

    Examples
    --------
    Scan the first five lanes of each eight-lane logical warp with an
    initial value. Every lane participates and receives the valid input
    aggregate; the kernel writes prefixes only for the first five lanes.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-exclusive-scan-example-begin
        :end-before: # qualified-exclusive-scan-example-end
        :dedent: 4
    """

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
    """Return an inclusive prefix using a built-in or device operator.

    Extends :func:`cuda.coop.inclusive_scan` with the qualified options
    documented in :func:`cuda.coop.numba_mlir.scan`. The mode is fixed to
    inclusive and there is no ``initial_value``. Each output includes its
    corresponding input. A Block ``prefix_op`` can supply a prefix to
    combine with every result; see :ref:`coop-prefix-callbacks`.

    Blocks accept scalar or fixed-size :ref:`per-thread payloads
    <coop-thread-data>`; warps accept scalars. The return has the input
    dtype and item count and preserves the input. Warp ``valid_items``
    restricts defined results to the valid prefix while all lanes still
    participate. ``aggregate_output`` receives the input aggregate on
    every member and cannot be combined with ``prefix_op``.

    See :func:`cuda.coop.numba_mlir.scan` for the full parameter list,
    callback state rules, algorithms, and :ref:`scratch <coop-temp-storage>`.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``InclusiveScan``.

    Examples
    --------
    Supply a stateless binary device function and a local array holding
    two items per thread. Maximum also has a built-in ``"max"`` spelling;
    this example shows the device-function interface for an associative
    operator.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-inclusive-scan-example-begin
        :end-before: # qualified-inclusive-scan-example-end
        :dedent: 4
    """

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
    """Return exclusive sums, optionally carrying a prefix across Block tiles.

    Extends :func:`cuda.coop.exclusive_sum` with the qualified options
    documented in :func:`cuda.coop.numba_mlir.scan`. The mode is exclusive
    and the operator is sum. The first result is zero unless a Block
    ``prefix_op`` supplies it. Use :func:`cuda.coop.numba_mlir.exclusive_scan`
    for an explicit ``initial_value`` or a different operator.

    Blocks accept scalar or fixed-size :ref:`per-thread payloads
    <coop-thread-data>`; warps accept scalars. The return has the input
    dtype and item count and preserves the input. Warp ``valid_items``
    restricts defined results to the valid prefix while all lanes still
    participate. ``aggregate_output`` receives the input sum on every
    member and cannot be combined with ``prefix_op``.

    See :func:`cuda.coop.numba_mlir.scan` for the full parameter list,
    callback state rules, algorithms, and :ref:`scratch <coop-temp-storage>`.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``ExclusiveSum``.

    Examples
    --------
    Scan three successive tiles in one block. Each thread initializes its
    own callback state; only block thread zero writes the final state.
    Automatic scratch barriers separate successive uses. See
    :ref:`coop-prefix-callbacks` for the state ownership rules.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-exclusive-sum-example-begin
        :end-before: # qualified-exclusive-sum-example-end
        :dedent: 4
    """

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
    """Return inclusive sums, optionally carrying a prefix across Block tiles.

    Extends :func:`cuda.coop.inclusive_sum` with the qualified options
    documented in :func:`cuda.coop.numba_mlir.scan`. The mode is inclusive
    and the operator is sum. Each output includes its corresponding input.
    A Block ``prefix_op`` can supply a prefix to add to every result.

    Blocks accept scalar or fixed-size :ref:`per-thread payloads
    <coop-thread-data>`; warps accept scalars. The return has the input
    dtype and item count and preserves the input. Warp ``valid_items``
    restricts defined results to the valid prefix while all lanes still
    participate. ``aggregate_output`` receives the input sum on every
    member and cannot be combined with ``prefix_op``.

    See :func:`cuda.coop.numba_mlir.scan` for the full parameter list,
    callback state rules, algorithms, and :ref:`scratch <coop-temp-storage>`.
    The :func:`cuda.coop.numba_mlir.exclusive_sum` example demonstrates
    carrying a :ref:`stateful prefix <coop-prefix-callbacks>` across tiles;
    replacing that call with ``inclusive_sum`` includes the current item
    in each prefix and leaves the callback state update unchanged.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``InclusiveSum``.
    """

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
