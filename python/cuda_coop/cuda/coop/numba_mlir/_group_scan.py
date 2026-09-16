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
    """Scan with device operators, aggregate outputs, or prefix callbacks.

    Extends :func:`cuda.coop.scan` with the options below. Group requirements,
    modes, algorithms, and temporary storage follow the portable function.

    Parameters
    ----------
    value : numeric scalar, ThreadData, or local array
        Blocks also accept a fixed-size one-dimensional local array, with
        the same dtype and extent across threads. Warps accept scalars only.
    prefix_state : ThreadData or local array, optional
        Writable one-item state for a
        :class:`~cuda.coop.numba_mlir.StatefulFunction`, supplied as the
        third positional argument. Its dtype must match the descriptor's
        dtype, which may differ from the input dtype. Initialize every
        thread's copy identically; read the final state from block thread
        zero. Requires ``None`` for stateless callbacks or no callback.
    scan_op : str or device callable, optional
        Also accepts ``operator``/NumPy aliases for the built-in strings or
        a stateless device function ``op(left, right)``. The compile-time
        operator must be associative and return the input dtype. ``None``
        selects sum. Exclusive custom scans require ``initial_value`` unless
        ``prefix_op`` supplies the seed. Stateful binary operators are
        unsupported.
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
    prefix_op : device callable or StatefulFunction, optional
        Block-only callback supplying this tile's prefix. A stateless
        callback takes ``tile_aggregate``; a ``StatefulFunction`` takes
        ``(state, tile_aggregate)`` and may update the state. Both return the
        input dtype. CUB invokes the callback in the first warp and applies
        lane zero's prefix. Requires ``initial_value=None`` and
        ``aggregate_output=None``. See :ref:`coop-prefix-callbacks`.

    Returns
    -------
    numeric scalar or per-thread payload
        This thread's prefixes, with the input dtype and item count. A
        payload input produces a fresh payload and remains unchanged. With
        ``valid_items``, read only valid lanes. ``aggregate_output`` and
        ``prefix_state`` are written separately.

    Notes
    -----
    A stateful prefix callback can carry a prefix across successive tiles in
    one block. Keep automatic scratch synchronization enabled or supply the
    required block barriers between calls. A wider state dtype does not
    widen the scan outputs.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ Scan, Sum, aggregate, and prefix callback overloads.

    Examples
    --------
    See :func:`~cuda.coop.numba_mlir.inclusive_scan` for a device operator,
    :func:`~cuda.coop.numba_mlir.exclusive_scan` for a partial warp and
    aggregate output, and :func:`~cuda.coop.numba_mlir.exclusive_sum` for a
    stateful prefix across tiles.
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

    Extends :func:`cuda.coop.exclusive_scan` with the parameters and return
    behavior of :func:`cuda.coop.numba_mlir.scan`, with exclusive mode fixed.
    Non-sum operators require ``initial_value`` or a block ``prefix_op``.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``ExclusiveScan``.

    Examples
    --------
    Scan five lanes of each eight-lane logical warp with an initial value.
    All lanes participate and receive the aggregate; only valid lanes write
    prefixes.

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

    Extends :func:`cuda.coop.inclusive_scan` with the parameters and return
    behavior of :func:`cuda.coop.numba_mlir.scan`, with inclusive mode fixed
    and no ``initial_value``. A block ``prefix_op`` combines its prefix with
    every result.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``InclusiveScan``.

    Examples
    --------
    Supply a device maximum operator and a local array with two items per
    thread. The built-in ``"max"`` operator gives the same result.

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
    """Return exclusive sums, optionally carrying a prefix across block tiles.

    Extends :func:`cuda.coop.exclusive_sum` with the parameters and return
    behavior of :func:`cuda.coop.numba_mlir.scan`, with exclusive mode and
    sum fixed. Each group starts from zero unless a block ``prefix_op``
    supplies its prefix. Use :func:`cuda.coop.numba_mlir.exclusive_scan` for
    an explicit ``initial_value`` or a different operator.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ collective types providing ``ExclusiveSum``.

    Examples
    --------
    Scan three tiles in one block. Each thread initializes its callback
    state; block thread zero writes the final state. Automatic scratch
    barriers separate calls; see :ref:`coop-prefix-callbacks`.

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
    """Return inclusive sums, optionally carrying a prefix across block tiles.

    Extends :func:`cuda.coop.inclusive_sum` with the parameters and return
    behavior of :func:`cuda.coop.numba_mlir.scan`, with inclusive mode and
    sum fixed. A block ``prefix_op`` adds its prefix to every result.

    Replace ``exclusive_sum`` in its
    :func:`stateful example <cuda.coop.numba_mlir.exclusive_sum>` with this
    function to include the current item in each prefix. The callback state
    update is unchanged.

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
