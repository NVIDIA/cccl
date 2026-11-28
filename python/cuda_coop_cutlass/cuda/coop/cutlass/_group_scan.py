# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A scan gives each item the running result of the items before it.

Items are ordered across the group thread by thread: all of thread 0's
items, then thread 1's, and so on. An exclusive scan leaves each item out of
its own result, and an inclusive scan includes it.

.. figure:: visualizations/scan.svg
   :alt: Eight threads hold the values 1 to 16, two each. The sum of all
         earlier threads' items enters each thread, and that thread's two
         results continue from it: 0 and 1 in thread 0, 3 and 6 in thread
         1, up to 105 and 120 in thread 7.
   :width: 100%

   ``exclusive_sum`` over eight threads with two items each. The middle row
   is the sum of all earlier threads' items. ``inclusive_sum`` of the same
   values is 1, 3, 6, 10 and so on, up to 136.

``scan`` takes every choice as an argument, and the four helpers make some
of the choices for you:

.. list-table::
   :header-rows: 1
   :widths: 24 24 24 28

   * - Function
     - Mode
     - Operator
     - Starts from
   * - ``scan``
     - ``mode``, exclusive by default
     - ``scan_op``, sum by default
     - ``initial_value``, in exclusive mode
   * - ``exclusive_scan``
     - Exclusive
     - ``scan_op``, sum by default
     - ``initial_value``
   * - ``inclusive_scan``
     - Inclusive
     - ``scan_op``, sum by default
     - The first item
   * - ``exclusive_sum``
     - Exclusive
     - Sum
     - Zero
   * - ``inclusive_sum``
     - Inclusive
     - Sum
     - The first item

On a block, each thread passes a payload or one scalar and gets the same
form back. On a warp or a group of lanes, each thread passes one scalar. All
five functions take these controls:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Control
     - Groups
     - What it does
   * - ``aggregate_output``
     - All
     - Gives every thread the result of combining the inputs that were
       scanned.
   * - ``valid_items``
     - Warps and groups of lanes
     - Scans only the first ``valid_items`` lanes of the group.
   * - ``algorithm``
     - Blocks
     - Selects ``"raking"``, ``"raking_memoize"`` or ``"warp_scans"``.
   * - ``temp_storage``
     - Blocks
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.

Each group scans by itself, and a scan in a loop starts again on every
trip. To continue one scan across tiles, start each tile from the total so
far. In this kernel ``aggregate_output`` returns each tile's total and
``initial_value`` passes the running total to the next tile:

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
    :language: python
    :start-after: # cutlass-scan-running-total-example-begin
    :end-before: # cutlass-scan-running-total-example-end
    :dedent: 4

A scan reads a payload thread by thread, in :term:`blocked` order, whichever
call made the payload. If the payload was loaded with
``algorithm="striped"``, run it through
:func:`~cuda.coop.cutlass.exchange` before scanning it.
"""

# Implementation notes.
#
# Trace BlockScan and WarpScan calls from CuTe kernels. Normalize mode and
# algorithm strings and adapt register tensors and TensorSSA values to
# ThreadData before planning. The provider reconciles dtypes, owns scratch,
# and emits the C++ call. The public functions support built-in and compiled
# operators, per-warp valid prefixes, and aggregate outputs.

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal, TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    NumericScalar,
    ReadableThreadDataLike,
    ScanAlgorithm,
    ScanOperator,
    ValidItems,
)
from cuda.coop.cutlass._validation import _validate_temp_storage

from ._device_operator import DeviceOperator
from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _coerce_thread_payload,
)
from ._thread_group import _require_complete_warp_partition
from ._typing import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})


def _selector(value, *, name, choices):
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(f"{_SCOPE}.scan {name} must be a string")
    value = value.strip().lower().replace("-", "_")
    if value not in choices:
        raise ValueError(
            f"{_SCOPE}.scan {name} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def scan(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOperator
    | DeviceOperator
    | Callable[[object, object], object]
    | None = None,
    initial_value: NumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData | _ItemT:
    """Scan a group's values with an operator.

    Each result combines the items before it with ``scan_op``. Exclusive
    mode starts from ``initial_value`` and leaves the item itself out.
    Inclusive mode includes it. On a block, items are ordered thread by
    thread, by the threads' rank in the block.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or ``this_warp().group_by(n)``
        with ``n`` of 1, 2, 4, 8, 16 or 32. Every thread of the group makes
        the call, on every trip of a loop that contains it. A warp group
        needs a block whose size is a multiple of 32.
    value : scalar, readable payload, CuTe register tensor or TensorSSA
        On a block, one scalar per thread or a payload, with the same
        element type and item count in every thread. A register tensor or
        ``TensorSSA`` is read as a payload, and the result is a
        ``ThreadData`` that ``to_register_tensor`` or ``to_tensor_ssa``
        converts back. On a warp or a group of lanes, one scalar per thread:
        pass ``values[0]`` for a payload of one item. A scalar or an item
        has one of the ten element types listed under
        :ref:`ThreadData <coop-thread-data>`.
    mode : {"exclusive", "inclusive"}, optional
        A Python string. The default is ``"exclusive"``.
    scan_op : str, function or DeviceOperator, optional
        The operator, sum by default. It takes the same seven operators as
        :func:`~cuda.coop.cutlass.reduce`: ``"sum"``, ``"multiplies"``,
        ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"`` and ``"bit_xor"``,
        or the matching function from ``operator`` or NumPy, such as
        ``operator.add`` or ``numpy.maximum``. The three bitwise operators
        need an integer type. On floats, ``"min"`` and ``"max"`` follow
        ``fmin`` and ``fmax``, so a number wins over NaN. A
        :class:`~cuda.coop.cutlass.DeviceOperator` from
        :func:`~cuda.coop.cutlass.compile_operator` runs your own
        associative function on the element type of ``value``.
    initial_value : scalar, optional
        The value an exclusive scan starts from. It is the first result,
        and every later result includes it: with ``"max"`` and an initial
        value of 10, every result is at least 10. A sum starts from zero
        when it is left out. Every other operator needs it. Inclusive mode
        takes ``None``.

        A CuTe or NumPy scalar must have the element type of ``value``. A
        Python integer must fit that type. A Python float needs a floating
        type and must be finite, so start a floating ``"min"`` from a typed
        infinity such as ``numpy.float32(numpy.inf)``. The value can be
        computed while the kernel runs, and every thread of the group
        passes the same one.
    valid_items : int or CuTe integer, optional
        For warps and groups of lanes. The number of lanes that contribute,
        from one to the group's size, the same in every lane. Every lane
        still makes the call. Lanes at or past the count get no defined
        result. A Python count out of range is rejected when the kernel is
        compiled, and a runtime count out of range traps on the device.
        Blocks take ``None``.
    aggregate_output : ThreadData, optional
        A payload of one item, such as ``coop.ThreadData(1)``, that
        receives the result of combining the inputs that were scanned. The
        result has the element type of ``value``, so leave the payload's
        ``dtype`` out or give that type. Every thread receives it. It
        leaves out ``initial_value`` and, with ``valid_items``, the lanes
        at or past the count. Inside a runtime ``if`` or loop, create the
        payload there and read its item there, as in
        :ref:`this FAQ <coop-faq-scan-loop-aggregate>`.

    algorithm : str, optional
        For blocks. ``"raking"``, the default, is CUB's high-throughput
        algorithm. ``"raking_memoize"`` is the same with fewer
        shared-memory reads and more registers. ``"warp_scans"`` is CUB's
        low-latency algorithm and needs a block size that is a multiple of
        32. Warps and groups of lanes take ``None``.
    temp_storage : TempStorage, optional
        For blocks. Every block algorithm uses scratch. Leave this out and
        the compiler provides the scratch and a barrier after the call.
        Pass a :ref:`TempStorage <coop-temp-storage>` to share the bytes
        with other calls, such as Load and Store. A barrier then follows
        the call only if the ``TempStorage`` was created with
        ``auto_sync=True``. Warps and groups of lanes take ``None``.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        This thread's results, in the element type of ``value``. A scalar
        gives a scalar. A payload gives a new payload with the same item
        count and alignment, and the input keeps its values.

    Notes
    -----
    A scan regroups its operations, so the operator must be associative. It
    need not be commutative. Floating-point results can differ from a
    sequential loop, and integer results are computed in the element type.

    CUB's prefix callbacks have no counterpart here. The running-total
    kernel at the top of the :ref:`Scan <coop-cutlass-scan>` section does
    their usual job.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes behind the call.

    Examples
    --------
    Four scans of one payload: a sum that starts from seven, a running
    maximum, and the two sum helpers. The calls share one ``TempStorage``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
        :language: python
        :start-after: # cutlass-scan-example-begin
        :end-before: # cutlass-scan-example-end
        :dedent: 4
    """
    from ._compiler._launch import current_kernel_launch_facts
    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.scan requires a block or warp group"
        )
    mode = _selector(mode, name="mode", choices={"inclusive", "exclusive"})
    if algorithm is not None:
        algorithm = _selector(algorithm, name="algorithm", choices=_ALGORITHMS)
    op = normalize_operator(scan_op, primitive="scan")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(
            f"{_SCOPE}.scan inclusive mode does not accept initial_value"
        )
    if mode == "exclusive" and op != "sum" and initial_value is None:
        raise ValueError(f"{_SCOPE}.scan exclusive {op} requires initial_value")
    if group.kind != "block":
        if algorithm is not None:
            raise ValueError(
                f"{_SCOPE}.scan algorithm selection requires a block"
            )
        if temp_storage is not None:
            raise ValueError(
                f"{_SCOPE}.scan temp_storage is supported only for blocks"
            )
    elif temp_storage is not None:
        _validate_temp_storage("scan", temp_storage)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="scan", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._scan import provider_scan

    return provider_scan(
        group=group,
        launch=launch,
        value=value,
        mode=mode,
        op=op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_scan(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    scan_op: ScanOperator
    | DeviceOperator
    | Callable[[object, object], object]
    | None = None,
    initial_value: NumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData | _ItemT:
    """Combine the items before each item, starting from ``initial_value``.

    The same as :func:`~cuda.coop.cutlass.scan` in exclusive mode, with the
    same parameters apart from ``mode``. A sum starts from zero when
    ``initial_value`` is left out, and every other operator needs it.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        For each item, ``initial_value`` combined with the items before it.
        A payload gives a new payload. With ``valid_items``, only lanes
        below the count have a defined result.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes behind the call.

    Examples
    --------
    In each group of eight lanes, sum the first five lanes, with an initial
    value of seven. Every lane makes the call and receives the sum of those
    five lanes, without the seven, in ``aggregate``. Only those five lanes
    store a result.
    ``operator.add`` selects the built-in sum.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
        :language: python
        :start-after: # cutlass-exclusive-scan-example-begin
        :end-before: # cutlass-exclusive-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_scan(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    scan_op: ScanOperator
    | DeviceOperator
    | Callable[[object, object], object]
    | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData | _ItemT:
    """Combine each item with the items before it.

    The same as :func:`~cuda.coop.cutlass.scan` in inclusive mode, with the
    same parameters apart from ``mode`` and ``initial_value``, which an
    inclusive scan does not take.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        For each item, the items up to and including it, combined. A
        payload gives a new payload. With ``valid_items``, only lanes below
        the count have a defined result.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes behind the call.

    Examples
    --------
    ``running_maxima`` below is an inclusive scan with ``"max"``.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
        :language: python
        :start-after: # cutlass-scan-example-begin
        :end-before: # cutlass-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_sum(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData | _ItemT:
    """Sum the items before each item, starting from zero.

    The same as :func:`~cuda.coop.cutlass.scan` with its defaults. For a sum
    that starts from another value, pass ``initial_value`` to
    :func:`~cuda.coop.cutlass.exclusive_scan`.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        For each item, the sum of the items before it. A payload gives a
        new payload. With ``valid_items``, only lanes below the count have
        a defined result.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes behind the call.

    Examples
    --------
    ``before`` below holds the exclusive sums.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
        :language: python
        :start-after: # cutlass-scan-example-begin
        :end-before: # cutlass-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="exclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_sum(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadData | None = None,
) -> ThreadData | _ItemT:
    """Sum each item and the items before it.

    The same as :func:`~cuda.coop.cutlass.scan` in inclusive mode with the
    default operator.

    Returns
    -------
    CuTe scalar or cuda.coop.cutlass.ThreadData
        For each item, the sum of the items up to and including it. A
        payload gives a new payload. With ``valid_items``, only lanes below
        the count have a defined result.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        The CUB classes behind the call.

    Examples
    --------
    ``through`` below holds the inclusive sums. The
    :ref:`first program <coop-cutlass-load-store>` in the Programming Guide
    is a whole kernel built on this call.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
        :language: python
        :start-after: # cutlass-scan-example-begin
        :end-before: # cutlass-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="inclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
