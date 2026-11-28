# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A reduction combines the values of a group's threads into one result.

Each thread passes one scalar or a payload, and every item of every payload
goes into the result. The thread with rank 0 in the group receives it: the
first thread of a block, lane 0 of a warp, and the first lane of each group
of lanes.

.. figure:: visualizations/reduce.svg
   :alt: Eight threads hold the values 1 to 16, two each. Each thread adds
         its own two values, which gives 3, 7, 11, 15, 19, 23, 27 and 31.
         Thread 0 returns 136, and the other seven threads return a
         question mark.
   :width: 100%

   ``sum`` over eight threads with two items each. Each thread adds its own
   items, and the group combines the eight sums into 136. Only thread 0
   holds that result.

``reduce`` takes the operator as ``binary_op``, and ``sum`` is ``reduce``
with addition. There are seven built-in operators. Each has a name, other
names that mean the same, and functions that select it:

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Operator
     - Other names
     - Functions
   * - ``"sum"``
     - ``"+"``, ``"add"``, ``"plus"``
     - ``operator.add``, ``numpy.add``
   * - ``"multiplies"``
     - ``"*"``, ``"mul"``, ``"multiply"``
     - ``operator.mul``, ``numpy.multiply``
   * - ``"min"``
     - ``"minimum"``
     - ``numpy.minimum``
   * - ``"max"``
     - ``"maximum"``
     - ``numpy.maximum``
   * - ``"bit_and"``
     - ``"&"``
     - ``operator.and_``, ``numpy.bitwise_and``
   * - ``"bit_or"``
     - ``"|"``
     - ``operator.or_``, ``numpy.bitwise_or``
   * - ``"bit_xor"``
     - ``"^"``
     - ``operator.xor``, ``numpy.bitwise_xor``

Both functions take these controls:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Control
     - Groups
     - What it does
   * - ``valid_items``
     - All
     - Reduces the scalars of the first ``valid_items`` threads of the
       group.
   * - ``algorithm``
     - Blocks
     - Selects ``"warp_reductions"``, ``"raking"`` or
       ``"raking_commutative_only"``.
   * - ``temp_storage``
     - Blocks
     - Puts the call's scratch in a :ref:`TempStorage <coop-temp-storage>`.

:func:`~cuda.coop.cutlass.reduce_batched`, further down, reduces item 0 of
every lane, item 1 of every lane, and so on, and gives one result for each
item of a lane's payload.
"""

# Implementation notes.
#
# Validate Reduce calls for CuTe kernels and adapt register payloads.
# Convert readable payloads, register tensors, and TensorSSA values to
# ThreadData. The planner selects CUB block or warp reductions and records
# root-only results and the scratch allocation policy.

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from numbers import Integral
from typing import TypeVar

from cuda.coop.cutlass._planning import ArgumentBinding
from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    NumericScalar,
    ReadableThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    ValidItems,
)

from ._compiler._launch import current_kernel_launch_facts
from ._device_operator import DeviceOperator
from ._group_load_store import _is_boolean
from ._temp_storage import TempStorage
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    _coerce_thread_payload,
)
from ._thread_group import _require_complete_warp_partition
from ._typing import BlockGroup, WarpGroup
from ._validation import _validate_temp_storage

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset(
    {"raking_commutative_only", "raking", "warp_reductions"}
)


def _classify_valid_items(value, *, primitive="reduce"):
    """Separate omitted, static, and runtime counts for Reduce and Scan.

    Python and NumPy integral values bind a static count; CuTe integer values
    supply runtime operands. Booleans are rejected. ``primitive`` selects the
    diagnostic name. Group planning checks static bounds, and generated
    wrappers check runtime bounds; classification alone does not establish a
    valid count.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.{primitive} valid_items must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_SCOPE}.{primitive} valid_items must be an integer")


def _normalize_algorithm(algorithm):
    """Normalize a CUB block algorithm name.

    Strip whitespace, lowercase, and replace hyphens with underscores. Accept
    strings for the three deterministic strategies; Enum values and the
    nondeterministic warp-reduction strategy are rejected. Leave None so the
    shared planner can choose a route.
    """

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
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | _ItemT
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    binary_op: ReduceOperator
    | DeviceOperator
    | Callable[[object, object], object]
    | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
) -> _ItemT:
    """Combine a group's values into one result with an operator.

    The operator combines every value the group's threads pass, or the
    values of the first ``valid_items`` threads. The thread with rank 0 in
    the group receives the result.

    Parameters
    ----------
    group : ThreadGroup
        ``this_block()``, ``this_warp()``, or ``this_warp().group_by(n)``
        with ``n`` of 1, 2, 4, 8, 16 or 32, or from 17 to 31. Every thread
        of the group makes the call, on every trip of a loop that contains
        it. A warp group needs a block whose size is a multiple of 32.

        A group of 17 to 31 lanes is lanes 0 to ``n - 1`` of each warp.
        Make it with ``group_by(n, exhaustive=False)`` and put the call
        inside ``if group.is_member():``, so that the other lanes skip it.
        :meth:`~cuda.coop.cutlass.ThreadGroup.group_by` has an example.
    value : scalar, readable payload, CuTe register tensor or TensorSSA
        One scalar per thread or a payload. Every item of a payload goes
        into the result. A register tensor or ``TensorSSA`` is read as a
        payload. A scalar or an item has one of the ten element types
        listed under :ref:`ThreadData <coop-thread-data>`. A Python integer
        passed as the scalar is taken as ``Int32`` and a Python float as
        ``Float32``.
    binary_op : str, function or DeviceOperator, optional
        The operator, sum by default. A Python value: a name or a function
        from the table of operators under
        :ref:`Reduction <coop-cutlass-reduce>`. The three bitwise operators
        need an integer type. On floats, ``"min"`` and ``"max"`` follow
        ``fmin`` and ``fmax``: a number wins over NaN, and the result is
        NaN only when every value that goes into it is NaN.

        A :class:`~cuda.coop.cutlass.DeviceOperator` from
        :func:`~cuda.coop.cutlass.compile_operator` runs your own
        associative function on the element type of ``value``. Any other
        function is rejected when the kernel is compiled, Python's ``min``
        and ``max`` included: write ``"min"`` and ``"max"``.
    valid_items : int or CuTe integer, optional
        The number of threads whose values go into the result, counted from
        rank 0. It is from one to the group's size and the same in every
        thread. ``valid_items`` needs a scalar from each thread, so pass
        ``values[0]`` for a payload of one item. Every thread still makes
        the call. A Python count out of range is rejected when the kernel
        is compiled, and a runtime count out of range traps on the device.
        ``None``, the default, takes every thread's value.
    algorithm : str, optional
        For blocks. A Python string. ``"warp_reductions"``, the default, is
        CUB's low-latency algorithm. ``"raking"`` is CUB's high-throughput
        algorithm. ``"raking_commutative_only"`` is the raking algorithm
        for commutative operators: it takes the seven built-in operators,
        and a ``DeviceOperator`` is rejected with it when the kernel is
        compiled. Warps and groups of lanes take ``None``.
    temp_storage : TempStorage, optional
        For blocks. Leave this out and the compiler provides the call's
        scratch and a barrier after the call. Pass a
        :ref:`TempStorage <coop-temp-storage>` to share the bytes with
        other calls, such as Load and Store. A barrier then follows the
        call only if the ``TempStorage`` was created with
        ``auto_sync=True``. Warps and groups of lanes take ``None``.

    Returns
    -------
    CuTe scalar
        The result, in the element type of ``value``. Only the thread with
        rank 0 in the group holds it: the first thread of a block, lane 0
        of a warp, the first lane of each group of lanes. The value in
        every other thread is undefined. The input keeps its values.

    Notes
    -----
    A reduction regroups its operations, so the operator must be
    associative. Floating-point results can differ from a sequential loop,
    and integer results are computed in the element type.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        The CUB classes behind the call.

    Examples
    --------
    Sum and maximum of one payload over a block of 64 threads. The thread
    with rank 0 stores both results.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_reduce_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_reduce_documentation.py
        :language: python
        :start-after: # cutlass-reduce-example-begin
        :end-before: # cutlass-reduce-example-end
        :dedent: 4

    Three reductions in one kernel. ``_THREADS`` is 64 and ``_BLOCK`` is
    ``(8, 4, 2)``. ``total`` is the sum of every item in the block.
    ``maximum`` is the largest ``payload[0]`` in each group of eight lanes.
    ``prefix`` is the sum of ``payload[0]`` over the 23 threads with the
    lowest rank. The thread with rank 0 in the block stores ``total`` and
    ``prefix``, and the first lane of each group stores its ``maximum``.

    `Full example on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/reduce.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/examples/reduce.py
        :language: python
        :start-after: docs: start cutlass-reduce
        :end-before: docs: end cutlass-reduce
        :dedent: 4
    """

    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.reduce requires a block, physical warp, "
            "or logical warp group"
        )
    if temp_storage is not None:
        if group.kind != "block":
            raise NotImplementedError(
                f"{_SCOPE}.reduce explicit TempStorage is supported only "
                "for block groups"
            )
        _validate_temp_storage("reduce", temp_storage)
    op = normalize_operator(binary_op)
    algorithm = _normalize_algorithm(algorithm)
    valid_binding = _classify_valid_items(valid_items)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
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
        temp_storage=temp_storage,
        valid_items=valid_items,
        valid_items_binding=valid_binding,
        algorithm=algorithm,
    )


def sum(
    group: BlockGroup | WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | _ItemT
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorage | None = None,
) -> _ItemT:
    """Add a group's values into one result.

    The same as :func:`~cuda.coop.cutlass.reduce` with ``binary_op="sum"``,
    with the same parameters apart from ``binary_op``.

    Returns
    -------
    CuTe scalar
        The sum, in the element type of ``value``. Only the thread with
        rank 0 in the group holds it. The input keeps its values.

    See Also
    --------
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        The CUB classes behind the call.

    Examples
    --------
    ``total`` below is the sum of every item in a block of 64 threads.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_reduce_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_reduce_documentation.py
        :language: python
        :start-after: # cutlass-reduce-example-begin
        :end-before: # cutlass-reduce-example-end
        :dedent: 4
    """
    return reduce(
        group,
        value,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = [
    "_classify_valid_items",
    "reduce",
    "sum",
]
