# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A batched reduction gives one result for each item of a lane's payload.

Item ``b`` of every lane belongs to batch ``b``, and
:func:`~cuda.coop.cutlass.reduce_batched` reduces each batch across the
lanes of a warp or of a group of lanes. A payload of three items gives three
results, where :func:`~cuda.coop.cutlass.reduce` gives one.

.. figure:: visualizations/reduce-batched.svg
   :alt: Four lanes hold three items each: 1, 2, 3 in lane 0, then 3, 4, 5,
         then 5, 6, 7, then 7, 8, 9. The three batches sum to 16, 20 and
         24. Lane 0 returns 16, lane 1 returns 20, lane 2 returns 24 and
         lane 3 returns a question mark.
   :width: 100%

   Three batches over a group of four lanes. The sums are 16, 20 and 24.
   Lanes 0, 1 and 2 each hold one of them, and lane 3 holds no result. The
   picture calls the group a logical warp and an item a slot.

Each result is in one lane. With ``B`` batches and ``W`` lanes, every lane
gets a payload of ``ceil(B / W)`` items back, and ``output_layout`` sets
which batches a lane holds. Ten batches over four lanes give three items
per lane:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - ``output_layout``
     - Lane 0
     - Lane 1
     - Lane 2
     - Lane 3
   * - ``"striped"`` (default)
     - 0, 4, 8
     - 1, 5, 9
     - 2, 6, none
     - 3, 7, none
   * - ``"blocked"``
     - 0, 1, 2
     - 3, 4, 5
     - 6, 7, 8
     - 9, none, none

An item marked "none" has no batch, and its value is undefined.
"""

# Implementation notes.
#
# Adapt readable or register payloads for batched warp reduction. Input
# slot i in every lane belongs to batch i. Check the group, normalize the
# built-in operator and output layout, and copy the input. Then require the
# enclosing block to hold only complete 32-thread warps and pass the call to
# the CUB lowering. Batch totals are spread across lanes, and some lanes may
# get none.

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal, TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    NumericScalar,
    ReadableThreadDataLike,
    ReduceOperator,
    WarpGroup,
)

from ._compiler._launch import current_kernel_launch_facts
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _snapshot_readable_payload,
)
from ._thread_group import _require_complete_warp_partition

_ItemT = TypeVar("_ItemT", bound=NumericScalar)


def reduce_batched(
    group: WarpGroup,
    value: ReadableThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    binary_op: ReduceOperator
    | Callable[[object, object], object]
    | None = None,
    output_layout: Literal["striped", "blocked"] = "striped",
) -> ThreadData:
    """Reduce each batch across the lanes of a warp or a group of lanes.

    Item ``b`` of every lane's payload is batch ``b``. Each batch is reduced
    with ``binary_op``, and its result goes to one lane.

    Parameters
    ----------
    group : ThreadGroup
        ``this_warp()``, or ``this_warp().group_by(n)`` with ``n`` of 1, 2,
        4, 8, 16 or 32. Every lane of the group makes the call, including a
        lane that receives no result. The other groups of lanes of the warp
        can skip the call. The block's size must be a multiple of 32.
    value : ThreadData, CuTe register tensor or TensorSSA
        A payload of ``B`` items, one for each batch. A register tensor or
        ``TensorSSA`` is read as a payload. An item has one of the ten
        element types listed under :ref:`ThreadData <coop-thread-data>`.
    binary_op : str or function, optional
        The operator, sum by default. A Python value: a name or a function
        from the table of operators under
        :ref:`Reduction <coop-cutlass-reduce>`. The rules that
        :func:`~cuda.coop.cutlass.reduce` gives for bitwise operators and
        for NaN hold here too. The seven built-in operators are the only
        ones this call takes, and a ``DeviceOperator`` is rejected when the
        kernel is compiled.
    output_layout : {"striped", "blocked"}, optional
        A Python string that sets which batch each item of the result
        holds. ``W`` is the number of lanes in the group, and ``r`` is this
        lane's rank in it. With ``"striped"``, the default, item ``i``
        holds batch ``r + i * W``. With ``"blocked"``, item ``i`` holds
        batch ``r * ceil(B / W) + i``. When ``B`` is at most ``W`` the two
        are the same: lane ``r`` holds batch ``r``.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        A new payload of ``ceil(B / W)`` items, with the element type and
        alignment of ``value``. An item whose batch number is below ``B``
        holds that batch's result. Every other item is undefined, so test
        the batch number before reading an item. The input keeps its
        values.

    Notes
    -----
    The lanes combine their values in registers. The call uses no scratch
    and adds no barrier, so it takes no ``temp_storage``.

    Integer results are computed in the element type.

    See Also
    --------
    :cpp:struct:`cub::WarpReduceBatched`
        The CUB class behind the call.

    Examples
    --------
    Sum each batch over each of the two warps of a 64-thread block. The
    layout is striped, so ``totals[item]`` holds the sum of batch
    ``warp.rank() + item * 32``. The kernel names that number ``batch``. A
    lane stores the sum only when ``batch`` is below ``items_per_thread``,
    the number of batches.

    `Full example and test on GitHub
    <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_reduce_batched_documentation.py>`__

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_reduce_batched_documentation.py
        :language: python
        :start-after: # cutlass-reduce-batched-example-begin
        :end-before: # cutlass-reduce-batched-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError(
            "cuda.coop.cutlass.reduce_batched group must be a ThreadGroup"
        )
    if not isinstance(output_layout, str) or isinstance(output_layout, Enum):
        raise TypeError("reduce_batched output_layout must be a string")
    output_layout = output_layout.strip().lower().replace("-", "_")
    if output_layout not in {"striped", "blocked"}:
        raise ValueError(
            "reduce_batched output_layout must be striped or blocked"
        )
    from ._operators import normalize_operator

    op = normalize_operator(binary_op, primitive="reduce_batched")
    value = _snapshot_readable_payload(
        value, name="value", primitive="reduce_batched"
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="reduce_batched", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._reduce_batched import provider_reduce_batched

    return provider_reduce_batched(
        group=group,
        launch=launch,
        value=value,
        op=op,
        output_layout=output_layout,
    )


__all__ = ["reduce_batched"]
