# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from functools import cache

from .._caching import cache_with_registered_key_functions
from .._cpp_compile import compile_cpp_op_code
from .._serialization import NESTED, Serializable
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators import DiscardIterator
from ..op import OpAdapter, RawOp, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator
from ._three_way_partition import _ThreeWayPartition, make_three_way_partition


@cache
def _always_false_op(_target_cc):
    # ``_target_cc`` (the build's get_target_cc()) is part of the cache key so the
    # predicate's LTO-IR is recompiled per target arch: this RawOp is linked into
    # the three-way-partition build, and nvJitLink rejects a newer-arch input in
    # an older-arch result. Without the key, the first build's arch would leak
    # into every later build (module-global cache). compile_cpp_op_code() reads
    # the same target internally; the arg only distinguishes cache entries.
    source = """
extern "C" __device__ void always_false(void*, void* result) {{
    *static_cast<bool*>(result) = false;
}}
"""
    code = compile_cpp_op_code(source)
    return RawOp(ltoir=code, name="always_false")


def _get_always_false_op():
    """The always-false predicate compiled for the current build's target cc."""
    from .._target_cc import get_target_cc

    return _always_false_op(get_target_cc())


class _Select(Serializable):
    __slots__ = ["_bound_build_result", "partitioner", "always_false_op", "_discards"]

    __serialization_schema__ = (("partitioner", NESTED(_ThreeWayPartition)),)

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        d_num_selected_out: DeviceArrayLike,
        cond: OpAdapter,
        compute_capability=None,
    ):
        self.always_false_op = _get_always_false_op()
        d_second, d_unselected = self._discard_iterators(d_out)
        self.partitioner = make_three_way_partition(
            d_in=d_in,
            d_first_part_out=d_out,
            d_second_part_out=d_second,
            d_unselected_out=d_unselected,
            d_num_selected_out=d_num_selected_out,
            select_first_part_op=cond,
            select_second_part_op=self.always_false_op,
            compute_capability=compute_capability,
        )

    def _discard_iterators(self, d_out):
        # The second/unselected outputs are discarded; their iterators depend
        # only on d_out's type, so build the pair once and cache it. Bound
        # lazily (on first construction or first call) so a deserialized
        # _Select, which has no construction d_out, builds them on first use.
        try:
            return self._discards
        except AttributeError:
            self._discards = (DiscardIterator(d_out), DiscardIterator(d_out))
            return self._discards

    def _after_deserialize(self) -> None:
        # always_false_op (the always-false second predicate) is not serialized.
        # Its compiled LTO-IR is already baked into the (serialized) three-way
        # partition build result, and __call__ reads only this op's runtime state
        # (which is empty — the predicate is stateless). So reconstruct an
        # empty-state stand-in WITHOUT compiling: deserialize() must neither
        # recompile nor require a GPU, and calling _get_always_false_op() here
        # would do both (cold cache -> compile_cpp_op_code -> Device() fallback).
        self.always_false_op = RawOp(ltoir=b"", name="always_false")

    def __call__(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        d_num_selected_out,
        cond,
        num_items: int,
        stream=None,
    ):
        d_second, d_unselected = self._discard_iterators(d_out)
        return self.partitioner(
            temp_storage=temp_storage,
            d_in=d_in,
            d_first_part_out=d_out,
            d_second_part_out=d_second,
            d_unselected_out=d_unselected,
            d_num_selected_out=d_num_selected_out,
            select_first_part_op=make_op_adapter(cond),
            select_second_part_op=self.always_false_op,
            num_items=num_items,
            stream=stream,
        )


@cache_with_registered_key_functions
def make_select(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
    compute_capability=None,
):
    """
    Create a select object that can be called to select elements matching a condition.

    This is the object-oriented API that allows explicit control over temporary
    storage allocation. For simpler usage, consider using :func:`select`.

    Example:
        Below, ``make_select`` is used to create a select object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_object.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the selected output items.
        d_num_selected_out: Device array to store the number of items that passed the selection.
            The count is stored in ``d_num_selected_out[0]``.
        cond: Selection condition (predicate).
            The signature is ``(T) -> uint8``, where ``T`` is the input data type.
            Returns 1 (selected) or 0 (not selected).
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that performs the selection operation.
    """
    cond_adapter = make_op_adapter(cond)
    # Note: _Select internally calls make_three_way_partition which will
    # normalize the cond. But we've already normalized it, so the Op
    # will be passed through make_op unchanged.
    return _Select(
        d_in,
        d_out,
        d_num_selected_out,
        cond_adapter,
        compute_capability=compute_capability,
    )


def select(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    d_num_selected_out: DeviceArrayLike,
    cond: Operator,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide selection of elements based on a condition.

    Given an input sequence, this function selects all elements for which the condition
    function ``cond`` returns true (non-zero) and writes them to the output in a
    compacted form. The number of selected elements is written to ``d_num_selected_out[0]``.

    This function automatically handles temporary storage allocation and execution.

    The ``cond`` function can reference device arrays as globals or closures - they will
    be automatically captured as state arrays, enabling stateful operations like counting.

    Example:
        Below, ``select`` is used to select even numbers from an input array:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_basic.py
            :language: python
            :start-after: # example-begin

        You can also use iterators for more complex selection patterns:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/select/select_with_iterator.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items.
        d_out: Device array or iterator to store the selected output items.
        d_num_selected_out: Device array to store the number of items that passed the selection.
            The count is stored in ``d_num_selected_out[0]``.
        cond: Selection condition (predicate).
            The signature is ``(T) -> uint8``, where ``T`` is the input data type.
            Returns 1 (selected) or 0 (not selected).
            Can reference device arrays as globals/closures - they will be automatically captured.
        num_items: Number of items in the input sequence.
        stream: CUDA stream to use for the operation (optional).
    """
    # Create adapter to support stateful ops
    cond_adapter = make_op_adapter(cond)
    selector = make_select(
        d_in=d_in, d_out=d_out, d_num_selected_out=d_num_selected_out, cond=cond_adapter
    )

    tmp_storage_bytes = selector(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected_out,
        cond=cond_adapter,
        num_items=num_items,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    selector(
        temp_storage=tmp_storage,
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected_out,
        cond=cond_adapter,
        num_items=num_items,
        stream=stream,
    )
