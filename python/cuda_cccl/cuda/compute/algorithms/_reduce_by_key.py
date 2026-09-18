# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import set_cccl_iterator_state
from .._serialization import BUILD_RESULTS, ITER, OP, Serializable
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, IteratorT, Operator


class _ReduceByKey(Serializable):
    """Reduces runs of equal *adjacent* keys.

    Follows ``cub::DeviceReduce::ReduceByKey`` rather than the plain reduce: there is no initial value
    and no equality operator (keys compare with their own ``==``), and each run's representative key is
    its **last** key, not its first. The sequence is not sorted or grouped for you.
    """

    __slots__ = [
        "_bound_build_result",
        "build_results",
        "loaded_build_result",
        "d_keys_in_cccl",
        "d_values_in_cccl",
        "d_unique_out_cccl",
        "d_aggregates_out_cccl",
        "d_num_runs_out_cccl",
        "op_cccl",
    ]

    __serialization_schema__ = (
        ("d_keys_in_cccl", ITER),
        ("d_values_in_cccl", ITER),
        ("d_unique_out_cccl", ITER),
        ("d_aggregates_out_cccl", ITER),
        ("d_num_runs_out_cccl", ITER),
        ("op_cccl", OP),
        ("build_results", BUILD_RESULTS(_bindings.DeviceReduceByKeyBuildResult)),
    )

    def __init__(
        self,
        d_keys_in: DeviceArrayLike | IteratorT,
        d_values_in: DeviceArrayLike | IteratorT,
        d_unique_out: DeviceArrayLike | IteratorT,
        d_aggregates_out: DeviceArrayLike | IteratorT,
        d_num_runs_out: DeviceArrayLike,
        op: OpAdapter,
        compute_capability=None,
    ):
        self.d_keys_in_cccl = cccl.to_cccl_input_iter(d_keys_in)
        self.d_values_in_cccl = cccl.to_cccl_input_iter(d_values_in)
        self.d_unique_out_cccl = cccl.to_cccl_output_iter(d_unique_out)
        self.d_aggregates_out_cccl = cccl.to_cccl_output_iter(d_aggregates_out)
        self.d_num_runs_out_cccl = cccl.to_cccl_output_iter(d_num_runs_out)

        # The reduction operator combines values and yields a value; unlike unique_by_key's comparator
        # it is not a yes/no predicate.
        value_type = cccl.get_value_type(d_values_in)
        self.op_cccl = op.compile((value_type, value_type), value_type)

        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceReduceByKeyBuildResult,
            d_keys_in,
            d_values_in,
            d_unique_out,
            d_aggregates_out,
            d_num_runs_out,
            op,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceReduceByKeyBuildResult,
                self.d_keys_in_cccl,
                self.d_values_in_cccl,
                self.d_unique_out_cccl,
                self.d_aggregates_out_cccl,
                self.d_num_runs_out_cccl,
                self.op_cccl,
                compute_capability=compute_capability,
            ),
        )

    def __call__(
        self,
        *,
        temp_storage,
        d_keys_in: DeviceArrayLike | IteratorT,
        d_values_in: DeviceArrayLike | IteratorT,
        d_unique_out: DeviceArrayLike | IteratorT,
        d_aggregates_out: DeviceArrayLike | IteratorT,
        d_num_runs_out: DeviceArrayLike,
        op: Callable | OpAdapter,
        num_items: int,
        stream=None,
    ):
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )

        set_cccl_iterator_state(self.d_keys_in_cccl, d_keys_in)
        set_cccl_iterator_state(self.d_values_in_cccl, d_values_in)
        set_cccl_iterator_state(self.d_unique_out_cccl, d_unique_out)
        set_cccl_iterator_state(self.d_aggregates_out_cccl, d_aggregates_out)
        set_cccl_iterator_state(self.d_num_runs_out_cccl, d_num_runs_out)

        self.op_cccl.state = make_op_adapter(op).get_state()

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        # The run count stays on the device; it is deliberately never read back here.
        temp_storage_bytes = self.loaded_build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_keys_in_cccl,
            self.d_values_in_cccl,
            self.d_unique_out_cccl,
            self.d_aggregates_out_cccl,
            self.d_num_runs_out_cccl,
            num_items,
            self.op_cccl,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_registered_key_functions
def make_reduce_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_unique_out: DeviceArrayLike | IteratorT,
    d_aggregates_out: DeviceArrayLike | IteratorT,
    d_num_runs_out: DeviceArrayLike,
    op: Operator,
    compute_capability=None,
):
    """Builds a reusable reduce-by-key object.

    Args:
        d_keys_in: Device array of keys; only *adjacent* equality forms a run
        d_values_in: Device array of values to reduce
        d_unique_out: Output for the per-run representative key (the run's last key)
        d_aggregates_out: Output for the per-run aggregate
        d_num_runs_out: Single-element device output receiving the number of runs
        op: Binary reduction operator, ``(T, T) -> T``
        compute_capability: Compute capability to build for ahead of time

    Returns:
        A callable object that performs the reduction
    """
    return _ReduceByKey(
        d_keys_in,
        d_values_in,
        d_unique_out,
        d_aggregates_out,
        d_num_runs_out,
        make_op_adapter(op),
        compute_capability=compute_capability,
    )


def reduce_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_unique_out: DeviceArrayLike | IteratorT,
    d_aggregates_out: DeviceArrayLike | IteratorT,
    d_num_runs_out: DeviceArrayLike,
    op: Operator,
    num_items: int,
    stream=None,
):
    """Reduces runs of equal adjacent keys, allocating temporary storage internally.

    ``d_unique_out`` and ``d_aggregates_out`` must each hold ``num_items`` elements, the worst case
    where every key starts its own run.
    """
    reducer = make_reduce_by_key(
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_unique_out=d_unique_out,
        d_aggregates_out=d_aggregates_out,
        d_num_runs_out=d_num_runs_out,
        op=op,
    )
    tmp_storage_bytes = reducer(
        temp_storage=None,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_unique_out=d_unique_out,
        d_aggregates_out=d_aggregates_out,
        d_num_runs_out=d_num_runs_out,
        op=op,
        num_items=num_items,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(
        temp_storage=tmp_storage,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_unique_out=d_unique_out,
        d_aggregates_out=d_aggregates_out,
        d_num_runs_out=d_num_runs_out,
        op=op,
        num_items=num_items,
        stream=stream,
    )
