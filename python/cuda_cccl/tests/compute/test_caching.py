# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Behavioral tests for the cached algorithm wrappers."""

import gc
import weakref

import numpy as np
import pytest
from _utils.device_array import DeviceArray
from cuda.core import Device

import cuda.compute
from cuda.compute import OpKind, TransformIterator, types

NUM_ITEMS = 1024


def _require_jit_backend():
    """Skip when the JIT backend is absent, as on the minimal extras."""
    pytest.importorskip(
        "numba_cuda_mlir",
        reason="Python-callable operators require numba-cuda-mlir",
    )


def _double(x):
    return x * 2


def _run_reduce_into():
    d_in = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.int32))
    d_out = DeviceArray.empty(1, np.int32)
    h_init = np.zeros(1, dtype=np.int32)
    cuda.compute.reduce_into(
        d_in=d_in, d_out=d_out, num_items=NUM_ITEMS, op=OpKind.PLUS, h_init=h_init
    )
    return [d_in, d_out]


def _run_reduce_into_transform_iterator():
    _require_jit_backend()
    d_in = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.int32))
    d_out = DeviceArray.empty(1, np.int32)
    h_init = np.zeros(1, dtype=np.int32)
    cuda.compute.reduce_into(
        d_in=TransformIterator(d_in, _double),
        d_out=d_out,
        num_items=NUM_ITEMS,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    return [d_in, d_out]


def _run_reduce_into_transform_iterator_with_captured_array():
    _require_jit_backend()
    # A stateful transform op: the TransformIterator's cache key must not
    # embed the op adapter, which holds the function and its captured array.
    # Stateful ops cannot infer their return type, so it is given explicitly.
    scale = DeviceArray.from_numpy(np.array([3], dtype=np.int32))

    def scale_op(x):
        return x * scale[0]

    d_in = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.int32))
    d_out = DeviceArray.empty(1, np.int32)
    h_init = np.zeros(1, dtype=np.int32)
    cuda.compute.reduce_into(
        d_in=TransformIterator(d_in, scale_op, value_type=types.int32),
        d_out=d_out,
        num_items=NUM_ITEMS,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    return [d_in, d_out, scale]


def _run_unary_transform_with_captured_array():
    _require_jit_backend()
    # ``scale`` is captured by the operator's closure, which makes this a
    # stateful operator. The operator is keyed in the caches through
    # CachableFunction, which must not keep the function, and with it the
    # captured array, alive.
    scale = DeviceArray.from_numpy(np.array([3], dtype=np.int32))

    def scale_op(x):
        return x * scale[0]

    d_in = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.int32))
    d_out = DeviceArray.empty(NUM_ITEMS, np.int32)
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=scale_op, num_items=NUM_ITEMS
    )
    return [d_in, d_out, scale]


def _run_unique_by_key():
    d_in_keys = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.int32) // 4)
    d_in_items = DeviceArray.from_numpy(np.arange(NUM_ITEMS, dtype=np.float32))
    d_out_keys = DeviceArray.empty(NUM_ITEMS, np.int32)
    d_out_items = DeviceArray.empty(NUM_ITEMS, np.float32)
    d_out_num_selected = DeviceArray.empty(1, np.int32)
    cuda.compute.unique_by_key(
        d_in_keys=d_in_keys,
        d_in_items=d_in_items,
        d_out_keys=d_out_keys,
        d_out_items=d_out_items,
        d_out_num_selected=d_out_num_selected,
        op=OpKind.EQUAL_TO,
        num_items=NUM_ITEMS,
    )
    return [d_in_keys, d_in_items, d_out_keys, d_out_items, d_out_num_selected]


@pytest.mark.parametrize(
    "run_algorithm",
    [
        _run_reduce_into,
        _run_reduce_into_transform_iterator,
        _run_reduce_into_transform_iterator_with_captured_array,
        _run_unary_transform_with_captured_array,
        _run_unique_by_key,
    ],
    ids=[
        "reduce_into",
        "reduce_into_transform_iterator",
        "reduce_into_transform_iterator_with_captured_array",
        "unary_transform_with_captured_array",
        "unique_by_key",
    ],
)
def test_cached_wrapper_does_not_retain_caller_arrays(run_algorithm):
    # The single-phase API caches the algorithm object it builds, keyed by
    # argument types, for the rest of the process. That object must not keep
    # the arrays of any particular call alive: a user iterating over many type
    # combinations would otherwise accumulate device memory that cannot be
    # released (NVIDIA/cccl#12187).
    arrays = run_algorithm()
    Device().sync()
    array_refs = [weakref.ref(array) for array in arrays]

    del arrays
    gc.collect()

    still_alive = [ref() for ref in array_refs if ref() is not None]
    assert still_alive == []
