# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for bfloat16 support across cuda.compute algorithms.

NumPy has no native bfloat16; these tests use the ``ml_dtypes`` package, which
provides a NumPy bfloat16 extension dtype. numba cannot compile Python
callables for bfloat16, so all operators here are well-known ``OpKind``
operations (JIT-compiled entirely by the C library backends).
"""

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    ConstantIterator,
    OpKind,
    SortOrder,
    deserialize,
    make_reduce_into,
    serialize,
)
from cuda.compute import types as compute_types
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

ml_dtypes = pytest.importorskip("ml_dtypes")

BFLOAT16 = np.dtype(ml_dtypes.bfloat16)


def random_bfloat16(size, low=-10.0, high=10.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=size).astype(BFLOAT16)


def test_type_descriptor():
    td = compute_types.bfloat16
    assert td.size == 2
    assert td.alignment == 2
    assert td.dtype == BFLOAT16
    assert compute_types.from_numpy_dtype(BFLOAT16) is td


def test_reduce_sum_exact():
    # 0.5 * 200 elements: every partial sum is exactly representable in
    # bfloat16 (8 significand bits hold integers and halves up to 256 and 128
    # respectively), so the result is exact for any reduction order.
    num_items = 200
    h_input = np.full(num_items, 0.5, dtype=BFLOAT16)
    h_init = np.array([1.0], dtype=BFLOAT16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, BFLOAT16)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    h_output = d_output.copy_to_host()
    assert float(h_output[0]) == 101.0


def test_reduce_sum_random():
    num_items = 512
    h_input = random_bfloat16(num_items, seed=42)
    h_init = np.array([0.0], dtype=BFLOAT16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, BFLOAT16)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    h_output = d_output.copy_to_host()
    expected = h_input.astype(np.float32).sum()
    # bfloat16 has ~2-3 decimal digits of precision and the accumulator itself
    # is bfloat16; allow a generous relative error plus slack for cancellation.
    assert float(h_output[0]) == pytest.approx(expected, rel=0.15, abs=2.0)


@pytest.mark.parametrize(
    "op,reference",
    [(OpKind.MINIMUM, np.min), (OpKind.MAXIMUM, np.max)],
)
def test_reduce_min_max(op, reference):
    num_items = 1000
    h_input = random_bfloat16(num_items, seed=7)
    h_init = np.array([0.0], dtype=BFLOAT16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, BFLOAT16)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=op,
        h_init=h_init,
    )

    h_output = d_output.copy_to_host()
    expected = reference(np.append(h_input, h_init))
    assert h_output[0] == expected


def test_reduce_constant_iterator():
    num_items = 128
    d_input = ConstantIterator(np.array([0.5], dtype=BFLOAT16)[0])
    h_init = np.array([1.0], dtype=BFLOAT16)
    d_output = DeviceArray.empty(1, BFLOAT16)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    h_output = d_output.copy_to_host()
    assert float(h_output[0]) == 65.0


@pytest.mark.parametrize("force_inclusive", [True, False])
def test_scan_sum(force_inclusive):
    # 0/1 inputs keep every prefix sum exactly representable in bfloat16.
    num_items = 200
    rng = np.random.default_rng(3)
    h_input = rng.integers(0, 2, num_items).astype(BFLOAT16)
    h_init = np.array([0.0], dtype=BFLOAT16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(num_items, BFLOAT16)

    scan_algorithm = (
        cuda.compute.inclusive_scan if force_inclusive else cuda.compute.exclusive_scan
    )
    scan_algorithm(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=h_init,
        num_items=num_items,
    )

    h_output = d_output.copy_to_host()
    cumsum = h_input.astype(np.float32).cumsum()
    if force_inclusive:
        expected = cumsum
    else:
        expected = np.append([0.0], cumsum[:-1])
    np.testing.assert_array_equal(h_output.astype(np.float32), expected)


@pytest.mark.parametrize("order", [SortOrder.ASCENDING, SortOrder.DESCENDING])
def test_radix_sort_keys(order):
    num_items = 1000
    h_in_keys = random_bfloat16(num_items, seed=11)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(num_items, BFLOAT16)

    cuda.compute.radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        order=order,
        num_items=num_items,
    )

    h_out_keys = d_out_keys.copy_to_host()
    expected = np.sort(h_in_keys)
    if order is SortOrder.DESCENDING:
        expected = expected[::-1]
    np.testing.assert_array_equal(h_out_keys, expected)


def test_radix_sort_pairs():
    # Distinct keys (a shuffled range) make the key -> value mapping unique,
    # so the sorted payload is fully determined.
    num_items = 256
    rng = np.random.default_rng(13)
    h_in_keys = rng.permutation(num_items).astype(BFLOAT16)
    h_in_values = np.arange(num_items, dtype=np.uint32)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_values = DeviceArray.from_numpy(h_in_values)
    d_out_keys = DeviceArray.empty(num_items, BFLOAT16)
    d_out_values = DeviceArray.empty(num_items, np.uint32)

    cuda.compute.radix_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=d_in_values,
        d_out_values=d_out_values,
        order=SortOrder.ASCENDING,
        num_items=num_items,
    )

    argsort = np.argsort(h_in_keys.astype(np.float32))
    np.testing.assert_array_equal(d_out_keys.copy_to_host(), h_in_keys[argsort])
    np.testing.assert_array_equal(d_out_values.copy_to_host(), h_in_values[argsort])


def test_merge_sort_keys():
    num_items = 1000
    h_in_keys = random_bfloat16(num_items, seed=17)
    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(num_items, BFLOAT16)

    cuda.compute.merge_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        num_items=num_items,
        op=OpKind.LESS,
    )

    np.testing.assert_array_equal(d_out_keys.copy_to_host(), np.sort(h_in_keys))


def test_unary_transform_negate():
    num_items = 1000
    h_input = random_bfloat16(num_items, seed=23)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(num_items, BFLOAT16)

    cuda.compute.unary_transform(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.NEGATE,
        num_items=num_items,
    )

    np.testing.assert_array_equal(d_output.copy_to_host(), -h_input)


@pytest.mark.serialization
def test_reduce_serialize_deserialize_round_trip():
    num_items = 200
    h_input = np.full(num_items, 0.5, dtype=BFLOAT16)
    h_init = np.array([1.0], dtype=BFLOAT16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, BFLOAT16)

    reducer = make_reduce_into(
        d_in=d_input, d_out=d_output, op=OpKind.PLUS, h_init=h_init
    )
    blob = serialize(reducer)
    assert len(blob) > 0

    loaded = deserialize(blob)
    bytes_needed = loaded(
        temp_storage=None,
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    loaded(
        temp_storage=tmp,
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    assert float(d_output.copy_to_host()[0]) == 101.0


def test_python_callable_op_raises():
    # numba cannot compile Python callables for bfloat16; the error should be
    # a clear TypeError, not a cryptic numba failure.
    num_items = 100
    d_input = DeviceArray.from_numpy(np.zeros(num_items, dtype=BFLOAT16))
    d_output = DeviceArray.empty(1, BFLOAT16)
    h_init = np.array([0.0], dtype=BFLOAT16)

    with pytest.raises(TypeError, match="bfloat16 is not supported"):
        cuda.compute.reduce_into(
            d_in=d_input,
            d_out=d_output,
            num_items=num_items,
            op=lambda a, b: a + b,
            h_init=h_init,
        )
