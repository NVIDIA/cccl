# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for emulated floating point (``cuda::experimental::fpemu``) support.

Arrays use the explicit ``cuda.compute.fp64emu_{high,mid,low}`` dtypes; they
have the layout of float64, so data is moved in and out with ``.view``. Only well-known ``OpKind`` operators are used: they
resolve to ``cuda::std::plus<>`` etc., which use fpemu's own operators.
Operands are integer-valued, so every partial result is exactly representable
and the output is exact for any reduction order.
"""

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import OpKind
from cuda.compute import types as compute_types

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    not USING_V2, reason="emulated floating point requires the v2 (HostJIT) backend"
)

ACCURACIES = ["high", "mid", "low"]


def fpemu_dtype(accuracy):
    return getattr(cuda.compute, f"fp64emu_{accuracy}")


def _td(accuracy):
    return getattr(compute_types, f"fp64emu_{accuracy}")


@pytest.mark.parametrize("accuracy", ACCURACIES)
def test_type_descriptor(accuracy):
    td = _td(accuracy)
    assert td.size == 8
    assert td.alignment == 8
    assert td.dtype == fpemu_dtype(accuracy)
    assert compute_types.from_numpy_dtype(td.dtype) is td
    # Distinct from float64 and from each other.
    assert td.dtype != np.dtype("float64")
    assert len({fpemu_dtype(a) for a in ACCURACIES}) == 3


def test_dtype_numpy_behavior():
    dt = cuda.compute.fp64emu_mid
    assert dt.itemsize == 8
    # float scalars assign naturally, and .view round-trips with float64.
    h = np.array([1.5, 2.5], dtype=dt)
    assert (h.view(np.float64) == [1.5, 2.5]).all()
    f = np.arange(4, dtype=np.float64)
    assert np.shares_memory(f, f.view(dt))


@pytest.mark.parametrize("accuracy", ACCURACIES)
def test_reduce_sum(accuracy):
    dt = fpemu_dtype(accuracy)
    num_items = 1000
    h_input = np.arange(num_items, dtype=np.float64) % 17
    d_input = DeviceArray.from_numpy(h_input.view(dt))
    d_output = DeviceArray.empty(1, dt)
    h_init = np.array([3.0], dtype=dt)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    assert d_output.copy_to_host().view(np.float64)[0] == h_input.sum() + 3.0


@pytest.mark.parametrize(
    "op,reference", [(OpKind.MINIMUM, np.min), (OpKind.MAXIMUM, np.max)]
)
def test_reduce_min_max(op, reference):
    dt = cuda.compute.fp64emu_high
    num_items = 513
    rng = np.random.default_rng(0)
    h_input = rng.integers(-1000, 1000, num_items).astype(np.float64)
    d_input = DeviceArray.from_numpy(h_input.view(dt))
    d_output = DeviceArray.empty(1, dt)
    h_init = np.array([h_input[0]], dtype=dt)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=num_items,
        op=op,
        h_init=h_init,
    )

    assert d_output.copy_to_host().view(np.float64)[0] == reference(h_input)


def test_inclusive_scan_sum():
    dt = cuda.compute.fp64emu_high
    num_items = 777
    h_input = (np.arange(num_items) % 5).astype(np.float64)
    d_input = DeviceArray.from_numpy(h_input.view(dt))
    d_output = DeviceArray.empty(num_items, dt)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=None,
        num_items=num_items,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host().view(np.float64), np.cumsum(h_input)
    )


def test_binary_transform_plus():
    dt = cuda.compute.fp64emu_high
    num_items = 300
    h_a = np.arange(num_items, dtype=np.float64)
    h_b = np.full(num_items, 2.0)
    d_a = DeviceArray.from_numpy(h_a.view(dt))
    d_b = DeviceArray.from_numpy(h_b.view(dt))
    d_out = DeviceArray.empty(num_items, dt)

    cuda.compute.binary_transform(
        d_in1=d_a,
        d_in2=d_b,
        d_out=d_out,
        op=OpKind.PLUS,
        num_items=num_items,
    )

    np.testing.assert_array_equal(d_out.copy_to_host().view(np.float64), h_a + h_b)


def test_merge_sort_less():
    dt = cuda.compute.fp64emu_high
    num_items = 400
    rng = np.random.default_rng(1)
    h_keys = rng.permutation(num_items).astype(np.float64)
    d_keys = DeviceArray.from_numpy(h_keys.view(dt))
    d_out = DeviceArray.empty(num_items, dt)

    cuda.compute.merge_sort(
        d_in_keys=d_keys,
        d_in_values=None,
        d_out_keys=d_out,
        d_out_values=None,
        op=OpKind.LESS,
        num_items=num_items,
    )

    np.testing.assert_array_equal(
        d_out.copy_to_host().view(np.float64), np.sort(h_keys)
    )


def test_python_callable_op_rejected():
    dt = cuda.compute.fp64emu_high
    d_input = DeviceArray.from_numpy(np.arange(4, dtype=np.float64).view(dt))
    d_output = DeviceArray.empty(1, dt)

    def add(a, b):
        return a + b

    with pytest.raises(TypeError, match="fpemu"):
        cuda.compute.reduce_into(
            d_in=d_input,
            d_out=d_output,
            num_items=4,
            op=add,
            h_init=np.array([0.0], dtype=dt),
        )
