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
from _utils.device_array import DeviceArray, get_compute_capability

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
from cuda.compute.op import RawOp

ml_dtypes = pytest.importorskip("ml_dtypes")

BFLOAT16 = np.dtype(ml_dtypes.bfloat16)


def random_bfloat16(size, low=-10.0, high=10.0, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=size).astype(BFLOAT16)


def make_bf16_predicate_op(source: str, name: str) -> RawOp:
    """Compile a C++ unary predicate on __nv_bfloat16 to LTO-IR and wrap it in
    a RawOp. Python callables cannot be used with bfloat16 (numba has no
    support), so predicates must be supplied as pre-compiled device code."""
    from cuda.core import Program, ProgramOptions

    from cuda.compute._cpp_compile import _get_include_paths

    cc_major, cc_minor = get_compute_capability()
    opts = ProgramOptions(
        arch=f"sm_{cc_major}{cc_minor}",
        relocatable_device_code=True,
        link_time_optimization=True,
        include_path=_get_include_paths(),
    )
    ltoir = Program(source, "c++", options=opts).compile("ltoir").code
    return RawOp(ltoir=ltoir, name=name)


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


def test_binary_transform_plus():
    # 0/1 inputs keep every sum exactly representable in bfloat16.
    num_items = 200
    rng = np.random.default_rng(31)
    h_in1 = rng.integers(0, 2, num_items).astype(BFLOAT16)
    h_in2 = rng.integers(0, 2, num_items).astype(BFLOAT16)
    d_in1 = DeviceArray.from_numpy(h_in1)
    d_in2 = DeviceArray.from_numpy(h_in2)
    d_out = DeviceArray.empty(num_items, BFLOAT16)

    cuda.compute.binary_transform(
        d_in1=d_in1,
        d_in2=d_in2,
        d_out=d_out,
        op=OpKind.PLUS,
        num_items=num_items,
    )

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in1 + h_in2)


def test_segmented_reduce_sum():
    # 8 segments of 25 x 0.5 each: per-segment sums (12.5) are exact.
    num_segments = 8
    segment_size = 25
    num_items = num_segments * segment_size
    h_input = np.full(num_items, 0.5, dtype=BFLOAT16)
    h_offsets = np.arange(0, num_items + 1, segment_size, dtype=np.int64)
    h_init = np.array([0.0], dtype=BFLOAT16)

    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(num_segments, BFLOAT16)
    d_start_offsets = DeviceArray.from_numpy(h_offsets[:-1])
    d_end_offsets = DeviceArray.from_numpy(h_offsets[1:])

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=num_segments,
        start_offsets_in=d_start_offsets,
        end_offsets_in=d_end_offsets,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host().astype(np.float32), np.full(num_segments, 12.5)
    )


def test_segmented_sort_keys():
    num_segments = 4
    segment_size = 50
    num_items = num_segments * segment_size
    h_in_keys = random_bfloat16(num_items, seed=37)
    h_offsets = np.arange(0, num_items + 1, segment_size, dtype=np.int64)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_out_keys = DeviceArray.empty(num_items, BFLOAT16)
    d_start_offsets = DeviceArray.from_numpy(h_offsets[:-1])
    d_end_offsets = DeviceArray.from_numpy(h_offsets[1:])

    cuda.compute.segmented_sort(
        d_in_keys=d_in_keys,
        d_out_keys=d_out_keys,
        d_in_values=None,
        d_out_values=None,
        num_items=num_items,
        num_segments=num_segments,
        start_offsets_in=d_start_offsets,
        end_offsets_in=d_end_offsets,
        order=SortOrder.ASCENDING,
    )

    expected = np.concatenate(
        [np.sort(h_in_keys[start : start + segment_size]) for start in h_offsets[:-1]]
    )
    np.testing.assert_array_equal(d_out_keys.copy_to_host(), expected)


def test_lower_and_upper_bound():
    # Integer-valued bfloat16 data keeps searchsorted semantics exact.
    h_data = np.repeat(np.arange(0, 50, dtype=np.float32), 2).astype(BFLOAT16)
    h_values = np.array([-1.0, 0.0, 3.5, 20.0, 49.0, 100.0], dtype=BFLOAT16)
    d_data = DeviceArray.from_numpy(h_data)
    d_values = DeviceArray.from_numpy(h_values)

    for algorithm, side in (
        (cuda.compute.lower_bound, "left"),
        (cuda.compute.upper_bound, "right"),
    ):
        d_out = DeviceArray.empty(len(h_values), np.uintp)
        algorithm(
            d_data=d_data,
            num_items=len(h_data),
            d_values=d_values,
            num_values=len(h_values),
            d_out=d_out,
        )
        expected = np.searchsorted(
            h_data.astype(np.float32), h_values.astype(np.float32), side=side
        ).astype(np.uintp)
        np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_unique_by_key():
    # Runs of 3 equal keys; the first item of each run is kept.
    num_runs = 10
    h_in_keys = np.repeat(np.arange(num_runs, dtype=np.float32), 3).astype(BFLOAT16)
    h_in_items = np.arange(h_in_keys.size, dtype=np.uint32)

    d_in_keys = DeviceArray.from_numpy(h_in_keys)
    d_in_items = DeviceArray.from_numpy(h_in_items)
    d_out_keys = DeviceArray.empty(num_runs, BFLOAT16)
    d_out_items = DeviceArray.empty(num_runs, np.uint32)
    d_out_num_selected = DeviceArray.from_numpy(np.zeros(1, dtype=np.int32))

    cuda.compute.unique_by_key(
        d_in_keys=d_in_keys,
        d_in_items=d_in_items,
        d_out_keys=d_out_keys,
        d_out_items=d_out_items,
        d_out_num_selected=d_out_num_selected,
        op=OpKind.EQUAL_TO,
        num_items=h_in_keys.size,
    )

    assert int(d_out_num_selected.copy_to_host()[0]) == num_runs
    np.testing.assert_array_equal(
        d_out_keys.copy_to_host(),
        np.arange(num_runs, dtype=np.float32).astype(BFLOAT16),
    )
    np.testing.assert_array_equal(d_out_items.copy_to_host(), h_in_items[::3])


def test_select_with_cpp_predicate():
    num_items = 200
    h_input = random_bfloat16(num_items, seed=41)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(num_items, BFLOAT16)
    d_num_selected = DeviceArray.from_numpy(np.zeros(2, dtype=np.uint64))

    greater_than_zero = make_bf16_predicate_op(
        """
        #include <cuda_bf16.h>
        extern "C" __device__ void greater_than_zero(void* x_void, void* out_void) {
          const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(x_void);
          *static_cast<bool*>(out_void) = *x > __float2bfloat16(0.0f);
        }
        """,
        "greater_than_zero",
    )

    cuda.compute.select(
        d_in=d_input,
        d_out=d_output,
        d_num_selected_out=d_num_selected,
        cond=greater_than_zero,
        num_items=num_items,
    )

    expected = h_input[h_input.astype(np.float32) > 0.0]
    num_selected = int(d_num_selected.copy_to_host()[0])
    assert num_selected == expected.size
    np.testing.assert_array_equal(d_output.copy_to_host()[:num_selected], expected)


def test_three_way_partition_with_cpp_predicates():
    num_items = 200
    h_input = random_bfloat16(num_items, seed=43)
    d_input = DeviceArray.from_numpy(h_input)
    d_first = DeviceArray.empty(num_items, BFLOAT16)
    d_second = DeviceArray.empty(num_items, BFLOAT16)
    d_unselected = DeviceArray.empty(num_items, BFLOAT16)
    d_num_selected = DeviceArray.from_numpy(np.zeros(2, dtype=np.int32))

    less_than_minus_two = make_bf16_predicate_op(
        """
        #include <cuda_bf16.h>
        extern "C" __device__ void less_than_minus_two(void* x_void, void* out_void) {
          const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(x_void);
          *static_cast<bool*>(out_void) = *x < __float2bfloat16(-2.0f);
        }
        """,
        "less_than_minus_two",
    )
    greater_than_two = make_bf16_predicate_op(
        """
        #include <cuda_bf16.h>
        extern "C" __device__ void greater_than_two(void* x_void, void* out_void) {
          const __nv_bfloat16* x = static_cast<const __nv_bfloat16*>(x_void);
          *static_cast<bool*>(out_void) = *x > __float2bfloat16(2.0f);
        }
        """,
        "greater_than_two",
    )

    cuda.compute.three_way_partition(
        d_in=d_input,
        d_first_part_out=d_first,
        d_second_part_out=d_second,
        d_unselected_out=d_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=less_than_minus_two,
        select_second_part_op=greater_than_two,
        num_items=num_items,
    )

    values = h_input.astype(np.float32)
    expected_first = h_input[values < -2.0]
    expected_second = h_input[(values >= -2.0) & (values > 2.0)]
    expected_unselected = h_input[(values >= -2.0) & (values <= 2.0)]

    num_first, num_second = (int(n) for n in d_num_selected.copy_to_host())
    assert num_first == expected_first.size
    assert num_second == expected_second.size
    np.testing.assert_array_equal(d_first.copy_to_host()[:num_first], expected_first)
    np.testing.assert_array_equal(d_second.copy_to_host()[:num_second], expected_second)
    np.testing.assert_array_equal(
        d_unselected.copy_to_host()[: num_items - num_first - num_second],
        expected_unselected,
    )


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="CUB's HistogramEven bins bfloat16 samples incorrectly, so "
    "make_histogram_even rejects bfloat16; "
    "see https://github.com/NVIDIA/cccl/issues/10940",
)
def test_histogram_even():
    num_samples = 1000
    num_bins = 8
    h_samples = random_bfloat16(num_samples, low=0.0, high=8.0, seed=29)
    d_samples = DeviceArray.from_numpy(h_samples)
    d_histogram = DeviceArray.from_numpy(np.zeros(num_bins, dtype=np.uint32))

    cuda.compute.histogram_even(
        d_samples=d_samples,
        d_histogram=d_histogram,
        num_output_levels=num_bins + 1,
        lower_level=np.array([0.0], dtype=BFLOAT16)[0],
        upper_level=np.array([8.0], dtype=BFLOAT16)[0],
        num_samples=num_samples,
    )

    # histogram_even uses half-open [lower, upper) bins; samples that rounded
    # up to exactly 8.0 in bfloat16 are out of range.
    samples = h_samples.astype(np.float64)
    expected = np.histogram(samples[samples < 8.0], bins=num_bins, range=(0, 8))[0]
    np.testing.assert_array_equal(d_histogram.copy_to_host(), expected)


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
