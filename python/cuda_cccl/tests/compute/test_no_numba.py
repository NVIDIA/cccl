# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import (
    CacheModifiedInputIterator,
    ConstantIterator,
    CountingIterator,
    DiscardIterator,
    OpKind,
    PermutationIterator,
    ReverseIterator,
    ShuffleIterator,
    SortOrder,
    TransformIterator,
    TransformOutputIterator,
    ZipIterator,
)
from cuda.compute._cpp_compile import compile_cpp_op_code
from cuda.compute.op import RawOp
from cuda.compute.types import int16 as cccl_int16
from cuda.compute.types import int32 as cccl_int32

# These tests define the minimal-extra integration contract. They intentionally
# use small fixed inputs and avoid the Python-callable operator path.
pytestmark = pytest.mark.no_numba


def _raw_op(source: str, name: str) -> RawOp:
    return RawOp(ltoir=compile_cpp_op_code(source), name=name)


def _raw_even_i32_op() -> RawOp:
    source = """
extern "C" __device__ void no_numba_even_i32(void* x, void* result) {
    int value = *static_cast<int*>(x);
    *static_cast<bool*>(result) = (value % 2) == 0;
}
"""
    return _raw_op(source, "no_numba_even_i32")


def _raw_less_than_i32_op(name: str, threshold: int) -> RawOp:
    source = f"""
extern "C" __device__ void {name}(void* x, void* result) {{
    int value = *static_cast<int*>(x);
    *static_cast<unsigned char*>(result) = value < {threshold} ? 1 : 0;
}}
"""
    return _raw_op(source, name)


def _raw_plus_i64_op() -> RawOp:
    source = """
extern "C" __device__ void no_numba_plus_i64(
    void* lhs,
    void* rhs,
    void* result
) {
    *static_cast<long long*>(result) =
        *static_cast<long long*>(lhs) + *static_cast<long long*>(rhs);
}
"""
    return _raw_op(source, "no_numba_plus_i64")


def _raw_square_i32_op() -> RawOp:
    source = """
extern "C" __device__ void no_numba_square_i32(void* x, void* result) {
    int value = *static_cast<int*>(x);
    *static_cast<int*>(result) = value * value;
}
"""
    return _raw_op(source, "no_numba_square_i32")


def _raw_zip_sum_i32_op() -> RawOp:
    source = """
struct Zip2I32 {
    int field_0;
    int field_1;
};

extern "C" __device__ void no_numba_zip_sum_i32(void* x, void* result) {
    auto values = static_cast<Zip2I32*>(x);
    *static_cast<int*>(result) = values->field_0 + values->field_1;
}
"""
    return _raw_op(source, "no_numba_zip_sum_i32")


def _raw_negate_i16_op() -> RawOp:
    source = """
extern "C" __device__ void no_numba_negate_i16(void* x, void* result) {
    *static_cast<short*>(result) = -*static_cast<short*>(x);
}
"""
    return _raw_op(source, "no_numba_negate_i16")


def test_import_numba_raises():
    with pytest.raises(
        ImportError, match="This test is marked 'no_numba' but attempted to import it"
    ):
        import numba.cuda  # noqa: F401


def test_reduce_well_known_plus():
    h_input = np.arange(1, 14, dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, np.int32)
    h_init = np.array([5], dtype=np.int32)

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        num_items=h_input.size,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    assert d_output.copy_to_host()[0] == np.sum(h_input, initial=h_init[0])


def test_exclusive_scan_well_known_plus():
    h_input = np.asarray([2, 4, 6, 8, 10, 12], dtype=np.uint16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)
    h_init = np.array([1], dtype=np.uint16)

    cuda.compute.exclusive_scan(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=h_init,
        num_items=h_input.size,
    )

    expected = np.asarray([1, 3, 7, 13, 21, 31], dtype=np.uint16)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def test_binary_transform_well_known_plus():
    h_lhs = np.asarray([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    h_rhs = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    d_lhs = DeviceArray.from_numpy(h_lhs)
    d_rhs = DeviceArray.from_numpy(h_rhs)
    d_output = DeviceArray.empty(h_lhs.shape, h_lhs.dtype)

    cuda.compute.binary_transform(
        d_in1=d_lhs,
        d_in2=d_rhs,
        d_out=d_output,
        op=OpKind.PLUS,
        num_items=h_lhs.size,
    )

    np.testing.assert_allclose(d_output.copy_to_host(), h_lhs + h_rhs)


def test_unary_transform_well_known_negate():
    h_input = np.asarray([-4, -2, 0, 2, 4], dtype=np.int8)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.unary_transform(
        d_in=d_input,
        d_out=d_output,
        op=OpKind.NEGATE,
        num_items=h_input.size,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host(), np.asarray([4, 2, 0, -2, -4])
    )


@pytest.mark.parametrize(
    "search, side",
    [
        (cuda.compute.lower_bound, "left"),
        (cuda.compute.upper_bound, "right"),
    ],
)
def test_binary_search_explicit_opkind_less(search, side):
    h_data = np.asarray([1, 3, 3, 7, 9, 11], dtype=np.int64)
    h_values = np.asarray([0, 3, 4, 10, 12], dtype=np.int64)
    d_out = DeviceArray.empty(h_values.shape, np.uintp)

    search(
        d_data=DeviceArray.from_numpy(h_data),
        num_items=h_data.size,
        d_values=DeviceArray.from_numpy(h_values),
        num_values=h_values.size,
        d_out=d_out,
        comp=OpKind.LESS,
    )

    expected = np.searchsorted(h_data, h_values, side=side).astype(np.uintp)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_segmented_reduce_well_known_plus(monkeypatch):
    monkeypatch.setattr(cuda.compute._cccl_interop, "_check_sass", False)

    h_input = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
    h_starts = np.asarray([0, 3, 5], dtype=np.int32)
    h_ends = np.asarray([3, 5, 8], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_starts = DeviceArray.from_numpy(h_starts)
    d_ends = DeviceArray.from_numpy(h_ends)
    d_output = DeviceArray.empty(3, np.uint32)
    h_init = np.array([0], dtype=np.uint32)

    cuda.compute.segmented_reduce(
        d_in=d_input,
        d_out=d_output,
        num_segments=3,
        start_offsets_in=d_starts,
        end_offsets_in=d_ends,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    np.testing.assert_array_equal(d_output.copy_to_host(), np.asarray([6, 9, 21]))


def test_merge_sort_well_known_less():
    h_input = np.asarray([3.5, -1.0, 2.25, 2.0, 7.0], dtype=np.float64)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.merge_sort(
        d_in_keys=d_input,
        d_in_values=None,
        d_out_keys=d_output,
        d_out_values=None,
        num_items=h_input.size,
        op=OpKind.LESS,
    )

    np.testing.assert_array_equal(d_output.copy_to_host(), np.sort(h_input))


def test_radix_sort_key_value_pairs():
    h_keys = np.asarray([4, -2, 7, 1, -2, 0], dtype=np.int16)
    h_values = np.asarray([40, 20, 70, 10, 21, 0], dtype=np.uint8)
    d_out_keys = DeviceArray.empty(h_keys.shape, h_keys.dtype)
    d_out_values = DeviceArray.empty(h_values.shape, h_values.dtype)

    cuda.compute.radix_sort(
        d_in_keys=DeviceArray.from_numpy(h_keys),
        d_out_keys=d_out_keys,
        d_in_values=DeviceArray.from_numpy(h_values),
        d_out_values=d_out_values,
        num_items=h_keys.size,
        order=SortOrder.ASCENDING,
    )

    order = np.argsort(h_keys, stable=True)
    np.testing.assert_array_equal(d_out_keys.copy_to_host(), h_keys[order])
    np.testing.assert_array_equal(d_out_values.copy_to_host(), h_values[order])


def test_segmented_sort_keys():
    h_keys = np.asarray([3, 1, 2, 9, 7, 8, 6, 5], dtype=np.uint64)
    h_offsets = np.asarray([0, 3, 6, 8], dtype=np.int64)
    d_output = DeviceArray.empty(h_keys.shape, h_keys.dtype)

    cuda.compute.segmented_sort(
        d_in_keys=DeviceArray.from_numpy(h_keys),
        d_out_keys=d_output,
        d_in_values=None,
        d_out_values=None,
        num_items=h_keys.size,
        num_segments=h_offsets.size - 1,
        start_offsets_in=DeviceArray.from_numpy(h_offsets[:-1]),
        end_offsets_in=DeviceArray.from_numpy(h_offsets[1:]),
        order=SortOrder.ASCENDING,
    )

    expected = np.asarray([1, 2, 3, 7, 8, 9, 5, 6], dtype=np.uint64)
    np.testing.assert_array_equal(d_output.copy_to_host(), expected)


def test_unique_by_key_well_known_equal_to(monkeypatch):
    cc_major, _ = cuda.compute._cccl_interop.CudaDevice().compute_capability
    if cc_major >= 9:
        monkeypatch.setattr(cuda.compute._cccl_interop, "_check_sass", False)

    h_keys = np.asarray([1, 1, 2, 2, 2, 3, 4, 4], dtype=np.int16)
    h_values = np.asarray([10, 11, 20, 21, 22, 30, 40, 41], dtype=np.int8)
    d_keys = DeviceArray.from_numpy(h_keys)
    d_values = DeviceArray.from_numpy(h_values)
    d_out_keys = DeviceArray.empty(h_keys.shape, h_keys.dtype)
    d_out_values = DeviceArray.empty(h_values.shape, h_values.dtype)
    d_num_selected = DeviceArray.empty(1, np.int64)

    cuda.compute.unique_by_key(
        d_in_keys=d_keys,
        d_in_items=d_values,
        d_out_keys=d_out_keys,
        d_out_items=d_out_values,
        d_out_num_selected=d_num_selected,
        op=OpKind.EQUAL_TO,
        num_items=h_keys.size,
    )

    num_selected = int(d_num_selected.copy_to_host()[0])
    np.testing.assert_array_equal(
        d_out_keys.copy_to_host()[:num_selected], [1, 2, 3, 4]
    )
    np.testing.assert_array_equal(
        d_out_values.copy_to_host()[:num_selected], [10, 20, 30, 40]
    )


def test_histogram_even_small_range():
    h_samples = np.asarray([0.5, 1.5, 2.5, 2.75, 3.0, 3.5], dtype=np.float32)
    d_histogram = DeviceArray.empty(4, np.int32)

    cuda.compute.histogram_even(
        d_samples=DeviceArray.from_numpy(h_samples),
        d_histogram=d_histogram,
        num_output_levels=5,
        lower_level=np.float32(0.0),
        upper_level=np.float32(4.0),
        num_samples=h_samples.size,
    )

    expected, _ = np.histogram(h_samples, bins=4, range=(0.0, 4.0))
    np.testing.assert_array_equal(d_histogram.copy_to_host(), expected.astype(np.int32))


def test_select_raw_op():
    h_input = np.arange(12, dtype=np.int32)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)
    d_num_selected = DeviceArray.empty(1, np.uint64)

    cuda.compute.select(
        d_in=DeviceArray.from_numpy(h_input),
        d_out=d_output,
        d_num_selected_out=d_num_selected,
        cond=_raw_even_i32_op(),
        num_items=h_input.size,
    )

    num_selected = int(d_num_selected.copy_to_host()[0])
    np.testing.assert_array_equal(d_output.copy_to_host()[:num_selected], h_input[::2])


def test_three_way_partition_raw_op():
    h_input = np.arange(12, dtype=np.int32)
    d_first = DeviceArray.empty(h_input.shape, h_input.dtype)
    d_second = DeviceArray.empty(h_input.shape, h_input.dtype)
    d_unselected = DeviceArray.empty(h_input.shape, h_input.dtype)
    d_num_selected = DeviceArray.empty(2, np.uint64)

    cuda.compute.three_way_partition(
        d_in=DeviceArray.from_numpy(h_input),
        d_first_part_out=d_first,
        d_second_part_out=d_second,
        d_unselected_out=d_unselected,
        d_num_selected_out=d_num_selected,
        select_first_part_op=_raw_less_than_i32_op("no_numba_less_than_4_i32", 4),
        select_second_part_op=_raw_less_than_i32_op("no_numba_less_than_8_i32", 8),
        num_items=h_input.size,
    )

    selected = d_num_selected.copy_to_host()
    first_count = int(selected[0])
    second_count = int(selected[1])
    unselected_count = h_input.size - first_count - second_count

    np.testing.assert_array_equal(d_first.copy_to_host()[:first_count], h_input[:4])
    np.testing.assert_array_equal(d_second.copy_to_host()[:second_count], h_input[4:8])
    np.testing.assert_array_equal(
        d_unselected.copy_to_host()[:unselected_count], h_input[8:]
    )


def test_raw_op_reduce():
    h_input = np.asarray([10, 20, 30, 40], dtype=np.int64)
    d_output = DeviceArray.empty(1, np.int64)

    cuda.compute.reduce_into(
        d_in=DeviceArray.from_numpy(h_input),
        d_out=d_output,
        num_items=h_input.size,
        op=_raw_plus_i64_op(),
        h_init=np.array([5], dtype=np.int64),
    )

    assert d_output.copy_to_host()[0] == 105


def test_stream_argument(cuda_stream):
    h_lhs = np.asarray([2, 4, 6, 8, 10], dtype=np.int32)
    h_rhs = np.asarray([1, 3, 5, 7, 9], dtype=np.int32)
    d_lhs = DeviceArray.from_numpy(h_lhs, stream=cuda_stream)
    d_rhs = DeviceArray.from_numpy(h_rhs, stream=cuda_stream)
    d_output = DeviceArray.empty(h_lhs.shape, h_lhs.dtype, stream=cuda_stream)

    cuda.compute.binary_transform(
        d_in1=d_lhs,
        d_in2=d_rhs,
        d_out=d_output,
        op=OpKind.PLUS,
        num_items=h_lhs.size,
        stream=cuda_stream,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host(stream=cuda_stream),
        np.asarray([3, 7, 11, 15, 19]),
    )


def test_counting_iterator_reduce():
    d_output = DeviceArray.empty(1, np.int32)

    cuda.compute.reduce_into(
        d_in=CountingIterator(np.int32(3)),
        d_out=d_output,
        num_items=8,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.int32),
    )

    assert d_output.copy_to_host()[0] == 52


def test_constant_iterator_reduce():
    d_output = DeviceArray.empty(1, np.float32)

    cuda.compute.reduce_into(
        d_in=ConstantIterator(np.float32(1.5)),
        d_out=d_output,
        num_items=8,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.float32),
    )

    np.testing.assert_allclose(d_output.copy_to_host()[0], np.float32(12.0))


def test_cache_modified_input_iterator_reduce():
    h_input = np.asarray([2, 4, 6, 8, 10], dtype=np.uint16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, np.uint16)
    iterator = CacheModifiedInputIterator(d_input, modifier="stream")

    cuda.compute.reduce_into(
        d_in=iterator,
        d_out=d_output,
        num_items=h_input.size,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.uint16),
    )

    assert d_output.copy_to_host()[0] == 30


def test_reverse_input_iterator_scan():
    h_input = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.inclusive_scan(
        d_in=ReverseIterator(d_input),
        d_out=d_output,
        op=OpKind.PLUS,
        init_value=np.array([0], dtype=np.int32),
        num_items=h_input.size,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host(), np.asarray([5, 9, 12, 14, 15])
    )


def test_reverse_output_iterator_scan():
    h_input = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(h_input.shape, h_input.dtype)

    cuda.compute.inclusive_scan(
        d_in=d_input,
        d_out=ReverseIterator(d_output),
        op=OpKind.PLUS,
        init_value=np.array([0], dtype=np.int32),
        num_items=h_input.size,
    )

    np.testing.assert_array_equal(
        d_output.copy_to_host(), np.asarray([15, 10, 6, 3, 1])
    )


def test_permutation_iterator_reduce():
    h_values = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
    h_indices = np.asarray([4, 2, 5, 1], dtype=np.int32)
    d_values = DeviceArray.from_numpy(h_values)
    d_indices = DeviceArray.from_numpy(h_indices)
    d_output = DeviceArray.empty(1, np.int64)

    cuda.compute.reduce_into(
        d_in=PermutationIterator(d_values, d_indices),
        d_out=d_output,
        num_items=h_indices.size,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.int64),
    )

    assert d_output.copy_to_host()[0] == 160


def test_transform_iterator_reduce():
    d_output = DeviceArray.empty(1, np.int32)
    iterator = TransformIterator(
        CountingIterator(np.int32(1)), _raw_square_i32_op(), value_type=cccl_int32
    )

    cuda.compute.reduce_into(
        d_in=iterator,
        d_out=d_output,
        num_items=6,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.int32),
    )

    assert d_output.copy_to_host()[0] == 91


def test_transform_output_iterator_reduce():
    h_input = np.asarray([1, 2, 3, 4], dtype=np.int16)
    d_input = DeviceArray.from_numpy(h_input)
    d_output = DeviceArray.empty(1, np.int16)
    output_iterator = TransformOutputIterator(
        d_output, _raw_negate_i16_op(), output_value_type=cccl_int16
    )

    cuda.compute.reduce_into(
        d_in=d_input,
        d_out=output_iterator,
        num_items=h_input.size,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.int16),
    )

    assert d_output.copy_to_host()[0] == -10


def test_zip_iterator_transform():
    h_lhs = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
    h_rhs = np.asarray([10, 20, 30, 40, 50], dtype=np.int32)
    d_lhs = DeviceArray.from_numpy(h_lhs)
    d_rhs = DeviceArray.from_numpy(h_rhs)
    d_output = DeviceArray.empty(h_lhs.shape, h_lhs.dtype)

    cuda.compute.unary_transform(
        d_in=ZipIterator(d_lhs, d_rhs),
        d_out=d_output,
        op=_raw_zip_sum_i32_op(),
        num_items=h_lhs.size,
    )

    np.testing.assert_array_equal(d_output.copy_to_host(), h_lhs + h_rhs)


def test_shuffle_iterator_transform():
    num_items = 17
    d_output = DeviceArray.empty(num_items, np.int64)

    cuda.compute.unary_transform(
        d_in=ShuffleIterator(num_items, seed=123),
        d_out=d_output,
        op=OpKind.IDENTITY,
        num_items=num_items,
    )

    result = d_output.copy_to_host()
    assert sorted(result.tolist()) == list(range(num_items))


def test_discard_iterator_transform():
    h_input = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
    h_reference = np.full_like(h_input, -1)
    d_input = DeviceArray.from_numpy(h_input)
    d_reference = DeviceArray.from_numpy(h_reference)

    cuda.compute.unary_transform(
        d_in=d_input,
        d_out=DiscardIterator(d_reference),
        op=OpKind.IDENTITY,
        num_items=h_input.size,
    )

    np.testing.assert_array_equal(
        d_reference.copy_to_host(), np.full(5, -1, dtype=np.int32)
    )
