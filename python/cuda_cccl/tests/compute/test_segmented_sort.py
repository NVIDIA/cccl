# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

import cupy as cp
import numba
import numpy as np
import pytest

import cuda.compute

DTYPE_LIST = [
    np.uint8,
    np.int16,
    np.uint32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
]


def get_mark_by_size(num_segments: int, segment_size: int):
    num_items = num_segments * segment_size
    return pytest.mark.large if num_items >= (1 << 20) else tuple()


NUM_SEGMENTS_AND_SEGMENT_SIZES = [
    (1, 1),
    (1, 2048),
    (13, 12),
    (1024, 1024),
    (2048, 1),
    (2048, 13),
]

DTYPE_SEGMENT_PARAMS = [
    pytest.param(dt, ns, ss, marks=get_mark_by_size(ns, ss))
    for dt in DTYPE_LIST
    for (ns, ss) in NUM_SEGMENTS_AND_SEGMENT_SIZES
]


def random_array(size, dtype, max_value=None) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        if max_value is None:
            max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        return np.random.uniform(low=-10.0, high=10.0, size=size).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def make_uniform_segments(
    num_segments: int, segment_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    start_offsets = np.arange(num_segments, dtype=np.int64) * segment_size
    end_offsets = start_offsets + segment_size
    return start_offsets, end_offsets


def host_segmented_sort(
    h_keys: np.ndarray,
    h_vals: np.ndarray | None,
    start_offsets: np.ndarray,
    end_offsets: np.ndarray,
    order: cuda.compute.SortOrder,
) -> Tuple[np.ndarray, np.ndarray | None]:
    assert start_offsets.shape == end_offsets.shape
    keys = h_keys.copy()
    vals = None if h_vals is None else h_vals.copy()

    for s, e in zip(start_offsets, end_offsets):
        if e <= s:
            continue
        if vals is None:
            if order is cuda.compute.SortOrder.DESCENDING:
                # stable descending
                signed_dtype = (
                    np.dtype(keys.dtype.name.replace("uint", "int"))
                    if np.issubdtype(keys.dtype, np.unsignedinteger)
                    else keys.dtype
                )
                idx = np.argsort(-keys[s:e].astype(signed_dtype), stable=True)
            else:
                idx = np.argsort(keys[s:e], stable=True)
            keys[s:e] = keys[s:e][idx]
        else:
            # build pairs for stable sort
            pairs = list(zip(keys[s:e], vals[s:e]))
            if order is cuda.compute.SortOrder.DESCENDING:
                pairs.sort(key=lambda kv: kv[0], reverse=True)
            else:
                pairs.sort(key=lambda kv: kv[0])
            ks, vs = zip(*pairs) if pairs else ([], [])
            keys[s:e] = np.array(ks, dtype=keys.dtype)
            vals[s:e] = np.array(vs, dtype=vals.dtype)

    return keys, vals


@pytest.mark.parametrize("dtype, num_segments, segment_size", DTYPE_SEGMENT_PARAMS)
def test_segmented_sort_keys(dtype, num_segments, segment_size, monkeypatch):
    # Disable SASS verification only for this test when dtype is int64
    if np.dtype(dtype) == np.dtype(np.int64):
        monkeypatch.setattr(
            cuda.compute._cccl_interop,
            "_check_sass",
            False,
        )
    order = cuda.compute.SortOrder.ASCENDING
    num_items = num_segments * segment_size

    h_in_keys = random_array(num_items, dtype, max_value=50)
    start_offsets, end_offsets = make_uniform_segments(num_segments, segment_size)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_out_keys = numba.cuda.to_device(np.empty_like(h_in_keys))

    cuda.compute.segmented_sort(
        d_in_keys,
        d_out_keys,
        None,
        None,
        num_items,
        num_segments,
        cp.asarray(start_offsets),
        cp.asarray(end_offsets),
        order,
    )

    h_out_keys = d_out_keys.copy_to_host()
    expected_keys, _ = host_segmented_sort(
        h_in_keys, None, start_offsets, end_offsets, order
    )

    np.testing.assert_array_equal(h_out_keys, expected_keys)


@pytest.mark.parametrize("dtype, num_segments, segment_size", DTYPE_SEGMENT_PARAMS)
def test_segmented_sort_pairs(dtype, num_segments, segment_size):
    order = cuda.compute.SortOrder.DESCENDING
    num_items = num_segments * segment_size

    h_in_keys = random_array(
        num_items, dtype, max_value=50 if np.isdtype(dtype, "integral") else None
    )
    h_in_vals = random_array(num_items, np.float32)

    start_offsets, end_offsets = make_uniform_segments(num_segments, segment_size)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_vals = numba.cuda.to_device(h_in_vals)
    d_out_keys = numba.cuda.to_device(np.empty_like(h_in_keys))
    d_out_vals = numba.cuda.to_device(np.empty_like(h_in_vals))

    cuda.compute.segmented_sort(
        d_in_keys,
        d_out_keys,
        d_in_vals,
        d_out_vals,
        num_items,
        num_segments,
        cp.asarray(start_offsets),
        cp.asarray(end_offsets),
        order,
    )

    h_out_keys = d_out_keys.copy_to_host()
    h_out_vals = d_out_vals.copy_to_host()

    expected_keys, expected_vals = host_segmented_sort(
        h_in_keys, h_in_vals, start_offsets, end_offsets, order
    )

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_vals, expected_vals)


@pytest.mark.parametrize("dtype, num_segments, segment_size", DTYPE_SEGMENT_PARAMS)
def test_segmented_sort_keys_double_buffer(dtype, num_segments, segment_size):
    order = cuda.compute.SortOrder.ASCENDING
    num_items = num_segments * segment_size

    h_in_keys = random_array(num_items, dtype, max_value=20)
    start_offsets, end_offsets = make_uniform_segments(num_segments, segment_size)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_tmp_keys = numba.cuda.to_device(np.empty_like(h_in_keys))
    keys_db = cuda.compute.DoubleBuffer(d_in_keys, d_tmp_keys)

    cuda.compute.segmented_sort(
        keys_db,
        None,
        None,
        None,
        num_items,
        num_segments,
        cp.asarray(start_offsets),
        cp.asarray(end_offsets),
        order,
    )

    h_out_keys = keys_db.current().copy_to_host()
    expected_keys, _ = host_segmented_sort(
        h_in_keys, None, start_offsets, end_offsets, order
    )
    np.testing.assert_array_equal(h_out_keys, expected_keys)


@pytest.mark.parametrize("dtype, num_segments, segment_size", DTYPE_SEGMENT_PARAMS)
def test_segmented_sort_pairs_double_buffer(dtype, num_segments, segment_size):
    order = cuda.compute.SortOrder.DESCENDING
    num_items = num_segments * segment_size

    h_in_keys = random_array(
        num_items, dtype, max_value=50 if np.isdtype(dtype, "integral") else None
    )
    h_in_vals = random_array(num_items, np.float32)

    start_offsets, end_offsets = make_uniform_segments(num_segments, segment_size)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_vals = numba.cuda.to_device(h_in_vals)
    d_tmp_keys = numba.cuda.to_device(np.empty_like(h_in_keys))
    d_tmp_vals = numba.cuda.to_device(np.empty_like(h_in_vals))

    keys_db = cuda.compute.DoubleBuffer(d_in_keys, d_tmp_keys)
    vals_db = cuda.compute.DoubleBuffer(d_in_vals, d_tmp_vals)

    cuda.compute.segmented_sort(
        keys_db,
        None,
        vals_db,
        None,
        num_items,
        num_segments,
        cp.asarray(start_offsets),
        cp.asarray(end_offsets),
        order,
    )

    h_out_keys = keys_db.current().copy_to_host()
    h_out_vals = vals_db.current().copy_to_host()

    expected_keys, expected_vals = host_segmented_sort(
        h_in_keys, h_in_vals, start_offsets, end_offsets, order
    )
    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_vals, expected_vals)


@pytest.mark.parametrize("num_segments", [20, 600])
def test_segmented_sort_variable_segment_sizes(num_segments):
    order = cuda.compute.SortOrder.ASCENDING
    base_pattern = [
        1,
        5,
        10,
        20,
        30,
        50,
        100,
        3,
        25,
        600,
        7,
        18,
        300,
        4,
        35,
        9,
        14,
        700,
        28,
        11,
    ]
    segment_sizes = []
    while len(segment_sizes) < num_segments:
        remaining = num_segments - len(segment_sizes)
        copy_count = min(remaining, len(base_pattern))
        segment_sizes.extend(base_pattern[:copy_count])

    start_offsets = np.zeros(num_segments, dtype=np.int64)
    end_offsets = np.zeros(num_segments, dtype=np.int64)
    current = 0
    for i, sz in enumerate(segment_sizes):
        start_offsets[i] = current
        current += sz
        end_offsets[i] = current
    num_items = current

    h_in_keys = random_array(num_items, np.int32, max_value=100)
    h_in_vals = random_array(num_items, np.float32)

    d_in_keys = numba.cuda.to_device(h_in_keys)
    d_in_vals = numba.cuda.to_device(h_in_vals)
    d_out_keys = numba.cuda.to_device(np.empty_like(h_in_keys))
    d_out_vals = numba.cuda.to_device(np.empty_like(h_in_vals))

    cuda.compute.segmented_sort(
        d_in_keys,
        d_out_keys,
        d_in_vals,
        d_out_vals,
        num_items,
        num_segments,
        cp.asarray(start_offsets),
        cp.asarray(end_offsets),
        order,
    )

    h_out_keys = d_out_keys.copy_to_host()
    h_out_vals = d_out_vals.copy_to_host()
    expected_keys, expected_vals = host_segmented_sort(
        h_in_keys, h_in_vals, start_offsets, end_offsets, order
    )

    np.testing.assert_array_equal(h_out_keys, expected_keys)
    np.testing.assert_array_equal(h_out_vals, expected_vals)
