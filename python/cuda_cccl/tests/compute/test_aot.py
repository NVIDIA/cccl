# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for AoT (ahead-of-time) compilation: save/load and disk cache."""

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    Determinism,
    OpKind,
    make_binary_transform,
    make_lower_bound,
    make_merge_sort,
    make_radix_sort,
    make_reduce_into,
    make_segmented_reduce,
    make_unary_transform,
    make_upper_bound,
)
from cuda.compute._binary_format import MAGIC
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._sort._sort_common import SortOrder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_reducer(reducer, d_in, d_out, op, h_init):
    """Execute a reducer (normal or loaded) and return host result."""
    n = len(d_in)
    tmp_bytes = reducer(None, d_in, d_out, op, n, h_init)
    tmp = TempStorageBuffer(tmp_bytes)
    reducer(tmp, d_in, d_out, op, n, h_init)
    return cp.asnumpy(d_out)[0]


def _run_sorter(sorter, d_in_keys, d_in_items, d_out_keys, d_out_items, op, n):
    """Execute a sorter (normal or loaded) and return host keys."""
    tmp_bytes = sorter(None, d_in_keys, d_in_items, d_out_keys, d_out_items, op, n)
    tmp = TempStorageBuffer(tmp_bytes)
    sorter(tmp, d_in_keys, d_in_items, d_out_keys, d_out_items, op, n)
    return cp.asnumpy(d_out_keys)


# ---------------------------------------------------------------------------
# Fixture: disk cache
# ---------------------------------------------------------------------------


@pytest.fixture
def disk_cache(tmp_path):
    """Enable the on-disk cache in tmp_path for one test, then tear down."""
    cuda.compute.set_cache_dir(tmp_path)
    cuda.compute.clear_all_caches()
    yield tmp_path
    cuda.compute.set_cache_dir(None)
    cuda.compute.clear_all_caches()


# ---------------------------------------------------------------------------
# Reduce: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def test_reduce_save_load_sum_float32(tmp_path):
    """Round-trip: save a float32 sum reducer and reload it."""
    h = np.array([1.0], dtype=np.float32)
    d_in = cp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_out = cp.zeros(1, dtype=np.float32)

    reducer = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    expected = _run_reducer(reducer, d_in, d_out, OpKind.PLUS, h)

    path = tmp_path / "sum_f32.alg"
    reducer.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(1, dtype=np.float32)
    result = _run_reducer(loaded, d_in, d_out2, OpKind.PLUS, h)

    np.testing.assert_allclose(result, expected)


def test_reduce_save_load_sum_int32(tmp_path):
    h = np.array([0], dtype=np.int32)
    d_in = cp.arange(10, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    expected = _run_reducer(reducer, d_in, d_out, OpKind.PLUS, h)

    path = tmp_path / "sum_i32.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(1, dtype=np.int32)
    result = _run_reducer(loaded, d_in, d_out2, OpKind.PLUS, h)
    assert result == expected


def test_reduce_save_load_min_float64(tmp_path):
    h = np.array([float("inf")], dtype=np.float64)
    d_in = cp.array([3.0, 1.0, 4.0, 1.5], dtype=np.float64)
    d_out = cp.zeros(1, dtype=np.float64)

    reducer = make_reduce_into(d_in, d_out, OpKind.MINIMUM, h)
    expected = _run_reducer(reducer, d_in, d_out, OpKind.MINIMUM, h)

    path = tmp_path / "min_f64.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(1, dtype=np.float64)
    result = _run_reducer(loaded, d_in, d_out2, OpKind.MINIMUM, h)
    np.testing.assert_allclose(result, expected)


def test_reduce_save_load_max_int64(tmp_path):
    h = np.array([np.iinfo(np.int64).min], dtype=np.int64)
    d_in = cp.array([-5, 100, 0, 42], dtype=np.int64)
    d_out = cp.zeros(1, dtype=np.int64)

    reducer = make_reduce_into(d_in, d_out, OpKind.MAXIMUM, h)
    expected = _run_reducer(reducer, d_in, d_out, OpKind.MAXIMUM, h)

    path = tmp_path / "max_i64.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(1, dtype=np.int64)
    result = _run_reducer(loaded, d_in, d_out2, OpKind.MAXIMUM, h)
    assert result == expected


def test_reduce_save_load_nondeterministic(tmp_path):
    """Nondeterministic (atomic) reduce round-trip."""
    h = np.array([0.0], dtype=np.float32)
    d_in = cp.ones(16, dtype=np.float32)
    d_out = cp.zeros(1, dtype=np.float32)

    reducer = make_reduce_into(
        d_in, d_out, OpKind.PLUS, h, determinism=Determinism.NOT_GUARANTEED
    )
    expected = _run_reducer(reducer, d_in, d_out, OpKind.PLUS, h)

    path = tmp_path / "nondet.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(1, dtype=np.float32)
    result = _run_reducer(loaded, d_in, d_out2, OpKind.PLUS, h)
    np.testing.assert_allclose(result, expected)


def test_reduce_loaded_reducer_reusable(tmp_path):
    """A loaded reducer can be called multiple times."""
    h = np.array([0], dtype=np.int32)
    d_in = cp.arange(8, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    path = tmp_path / "reuse.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(1, dtype=np.int32)
        result = _run_reducer(loaded, d_in, d_out2, OpKind.PLUS, h)
        assert result == 28  # sum(0..7)


def test_load_algorithm_unknown_tag_raises(tmp_path):
    """load_algorithm raises ValueError for unrecognised algorithm tags."""
    from cuda.compute._binary_format import write_cclb

    path = tmp_path / "bad.alg"
    write_cclb(path, "nonexistent_algo", {}, b"\x00")

    with pytest.raises(ValueError, match="nonexistent_algo"):
        cuda.compute.load_algorithm(path)


# ---------------------------------------------------------------------------
# Merge sort: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def test_merge_sort_save_load_keys_only_int32(tmp_path):
    """Round-trip: save a key-only int32 sorter and reload it."""
    n = 8
    data = np.array([5, 3, 8, 1, 4, 7, 2, 6], dtype=np.int32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.int32)

    sorter = make_merge_sort(d_in, None, d_out, None, OpKind.LESS)
    expected = _run_sorter(sorter, d_in, None, d_out, None, OpKind.LESS, n)

    path = tmp_path / "sort_i32.alg"
    sorter.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(n, dtype=np.int32)
    result = _run_sorter(loaded, d_in, None, d_out2, None, OpKind.LESS, n)

    np.testing.assert_array_equal(result, expected)


def test_merge_sort_save_load_keys_only_float32(tmp_path):
    """Round-trip with float32 keys."""
    n = 6
    data = np.array([3.1, 1.5, 4.2, 1.1, 5.9, 2.6], dtype=np.float32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.float32)

    sorter = make_merge_sort(d_in, None, d_out, None, OpKind.LESS)
    expected = _run_sorter(sorter, d_in, None, d_out, None, OpKind.LESS, n)

    path = tmp_path / "sort_f32.alg"
    sorter.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(n, dtype=np.float32)
    result = _run_sorter(loaded, d_in, None, d_out2, None, OpKind.LESS, n)
    np.testing.assert_array_almost_equal(result, expected)


def test_merge_sort_loaded_sorter_reusable(tmp_path):
    """A loaded sorter can be called multiple times."""
    n = 5
    data = np.array([9, 4, 7, 2, 5], dtype=np.int32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.int32)

    sorter = make_merge_sort(d_in, None, d_out, None, OpKind.LESS)
    path = tmp_path / "reuse_sort.alg"
    sorter.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(n, dtype=np.int32)
        result = _run_sorter(loaded, d_in, None, d_out2, None, OpKind.LESS, n)
        np.testing.assert_array_equal(result, np.sort(data))


def test_merge_sort_save_load_descending(tmp_path):
    """Sorter with GREATER op (descending) round-trips correctly."""
    n = 6
    data = np.array([3, 1, 4, 1, 5, 9], dtype=np.int32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.int32)

    sorter = make_merge_sort(d_in, None, d_out, None, OpKind.GREATER)
    expected = _run_sorter(sorter, d_in, None, d_out, None, OpKind.GREATER, n)

    path = tmp_path / "sort_desc.alg"
    sorter.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(n, dtype=np.int32)
    result = _run_sorter(loaded, d_in, None, d_out2, None, OpKind.GREATER, n)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Transparent disk cache
# ---------------------------------------------------------------------------


def test_disk_cache_hit_on_second_call(disk_cache):
    """Second call with same args hits disk cache (no NVRTC)."""
    tmp_path = disk_cache

    h = np.array([0], dtype=np.int32)
    d_in = cp.arange(4, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)

    # First call: compiles and writes to disk
    r1 = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    assert any(tmp_path.glob("*.alg")), "Disk cache file should have been written"

    # Clear in-memory cache only (not disk)
    from cuda.compute._caching import _cache_registry

    for cf in _cache_registry.values():
        cf.cache_clear()

    # Second call: should load from disk
    r2 = make_reduce_into(d_in, d_out, OpKind.PLUS, h)

    d_out1 = cp.zeros(1, dtype=np.int32)
    d_out2 = cp.zeros(1, dtype=np.int32)
    res1 = _run_reducer(r1, d_in, d_out1, OpKind.PLUS, h)
    res2 = _run_reducer(r2, d_in, d_out2, OpKind.PLUS, h)
    assert res1 == res2 == 6


def test_disk_cache_clear_removes_files(disk_cache):
    """clear_all_caches() removes .alg files from the disk cache dir."""
    tmp_path = disk_cache

    h = np.array([0.0], dtype=np.float32)
    d_in = cp.ones(4, dtype=np.float32)
    d_out = cp.zeros(1, dtype=np.float32)
    make_reduce_into(d_in, d_out, OpKind.PLUS, h)

    assert any(tmp_path.glob("*.alg"))
    cuda.compute.clear_all_caches()
    assert not any(tmp_path.glob("*.alg"))


def test_disk_cache_different_dtypes_different_entries(disk_cache):
    """Different dtypes produce separate disk-cache entries."""
    tmp_path = disk_cache

    for dtype in [np.int32, np.float32, np.float64]:
        h = np.array([0], dtype=dtype)
        d_in = cp.ones(4, dtype=dtype)
        d_out = cp.zeros(1, dtype=dtype)
        make_reduce_into(d_in, d_out, OpKind.PLUS, h)

    files = list(tmp_path.glob("*.alg"))
    assert len(files) == 3, f"Expected 3 cache files, got {len(files)}"


def test_disk_cache_corrupted_falls_back_to_recompile(disk_cache):
    """A corrupted disk cache file is silently ignored; recompile happens."""
    tmp_path = disk_cache

    h = np.array([0], dtype=np.int32)
    d_in = cp.arange(4, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)

    make_reduce_into(d_in, d_out, OpKind.PLUS, h)

    for f in tmp_path.glob("*.alg"):
        f.write_bytes(b"not valid cclb data")

    from cuda.compute._caching import _cache_registry

    for cf in _cache_registry.values():
        cf.cache_clear()

    r = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    d_out2 = cp.zeros(1, dtype=np.int32)
    result = _run_reducer(r, d_in, d_out2, OpKind.PLUS, h)
    assert result == 6


def test_disk_cache_env_var_enables_cache(disk_cache, monkeypatch):
    """CUDA_COMPUTE_CACHE_DIR env var enables disk cache at import time."""
    # Simulate the module-level env-var handling (can't re-import, so call
    # set_cache_dir directly as the equivalent).
    monkeypatch.setenv("CUDA_COMPUTE_CACHE_DIR", str(disk_cache))
    cuda.compute.set_cache_dir(disk_cache)
    h = np.array([0.0], dtype=np.float32)
    d_in = cp.ones(8, dtype=np.float32)
    d_out = cp.zeros(1, dtype=np.float32)
    make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    assert any(disk_cache.glob("*.alg"))


# ---------------------------------------------------------------------------
# CCLB binary format
# ---------------------------------------------------------------------------


def test_alg_file_uses_cclb_binary_format(tmp_path):
    """Saved .alg files use the CCLB binary format."""

    h = np.array([0], dtype=np.int32)
    d_in = cp.arange(4, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)

    reducer = make_reduce_into(d_in, d_out, OpKind.PLUS, h)
    path = tmp_path / "check_magic.alg"
    reducer.save(path)

    assert path.read_bytes()[:4] == MAGIC


# ---------------------------------------------------------------------------
# Segmented reduce: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def _run_segmented_reducer(
    reducer, d_in, d_out, op, offsets_start, offsets_end, h_init
):
    """Execute a segmented reducer and return host result."""
    n = len(offsets_start)
    tmp_bytes = reducer(None, d_in, d_out, op, n, offsets_start, offsets_end, h_init)
    tmp = TempStorageBuffer(tmp_bytes)
    reducer(tmp, d_in, d_out, op, n, offsets_start, offsets_end, h_init)
    return cp.asnumpy(d_out)


def test_segmented_reduce_save_load_sum_float32(tmp_path):
    """Round-trip: save a float32 segmented sum reducer and reload it."""
    h = np.array([0.0], dtype=np.float32)
    d_in = cp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    d_out = cp.zeros(3, dtype=np.float32)
    offsets_start = cp.array([0, 2, 4], dtype=np.int32)
    offsets_end = cp.array([2, 4, 6], dtype=np.int32)

    reducer = make_segmented_reduce(
        d_in, d_out, offsets_start, offsets_end, OpKind.PLUS, h
    )
    expected = _run_segmented_reducer(
        reducer, d_in, d_out, OpKind.PLUS, offsets_start, offsets_end, h
    )

    path = tmp_path / "seg_reduce_f32.alg"
    reducer.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(3, dtype=np.float32)
    result = _run_segmented_reducer(
        loaded, d_in, d_out2, OpKind.PLUS, offsets_start, offsets_end, h
    )
    np.testing.assert_allclose(result, expected)


def test_segmented_reduce_save_load_min_int32(tmp_path):
    """Round-trip: save an int32 min segmented reducer and reload it."""
    h = np.array([np.iinfo(np.int32).max], dtype=np.int32)
    d_in = cp.array([5, 3, 8, 1, 4, 7], dtype=np.int32)
    d_out = cp.zeros(2, dtype=np.int32)
    offsets_start = cp.array([0, 3], dtype=np.int32)
    offsets_end = cp.array([3, 6], dtype=np.int32)

    reducer = make_segmented_reduce(
        d_in, d_out, offsets_start, offsets_end, OpKind.MINIMUM, h
    )
    expected = _run_segmented_reducer(
        reducer, d_in, d_out, OpKind.MINIMUM, offsets_start, offsets_end, h
    )

    path = tmp_path / "seg_reduce_min_i32.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(2, dtype=np.int32)
    result = _run_segmented_reducer(
        loaded, d_in, d_out2, OpKind.MINIMUM, offsets_start, offsets_end, h
    )
    np.testing.assert_array_equal(result, expected)


def test_segmented_reduce_loaded_reusable(tmp_path):
    """A loaded segmented reducer can be called multiple times."""
    h = np.array([0], dtype=np.int32)
    d_in = cp.array([1, 2, 3, 4], dtype=np.int32)
    d_out = cp.zeros(2, dtype=np.int32)
    offsets_start = cp.array([0, 2], dtype=np.int32)
    offsets_end = cp.array([2, 4], dtype=np.int32)

    reducer = make_segmented_reduce(
        d_in, d_out, offsets_start, offsets_end, OpKind.PLUS, h
    )
    path = tmp_path / "seg_reduce_reuse.alg"
    reducer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(2, dtype=np.int32)
        result = _run_segmented_reducer(
            loaded, d_in, d_out2, OpKind.PLUS, offsets_start, offsets_end, h
        )
        np.testing.assert_array_equal(result, [3, 7])  # [1+2, 3+4]


# ---------------------------------------------------------------------------
# Unary transform: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def test_unary_transform_save_load_float32(tmp_path):
    """Round-trip: save a float32 unary transform and reload it."""
    d_in = cp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_out = cp.zeros(4, dtype=np.float32)

    def square(x):
        return x * x

    transformer = make_unary_transform(d_in, d_out, square)
    transformer(d_in, d_out, square, len(d_in))
    expected = cp.asnumpy(d_out)

    path = tmp_path / "unary_transform_f32.alg"
    transformer.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(4, dtype=np.float32)
    loaded(d_in, d_out2, square, len(d_in))
    np.testing.assert_allclose(cp.asnumpy(d_out2), expected)


def test_unary_transform_save_load_int32(tmp_path):
    """Round-trip: save an int32 unary transform and reload it."""
    d_in = cp.arange(8, dtype=np.int32)
    d_out = cp.zeros(8, dtype=np.int32)

    def double(x):
        return x + x

    transformer = make_unary_transform(d_in, d_out, double)
    transformer(d_in, d_out, double, len(d_in))
    expected = cp.asnumpy(d_out)

    path = tmp_path / "unary_transform_i32.alg"
    transformer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(8, dtype=np.int32)
    loaded(d_in, d_out2, double, len(d_in))
    np.testing.assert_array_equal(cp.asnumpy(d_out2), expected)


def test_unary_transform_loaded_reusable(tmp_path):
    """A loaded unary transformer can be called multiple times."""
    d_in = cp.array([1, 2, 3], dtype=np.int32)
    d_out = cp.zeros(3, dtype=np.int32)

    def negate(x):
        return -x

    transformer = make_unary_transform(d_in, d_out, negate)
    path = tmp_path / "unary_reuse.alg"
    transformer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(3, dtype=np.int32)
        loaded(d_in, d_out2, negate, len(d_in))
        np.testing.assert_array_equal(cp.asnumpy(d_out2), [-1, -2, -3])


# ---------------------------------------------------------------------------
# Binary transform: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def test_binary_transform_save_load_float32(tmp_path):
    """Round-trip: save a float32 binary transform and reload it."""
    d_in1 = cp.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d_in2 = cp.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    d_out = cp.zeros(4, dtype=np.float32)

    def add(a, b):
        return a + b

    transformer = make_binary_transform(d_in1, d_in2, d_out, add)
    transformer(d_in1, d_in2, d_out, add, len(d_in1))
    expected = cp.asnumpy(d_out)

    path = tmp_path / "binary_transform_f32.alg"
    transformer.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(4, dtype=np.float32)
    loaded(d_in1, d_in2, d_out2, add, len(d_in1))
    np.testing.assert_allclose(cp.asnumpy(d_out2), expected)


def test_binary_transform_save_load_int32(tmp_path):
    """Round-trip: save an int32 multiply transform and reload it."""
    d_in1 = cp.array([1, 2, 3, 4], dtype=np.int32)
    d_in2 = cp.array([2, 3, 4, 5], dtype=np.int32)
    d_out = cp.zeros(4, dtype=np.int32)

    def mul(a, b):
        return a * b

    transformer = make_binary_transform(d_in1, d_in2, d_out, mul)
    transformer(d_in1, d_in2, d_out, mul, len(d_in1))
    expected = cp.asnumpy(d_out)

    path = tmp_path / "binary_transform_i32.alg"
    transformer.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(4, dtype=np.int32)
    loaded(d_in1, d_in2, d_out2, mul, len(d_in1))
    np.testing.assert_array_equal(cp.asnumpy(d_out2), expected)


# ---------------------------------------------------------------------------
# Radix sort: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def _run_radix_sorter(sorter, d_in_keys, d_out_keys, n):
    """Execute a radix sorter and return host result."""
    tmp_bytes = sorter(None, d_in_keys, d_out_keys, None, None, n)
    tmp = TempStorageBuffer(tmp_bytes)
    sorter(tmp, d_in_keys, d_out_keys, None, None, n)
    return cp.asnumpy(d_out_keys)


def test_radix_sort_save_load_ascending_int32(tmp_path):
    """Round-trip: save an ascending int32 radix sorter and reload it."""
    n = 8
    data = np.array([5, 3, 8, 1, 4, 7, 2, 6], dtype=np.int32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.int32)

    sorter = make_radix_sort(d_in, d_out, None, None, SortOrder.ASCENDING)
    expected = _run_radix_sorter(sorter, d_in, d_out, n)

    path = tmp_path / "radix_sort_asc_i32.alg"
    sorter.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(n, dtype=np.int32)
    result = _run_radix_sorter(loaded, d_in, d_out2, n)
    np.testing.assert_array_equal(result, expected)


def test_radix_sort_save_load_descending_uint32(tmp_path):
    """Round-trip: save a descending uint32 radix sorter and reload it."""
    n = 6
    data = np.array([3, 1, 4, 1, 5, 9], dtype=np.uint32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.uint32)

    sorter = make_radix_sort(d_in, d_out, None, None, SortOrder.DESCENDING)
    expected = _run_radix_sorter(sorter, d_in, d_out, n)

    path = tmp_path / "radix_sort_desc_u32.alg"
    sorter.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(n, dtype=np.uint32)
    result = _run_radix_sorter(loaded, d_in, d_out2, n)
    np.testing.assert_array_equal(result, expected)


def test_radix_sort_loaded_reusable(tmp_path):
    """A loaded radix sorter can be called multiple times."""
    n = 5
    data = np.array([9, 4, 7, 2, 5], dtype=np.int32)
    d_in = cp.array(data)
    d_out = cp.zeros(n, dtype=np.int32)

    sorter = make_radix_sort(d_in, d_out, None, None, SortOrder.ASCENDING)
    path = tmp_path / "radix_reuse.alg"
    sorter.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(n, dtype=np.int32)
        result = _run_radix_sorter(loaded, d_in, d_out2, n)
        np.testing.assert_array_equal(result, np.sort(data))


# ---------------------------------------------------------------------------
# Binary search: explicit save / load_algorithm
# ---------------------------------------------------------------------------


def _run_lower_bound(searcher, d_data, d_values, d_out):
    """Execute a lower_bound searcher and return host result."""
    searcher(d_data, d_values, d_out, None, len(d_data), len(d_values))
    return cp.asnumpy(d_out)


def _run_upper_bound(searcher, d_data, d_values, d_out):
    """Execute an upper_bound searcher and return host result."""
    searcher(d_data, d_values, d_out, None, len(d_data), len(d_values))
    return cp.asnumpy(d_out)


def test_lower_bound_save_load_int32(tmp_path):
    """Round-trip: save a lower_bound searcher and reload it."""
    d_data = cp.array([1, 3, 5, 7, 9], dtype=np.int32)
    d_values = cp.array([2, 4, 6], dtype=np.int32)
    d_out = cp.zeros(3, dtype=np.uintp)

    searcher = make_lower_bound(d_data, d_values, d_out)
    expected = _run_lower_bound(searcher, d_data, d_values, d_out)

    path = tmp_path / "lower_bound_i32.alg"
    searcher.save(path)
    assert path.exists()

    loaded = cuda.compute.load_algorithm(path)
    d_out2 = cp.zeros(3, dtype=np.uintp)
    result = _run_lower_bound(loaded, d_data, d_values, d_out2)
    np.testing.assert_array_equal(result, expected)


def test_upper_bound_save_load_int32(tmp_path):
    """Round-trip: save an upper_bound searcher and reload it."""
    d_data = cp.array([1, 3, 5, 7, 9], dtype=np.int32)
    d_values = cp.array([3, 5, 7], dtype=np.int32)
    d_out = cp.zeros(3, dtype=np.uintp)

    searcher = make_upper_bound(d_data, d_values, d_out)
    expected = _run_upper_bound(searcher, d_data, d_values, d_out)

    path = tmp_path / "upper_bound_i32.alg"
    searcher.save(path)
    loaded = cuda.compute.load_algorithm(path)

    d_out2 = cp.zeros(3, dtype=np.uintp)
    result = _run_upper_bound(loaded, d_data, d_values, d_out2)
    np.testing.assert_array_equal(result, expected)


def test_binary_search_loaded_reusable(tmp_path):
    """A loaded binary searcher can be called multiple times."""
    d_data = cp.arange(10, dtype=np.int32)
    d_values = cp.array([3, 7], dtype=np.int32)
    d_out = cp.zeros(2, dtype=np.uintp)

    searcher = make_lower_bound(d_data, d_values, d_out)
    path = tmp_path / "lb_reuse.alg"
    searcher.save(path)
    loaded = cuda.compute.load_algorithm(path)

    for _ in range(3):
        d_out2 = cp.zeros(2, dtype=np.uintp)
        result = _run_lower_bound(loaded, d_data, d_values, d_out2)
        np.testing.assert_array_equal(result, [3, 7])
