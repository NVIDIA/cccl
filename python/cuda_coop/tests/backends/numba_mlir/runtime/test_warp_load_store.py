# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise independent physical and logical warp tiles on GPUs.

Two physical warps and smaller logical groups expose incorrect group origins
and barriers that include other groups. Host references add each group's tile
origin separately from the explicit offset and check Load and Store with
independent input patterns. Guarded cases cover counts that differ between
groups while remaining uniform within each group. Trap and divergence probes
run in bounded child processes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from functools import cache
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_WARP_THREADS = 32
_LOGICAL_WARP_THREADS = 8
_BLOCK_THREADS = 2 * _WARP_THREADS
_ITEMS_PER_THREAD = 2
_WARP_TILE_ITEMS = _WARP_THREADS * _ITEMS_PER_THREAD
_LOGICAL_TILE_ITEMS = _LOGICAL_WARP_THREADS * _ITEMS_PER_THREAD
_LOGICAL_GROUPS = _BLOCK_THREADS // _LOGICAL_WARP_THREADS
_BLOCK_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_LOAD_OFFSET = 5
_STORE_OFFSET = 7
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
_DTYPES = (
    pytest.param(np.dtype(np.int8), types.int8, id="int8"),
    pytest.param(np.dtype(np.uint8), types.uint8, id="uint8"),
    pytest.param(np.dtype(np.int16), types.int16, id="int16"),
    pytest.param(np.dtype(np.uint16), types.uint16, id="uint16"),
    pytest.param(np.dtype(np.int32), types.int32, id="int32"),
    pytest.param(np.dtype(np.uint32), types.uint32, id="uint32"),
    pytest.param(np.dtype(np.int64), types.int64, id="int64"),
    pytest.param(np.dtype(np.uint64), types.uint64, id="uint64"),
    pytest.param(np.dtype(np.float32), types.float32, id="float32"),
    pytest.param(np.dtype(np.float64), types.float64, id="float64"),
)
_GRID_STRIDE_ITEMS = 3 * _BLOCK_ITEMS - 17
_GRID_STRIDE_BLOCKS = (_GRID_STRIDE_ITEMS + _BLOCK_ITEMS - 1) // _BLOCK_ITEMS
_QUALIFIED_COOP_ORIGIN = Path(qualified_coop.__file__).resolve()
_SAFE_PATH_FLAG = "-P" if sys.version_info >= (3, 11) else "-I"


def _values(size: int, *, shift: int = 0) -> np.ndarray:
    return ((np.arange(size, dtype=np.int64) * 3 + shift) % 211 - 101).astype(
        np.int32
    )


def _dtype_values(dtype: np.dtype, size: int, *, shift: int = 0) -> np.ndarray:
    """Build dtype-sensitive inputs that expose accidental narrowing.

    Float64 values retain bits below float32 precision, and 64-bit integers
    exceed the 32-bit range. ``shift`` separates Load and Store patterns so
    each direction can be checked independently.
    """

    values = (np.arange(size, dtype=np.int64) * 3 + shift) % 97
    if dtype.kind in {"i", "f"}:
        values = values - 48
    values = values.astype(dtype)
    if dtype.kind == "f":
        values *= dtype.type(0.25)
        if dtype.itemsize == 8:
            values += dtype.type(2**-30)
    elif dtype.itemsize == 8:
        values += np.where(values < 0, -(2**40), 2**40).astype(dtype)
    return values


def _dtype_sentinel(dtype: np.dtype) -> object:
    return dtype.type(211 if dtype.kind == "u" else -101)


@cache
def _load_kernel(algorithm: str, qualified: bool, numba_dtype=types.int32):
    if qualified:
        selector = algorithm

        @cuda.jit
        def kernel(
            source,
            observed,
            valid_items,
            source_offset,
            oob_default,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm=selector,
                valid_items=valid_items,
                oob_default=oob_default,
                offset=source_offset,
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(
            source,
            observed,
            valid_items,
            source_offset,
            oob_default,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=oob_default,
                offset=source_offset,
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@cache
def _store_kernel(algorithm: str, qualified: bool, numba_dtype=types.int32):
    if qualified:
        selector = algorithm

        @cuda.jit
        def kernel(
            source,
            destination,
            valid_items,
            destination_offset,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                payload[item] = source[thread * items_per_thread + item]
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                payload,
                algorithm=selector,
                valid_items=valid_items,
                offset=destination_offset,
            )

    else:

        @cuda.jit
        def kernel(
            source,
            destination,
            valid_items,
            destination_offset,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                payload[item] = source[thread * items_per_thread + item]
            root_coop.store(
                root_coop.this_warp(),
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=destination_offset,
            )

    return kernel


@cache
def _direct_dtype_load_store_kernel(numba_dtype, qualified: bool):
    """Check both direct operations with independent input patterns.

    Observe Load with ordinary stores, then build the Store payload from
    different input data. A matching error in the two primitives cannot cancel
    through a Load-to-Store round trip.
    """

    if qualified:

        @cuda.jit
        def kernel(
            load_source, store_source, observed, destination, items_per_thread
        ):
            thread = cuda.threadIdx.x
            load_payload = qualified_coop.ThreadData(
                items_per_thread, dtype=numba_dtype
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                load_source,
                load_payload,
                algorithm="direct",
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                payload,
                algorithm="direct",
            )

    else:

        @cuda.jit
        def kernel(
            load_source, store_source, observed, destination, items_per_thread
        ):
            thread = cuda.threadIdx.x
            load_payload = root_coop.ThreadData(
                items_per_thread, dtype=numba_dtype
            )
            root_coop.load(
                root_coop.this_warp(),
                load_source,
                load_payload,
                algorithm="direct",
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            root_coop.store(
                root_coop.this_warp(),
                destination,
                payload,
                algorithm="direct",
            )

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize(("numpy_dtype", "numba_dtype"), _DTYPES)
def test_direct_multi_item_load_store_matches_oracles_for_every_dtype(
    qualified: bool, numpy_dtype: np.dtype, numba_dtype, *, items_per_thread
) -> None:
    load_source = _dtype_values(
        numpy_dtype, (_BLOCK_THREADS * items_per_thread), shift=7
    )
    store_source = _dtype_values(
        numpy_dtype, (_BLOCK_THREADS * items_per_thread), shift=29
    )
    sentinel = _dtype_sentinel(numpy_dtype)
    observed = np.full(
        (_BLOCK_THREADS * items_per_thread), sentinel, dtype=numpy_dtype
    )
    destination = np.full(
        (_BLOCK_THREADS * items_per_thread), sentinel, dtype=numpy_dtype
    )

    _direct_dtype_load_store_kernel(numba_dtype, qualified)[1, _BLOCK_THREADS](
        load_source, store_source, observed, destination, items_per_thread
    )

    np.testing.assert_array_equal(observed, load_source)
    np.testing.assert_array_equal(destination, store_source)


def _tile_index(
    algorithm: str, lane: int, item: int, *, items_per_thread
) -> int:
    """Map a lane's item to a blocked or striped index within its warp."""

    if algorithm == "striped":
        return lane + item * _WARP_THREADS
    return lane * items_per_thread + item


def _expected_loaded_payload(
    source: np.ndarray,
    *,
    algorithm: str,
    valid_items: int,
    offset: int,
    oob_default: int,
    items_per_thread,
) -> np.ndarray:
    """Compute separate Load results for both physical warps.

    Keep observed payload slots in thread-major order. For each valid item,
    add its warp's tile origin and the caller's offset to the layout index.
    Each warp applies the count independently; other slots use the default.
    """

    expected = np.full(
        (_BLOCK_THREADS * items_per_thread), oob_default, dtype=source.dtype
    )
    for thread in range(_BLOCK_THREADS):
        warp = thread // _WARP_THREADS
        lane = thread % _WARP_THREADS
        for item in range(items_per_thread):
            payload_index = thread * items_per_thread + item
            tile_index = _tile_index(
                algorithm, lane, item, items_per_thread=items_per_thread
            )
            if tile_index < valid_items:
                expected[payload_index] = source[
                    offset
                    + warp * (_WARP_THREADS * items_per_thread)
                    + tile_index
                ]
    return expected


def _expected_stored_payload(
    source: np.ndarray,
    destination: np.ndarray,
    *,
    algorithm: str,
    valid_items: int,
    offset: int,
    items_per_thread,
) -> np.ndarray:
    """Map per-thread Store inputs to each physical warp's destination tile.

    Add the warp origin and explicit offset after applying the layout. Begin
    with the original destination so invalid items and guard regions remain
    part of the reference check.
    """

    expected = destination.copy()
    for thread in range(_BLOCK_THREADS):
        warp = thread // _WARP_THREADS
        lane = thread % _WARP_THREADS
        for item in range(items_per_thread):
            payload_index = thread * items_per_thread + item
            tile_index = _tile_index(
                algorithm, lane, item, items_per_thread=items_per_thread
            )
            if tile_index < valid_items:
                expected[
                    offset
                    + warp * (_WARP_THREADS * items_per_thread)
                    + tile_index
                ] = source[payload_index]
    return expected


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid_items",
    (0, _WARP_TILE_ITEMS - 9, _WARP_TILE_ITEMS),
    ids=("zero", "partial", "full"),
)
def test_each_warp_load_algorithm_matches_an_independent_two_warp_oracle(
    qualified: bool, algorithm: str, valid_items: int, *, items_per_thread
) -> None:
    valid_items = min(valid_items, _WARP_THREADS * items_per_thread)
    source = _values(
        _LOAD_OFFSET + (_BLOCK_THREADS * items_per_thread) + 3, shift=31
    )
    observed = np.full((_BLOCK_THREADS * items_per_thread), 71, dtype=np.int32)
    expected = _expected_loaded_payload(
        source,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=_LOAD_OFFSET,
        oob_default=-29,
        items_per_thread=items_per_thread,
    )

    _load_kernel(algorithm, qualified)[1, _BLOCK_THREADS](
        source,
        observed,
        np.int32(valid_items),
        np.int64(_LOAD_OFFSET),
        np.int32(-29),
        items_per_thread,
    )

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid_items",
    (0, _WARP_TILE_ITEMS - 9, _WARP_TILE_ITEMS),
    ids=("zero", "partial", "full"),
)
def test_each_warp_store_algorithm_masks_each_warp(
    qualified: bool, algorithm: str, valid_items: int, *, items_per_thread
) -> None:
    valid_items = min(valid_items, _WARP_THREADS * items_per_thread)
    source = _values((_BLOCK_THREADS * items_per_thread), shift=43)
    destination = np.full(
        _STORE_OFFSET + (_BLOCK_THREADS * items_per_thread) + 3,
        -41,
        dtype=np.int32,
    )
    expected = _expected_stored_payload(
        source,
        destination,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=_STORE_OFFSET,
        items_per_thread=items_per_thread,
    )

    _store_kernel(algorithm, qualified)[1, _BLOCK_THREADS](
        source,
        destination,
        np.int32(valid_items),
        np.int64(_STORE_OFFSET),
        items_per_thread,
    )

    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(("numpy_dtype", "numba_dtype"), _DTYPES)
@pytest.mark.parametrize("algorithm", ("striped", "transpose"))
def test_non_direct_partial_load_store_matches_dtype_oracles(
    numpy_dtype, numba_dtype, algorithm, *, items_per_thread
):
    valid_items = (_WARP_THREADS * items_per_thread) - 9
    load_source = _dtype_values(
        numpy_dtype,
        _LOAD_OFFSET + (_BLOCK_THREADS * items_per_thread) + 3,
        shift=31,
    )
    store_source = _dtype_values(
        numpy_dtype, (_BLOCK_THREADS * items_per_thread), shift=43
    )
    sentinel = _dtype_sentinel(numpy_dtype)
    default = numpy_dtype.type(3.25 if numpy_dtype.kind == "f" else 103)
    observed = np.full(
        (_BLOCK_THREADS * items_per_thread), sentinel, dtype=numpy_dtype
    )
    destination = np.full(
        _STORE_OFFSET + (_BLOCK_THREADS * items_per_thread) + 3,
        sentinel,
        dtype=numpy_dtype,
    )
    expected_load = _expected_loaded_payload(
        load_source,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=_LOAD_OFFSET,
        oob_default=default,
        items_per_thread=items_per_thread,
    )
    expected_store = _expected_stored_payload(
        store_source,
        destination,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=_STORE_OFFSET,
        items_per_thread=items_per_thread,
    )

    _load_kernel(algorithm, False, numba_dtype)[1, _BLOCK_THREADS](
        load_source,
        observed,
        np.int32(valid_items),
        np.int64(_LOAD_OFFSET),
        default,
        items_per_thread,
    )
    _store_kernel(algorithm, True, numba_dtype)[1, _BLOCK_THREADS](
        store_source,
        destination,
        np.int32(valid_items),
        np.int64(_STORE_OFFSET),
        items_per_thread,
    )

    np.testing.assert_array_equal(observed, expected_load)
    np.testing.assert_array_equal(destination, expected_store)


def _logical_tile_index(
    algorithm: str, lane: int, item: int, width: int, *, items_per_thread
) -> int:
    if algorithm == "striped":
        return lane + item * width
    return lane * items_per_thread + item


@cache
def _logical_load_store_kernel(algorithm: str, qualified: bool):
    """Give each logical warp its own count and offset in a shared block.

    Each control value is uniform within its group. Load and Store use
    different source data so host references check both directions.
    """

    if qualified:
        load_algorithm = algorithm
        store_algorithm = algorithm

        @cuda.jit
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_by_group,
            offset_by_group,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            offset = offset_by_group[group_index]
            group = qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                group,
                load_source,
                payload,
                algorithm=load_algorithm,
                valid_items=valid_by_group[group_index],
                oob_default=types.int32(-127),
                offset=offset,
            )
            for item in range(items_per_thread):
                payload_index = thread * items_per_thread + item
                observed[payload_index] = payload[item]
                payload[item] = store_source[payload_index]
            qualified_coop.store(
                group,
                destination,
                payload,
                algorithm=store_algorithm,
                valid_items=valid_by_group[group_index],
                offset=offset,
            )

    else:

        @cuda.jit
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_by_group,
            offset_by_group,
            items_per_thread,
        ):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            offset = offset_by_group[group_index]
            group = root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                group,
                load_source,
                payload,
                algorithm=algorithm,
                valid_items=valid_by_group[group_index],
                oob_default=types.int32(-127),
                offset=offset,
            )
            for item in range(items_per_thread):
                payload_index = thread * items_per_thread + item
                observed[payload_index] = payload[item]
                payload[item] = store_source[payload_index]
            root_coop.store(
                group,
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_by_group[group_index],
                offset=offset,
            )

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_logical_warp_algorithms_use_independent_group_tiles(
    qualified: bool, algorithm: str, *, items_per_thread
) -> None:
    offsets_by_group = np.array([0, 1, 3, 4, 6, 7, 9, 10], dtype=np.int64)
    allocation_items = (
        int(offsets_by_group.max()) + (_BLOCK_THREADS * items_per_thread) + 3
    )
    load_source = _values(allocation_items, shift=47)
    store_source = _values((_BLOCK_THREADS * items_per_thread), shift=53)
    observed = np.full((_BLOCK_THREADS * items_per_thread), 19, dtype=np.int32)
    destination = np.full(allocation_items, -31, dtype=np.int32)
    valid_by_group = np.array(
        [0, 1, 5, (_LOGICAL_WARP_THREADS * items_per_thread), 3, 11, 7, 15],
        dtype=np.int32,
    )
    np.minimum(
        valid_by_group,
        _LOGICAL_WARP_THREADS * items_per_thread,
        out=valid_by_group,
    )
    expected_observed = np.full(
        (_BLOCK_THREADS * items_per_thread), -127, dtype=np.int32
    )
    expected_destination = destination.copy()
    for thread in range(_BLOCK_THREADS):
        group_index = thread // _LOGICAL_WARP_THREADS
        lane = thread % _LOGICAL_WARP_THREADS
        valid_items = valid_by_group[group_index]
        offset = offsets_by_group[group_index]
        for item in range(items_per_thread):
            payload_index = thread * items_per_thread + item
            tile_index = _logical_tile_index(
                algorithm,
                lane,
                item,
                _LOGICAL_WARP_THREADS,
                items_per_thread=items_per_thread,
            )
            if tile_index < valid_items:
                memory_index = (
                    offset
                    + group_index * (_LOGICAL_WARP_THREADS * items_per_thread)
                    + tile_index
                )
                expected_observed[payload_index] = load_source[memory_index]
                expected_destination[memory_index] = store_source[payload_index]

    _logical_load_store_kernel(algorithm, qualified)[1, _BLOCK_THREADS](
        load_source,
        store_source,
        observed,
        destination,
        valid_by_group,
        offsets_by_group,
        items_per_thread,
    )

    np.testing.assert_array_equal(observed, expected_observed)
    np.testing.assert_array_equal(destination, expected_destination)


@cache
def _logical_partial_transpose_load_kernel(qualified: bool):
    """Observe only valid items from each logical warp's transpose load.

    Omit a default and leave invalid payload slots unobserved. The host
    sentinel belongs to the observation array, not the payload contract.
    """

    if qualified:

        @cuda.jit
        def kernel(source, observed, valid_by_group, items_per_thread):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_by_group[group_index],
            )
            for item in range(items_per_thread):
                lane = thread % _LOGICAL_WARP_THREADS
                if lane * items_per_thread + item < valid_by_group[group_index]:
                    observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, valid_by_group, items_per_thread):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_by_group[group_index],
            )
            for item in range(items_per_thread):
                lane = thread % _LOGICAL_WARP_THREADS
                if lane * items_per_thread + item < valid_by_group[group_index]:
                    observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_logical_transpose_load_reads_valid_items_in_nonzero_groups(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values((_BLOCK_THREADS * items_per_thread), shift=61)
    observed = np.full((_BLOCK_THREADS * items_per_thread), 23, dtype=np.int32)
    valid_by_group = np.array(
        [0, 1, 5, (_LOGICAL_WARP_THREADS * items_per_thread), 3, 11, 7, 15],
        dtype=np.int32,
    )
    np.minimum(
        valid_by_group,
        _LOGICAL_WARP_THREADS * items_per_thread,
        out=valid_by_group,
    )
    expected = observed.copy()
    for thread in range(_BLOCK_THREADS):
        group_index = thread // _LOGICAL_WARP_THREADS
        lane = thread % _LOGICAL_WARP_THREADS
        valid_items = valid_by_group[group_index]
        for item in range(items_per_thread):
            payload_index = thread * items_per_thread + item
            tile_index = lane * items_per_thread + item
            if tile_index < valid_items:
                expected[payload_index] = source[
                    group_index * (_LOGICAL_WARP_THREADS * items_per_thread)
                    + tile_index
                ]

    _logical_partial_transpose_load_kernel(qualified)[1, _BLOCK_THREADS](
        source, observed, valid_by_group, items_per_thread
    )

    np.testing.assert_array_equal(observed, expected)


@cache
def _logical_width_direct_kernel(width: int, qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp().group_by(width),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp().group_by(width),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
def test_every_logical_warp_width_addresses_consecutive_tiles(
    qualified: bool, width: int, *, items_per_thread
) -> None:
    source = _values((_BLOCK_THREADS * items_per_thread), shift=67)
    observed = np.full((_BLOCK_THREADS * items_per_thread), -1, dtype=np.int32)

    _logical_width_direct_kernel(width, qualified)[1, _BLOCK_THREADS](
        source, observed, items_per_thread
    )

    np.testing.assert_array_equal(observed, source)


@cache
def _logical_direct_dtype_load_store_kernel(numba_dtype, qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(
            load_source, store_source, observed, destination, items_per_thread
        ):
            thread = cuda.threadIdx.x
            group = qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            load_payload = qualified_coop.ThreadData(
                items_per_thread, dtype=numba_dtype
            )
            qualified_coop.load(
                group,
                load_source,
                load_payload,
                algorithm="direct",
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            qualified_coop.store(
                group,
                destination,
                payload,
                algorithm="direct",
            )

    else:

        @cuda.jit
        def kernel(
            load_source, store_source, observed, destination, items_per_thread
        ):
            thread = cuda.threadIdx.x
            group = root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            load_payload = root_coop.ThreadData(
                items_per_thread, dtype=numba_dtype
            )
            root_coop.load(
                group,
                load_source,
                load_payload,
                algorithm="direct",
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=numba_dtype,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            root_coop.store(
                group,
                destination,
                payload,
                algorithm="direct",
            )

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize(("numpy_dtype", "numba_dtype"), _DTYPES)
def test_logical_direct_load_store_matches_every_dtype_oracle(
    qualified: bool, numpy_dtype: np.dtype, numba_dtype, *, items_per_thread
) -> None:
    load_source = _dtype_values(
        numpy_dtype, (_BLOCK_THREADS * items_per_thread), shift=13
    )
    store_source = _dtype_values(
        numpy_dtype, (_BLOCK_THREADS * items_per_thread), shift=37
    )
    sentinel = _dtype_sentinel(numpy_dtype)
    observed = np.full(
        (_BLOCK_THREADS * items_per_thread), sentinel, dtype=numpy_dtype
    )
    destination = np.full(
        (_BLOCK_THREADS * items_per_thread), sentinel, dtype=numpy_dtype
    )

    _logical_direct_dtype_load_store_kernel(numba_dtype, qualified)[
        1, _BLOCK_THREADS
    ](load_source, store_source, observed, destination, items_per_thread)

    np.testing.assert_array_equal(observed, load_source)
    np.testing.assert_array_equal(destination, store_source)


@cache
def _partial_load_kernel(algorithm: str, qualified: bool):
    """Observe only valid items according to each warp's selected layout.

    Without a default value, the test checks the valid prefix and leaves
    invalid payload slots unobserved. The host reference separately applies
    the same layout definition to each warp's tile origin.
    """

    striped = algorithm == "striped"
    if qualified:
        selector = algorithm

        @cuda.jit
        def kernel(source, observed, valid_items, items_per_thread):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm=selector,
                valid_items=valid_items,
            )
            for item in range(items_per_thread):
                lane = thread % _WARP_THREADS
                tile_index = (
                    lane + item * _WARP_THREADS
                    if striped
                    else lane * items_per_thread + item
                )
                if tile_index < valid_items:
                    observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, valid_items, items_per_thread):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
            )
            for item in range(items_per_thread):
                lane = thread % _WARP_THREADS
                tile_index = (
                    lane + item * _WARP_THREADS
                    if striped
                    else lane * items_per_thread + item
                )
                if tile_index < valid_items:
                    observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_partial_load_reads_valid_items_for_each_layout_and_warp(
    qualified: bool, algorithm: str, *, items_per_thread
) -> None:
    valid_items = (_WARP_THREADS * items_per_thread) - 11
    source = _values((_BLOCK_THREADS * items_per_thread), shift=59)
    observed = np.full((_BLOCK_THREADS * items_per_thread), 71, dtype=np.int32)
    expected = observed.copy()
    for thread in range(_BLOCK_THREADS):
        warp = thread // _WARP_THREADS
        lane = thread % _WARP_THREADS
        for item in range(items_per_thread):
            payload_index = thread * items_per_thread + item
            tile_index = _tile_index(
                algorithm, lane, item, items_per_thread=items_per_thread
            )
            if tile_index < valid_items:
                expected[payload_index] = source[
                    warp * (_WARP_THREADS * items_per_thread) + tile_index
                ]

    _partial_load_kernel(algorithm, qualified)[1, _BLOCK_THREADS](
        source, observed, np.int32(valid_items), items_per_thread
    )

    np.testing.assert_array_equal(observed, expected)


@cache
def _per_warp_valid_items_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, observed, valid_by_warp, items_per_thread):
            thread = cuda.threadIdx.x
            warp = thread // _WARP_THREADS
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
                valid_items=valid_by_warp[warp],
                oob_default=types.int32(-83),
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, valid_by_warp, items_per_thread):
            thread = cuda.threadIdx.x
            warp = thread // _WARP_THREADS
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
                valid_items=valid_by_warp[warp],
                oob_default=types.int32(-83),
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_runtime_valid_items_can_differ_between_physical_warps(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values((_BLOCK_THREADS * items_per_thread), shift=71)
    observed = np.full((_BLOCK_THREADS * items_per_thread), 17, dtype=np.int32)
    valid_by_warp = np.array(
        [13, (_WARP_THREADS * items_per_thread) - 5], dtype=np.int32
    )
    expected = np.full((_BLOCK_THREADS * items_per_thread), -83, dtype=np.int32)
    for warp, valid_items in enumerate(valid_by_warp):
        begin = warp * (_WARP_THREADS * items_per_thread)
        expected[begin : begin + valid_items] = source[
            begin : begin + valid_items
        ]

    _per_warp_valid_items_kernel(qualified)[1, _BLOCK_THREADS](
        source, observed, valid_by_warp, items_per_thread
    )

    np.testing.assert_array_equal(observed, expected)


@cache
def _multidimensional_load_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x + cuda.blockDim.x * (
                cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x + cuda.blockDim.x * (
                cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
@pytest.mark.parametrize(
    "block_shape",
    ((16, 4), (8, 4, 2)),
    ids=("2d", "3d"),
)
def test_physical_warp_origin_uses_x_major_multidimensional_rank(
    qualified: bool, block_shape: tuple[int, ...], *, items_per_thread
) -> None:
    source = _values((_BLOCK_THREADS * items_per_thread), shift=83)
    observed = np.full((_BLOCK_THREADS * items_per_thread), -1, dtype=np.int32)

    _multidimensional_load_kernel(qualified)[1, block_shape](
        source, observed, items_per_thread
    )

    np.testing.assert_array_equal(observed, source)


@cache
def _logical_multidimensional_load_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x + cuda.blockDim.x * (
                cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x + cuda.blockDim.x * (
                cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_logical_warp_origin_uses_x_major_multidimensional_rank(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values((_BLOCK_THREADS * items_per_thread), shift=79)
    observed = np.full((_BLOCK_THREADS * items_per_thread), -1, dtype=np.int32)

    _logical_multidimensional_load_kernel(qualified)[1, (8, 4, 2)](
        source, observed, items_per_thread
    )

    np.testing.assert_array_equal(observed, source)


@cache
def _static_control_load_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
                valid_items=(_WARP_THREADS * items_per_thread) - 7,
                oob_default=-113,
                offset=_LOAD_OFFSET,
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    else:

        @cuda.jit
        def kernel(source, observed, items_per_thread):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm="direct",
                valid_items=(_WARP_THREADS * items_per_thread) - 7,
                oob_default=-113,
                offset=_LOAD_OFFSET,
            )
            for item in range(items_per_thread):
                observed[thread * items_per_thread + item] = payload[item]

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_physical_warp_static_controls_share_runtime_addressing(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values(
        _LOAD_OFFSET + (_BLOCK_THREADS * items_per_thread), shift=89
    )
    observed = np.full((_BLOCK_THREADS * items_per_thread), -1, dtype=np.int32)
    expected = np.full(
        (_BLOCK_THREADS * items_per_thread), -113, dtype=np.int32
    )
    valid_items = (_WARP_THREADS * items_per_thread) - 7
    for warp in range(2):
        begin = warp * (_WARP_THREADS * items_per_thread)
        source_begin = _LOAD_OFFSET + begin
        expected[begin : begin + valid_items] = source[
            source_begin : source_begin + valid_items
        ]

    _static_control_load_kernel(qualified)[1, _BLOCK_THREADS](
        source, observed, items_per_thread
    )

    np.testing.assert_array_equal(observed, expected)


@cache
def _scalar_store_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            value = types.int32(source[thread] + 1)
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                value,
                algorithm="direct",
                offset=_STORE_OFFSET,
            )

    else:

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            value = types.int32(source[thread] + 1)
            root_coop.store(
                root_coop.this_warp(),
                destination,
                value,
                algorithm="direct",
                offset=_STORE_OFFSET,
            )

    return kernel


@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_physical_warp_scalar_store_uses_destination_dtype(
    qualified: bool,
) -> None:
    source = _values(_BLOCK_THREADS, shift=91)
    destination = np.full(
        _STORE_OFFSET + _BLOCK_THREADS + 3, -17, dtype=np.int32
    )
    expected = destination.copy()
    expected[_STORE_OFFSET : _STORE_OFFSET + _BLOCK_THREADS] = source + 1

    _scalar_store_kernel(qualified)[1, _BLOCK_THREADS](source, destination)

    np.testing.assert_array_equal(destination, expected)


@cache
def _literal_scalar_store_kernel(qualified: bool):
    if qualified:

        @cuda.jit
        def kernel(destination):
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
                offset=_STORE_OFFSET,
            )

    else:

        @cuda.jit
        def kernel(destination):
            root_coop.store(
                root_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
                offset=_STORE_OFFSET,
            )

    return kernel


@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_physical_warp_scalar_literal_infers_the_destination_dtype(
    qualified: bool,
) -> None:
    destination = np.full(
        _STORE_OFFSET + _BLOCK_THREADS + 3, -17, dtype=np.int32
    )
    expected = destination.copy()
    expected[_STORE_OFFSET : _STORE_OFFSET + _BLOCK_THREADS] = 23

    _literal_scalar_store_kernel(qualified)[1, _BLOCK_THREADS](destination)

    np.testing.assert_array_equal(destination, expected)


@cache
def _grid_stride_transpose_kernel(qualified: bool):
    """Clamp a separate valid count for each warp in each block's tile.

    Pass only the block origin as ``offset``. The provider adds the warp's
    tile origin, including for the partial last warp of the final block.
    """

    if qualified:

        @cuda.jit
        def kernel(source, destination, items_per_thread):
            thread = cuda.threadIdx.x
            warp = thread // _WARP_THREADS
            block_offset = cuda.blockIdx.x * (_BLOCK_THREADS * items_per_thread)
            remaining = (
                source.size
                - block_offset
                - warp * (_WARP_THREADS * items_per_thread)
            )
            valid_items = min(
                max(remaining, 0), (_WARP_THREADS * items_per_thread)
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )

    else:

        @cuda.jit
        def kernel(source, destination, items_per_thread):
            thread = cuda.threadIdx.x
            warp = thread // _WARP_THREADS
            block_offset = cuda.blockIdx.x * (_BLOCK_THREADS * items_per_thread)
            remaining = (
                source.size
                - block_offset
                - warp * (_WARP_THREADS * items_per_thread)
            )
            valid_items = min(
                max(remaining, 0), (_WARP_THREADS * items_per_thread)
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                root_coop.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )
            root_coop.store(
                root_coop.this_warp(),
                destination,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_grid_stride_tail_clamps_valid_items_per_physical_warp(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values((3 * (_BLOCK_THREADS * items_per_thread) - 17), shift=103)
    destination = np.full(
        (3 * (_BLOCK_THREADS * items_per_thread) - 17), -1, dtype=np.int32
    )

    _grid_stride_transpose_kernel(qualified)[
        (
            (
                3 * (_BLOCK_THREADS * items_per_thread)
                - 17
                + _BLOCK_THREADS * items_per_thread
                - 1
            )
            // (_BLOCK_THREADS * items_per_thread)
        ),
        _BLOCK_THREADS,
    ](source, destination, items_per_thread)

    np.testing.assert_array_equal(destination, source)


@cache
def _logical_grid_stride_transpose_kernel(qualified: bool):
    """Clamp the tail count for every logical group in each block's tile.

    Pass the block origin as ``offset`` and let the provider add the logical
    group's tile origin. Groups beyond the source use a zero valid count.
    """

    if qualified:

        @cuda.jit
        def kernel(source, destination, items_per_thread):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            block_offset = cuda.blockIdx.x * (_BLOCK_THREADS * items_per_thread)
            remaining = (
                source.size
                - block_offset
                - group_index * (_LOGICAL_WARP_THREADS * items_per_thread)
            )
            valid_items = min(
                max(remaining, 0), (_LOGICAL_WARP_THREADS * items_per_thread)
            )
            group = qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                group,
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )
            qualified_coop.store(
                group,
                destination,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )

    else:

        @cuda.jit
        def kernel(source, destination, items_per_thread):
            thread = cuda.threadIdx.x
            group_index = thread // _LOGICAL_WARP_THREADS
            block_offset = cuda.blockIdx.x * (_BLOCK_THREADS * items_per_thread)
            remaining = (
                source.size
                - block_offset
                - group_index * (_LOGICAL_WARP_THREADS * items_per_thread)
            )
            valid_items = min(
                max(remaining, 0), (_LOGICAL_WARP_THREADS * items_per_thread)
            )
            group = root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                group,
                source,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )
            root_coop.store(
                group,
                destination,
                payload,
                algorithm="transpose",
                valid_items=valid_items,
                offset=block_offset,
            )

    return kernel


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_grid_stride_tail_clamps_valid_items_per_logical_warp(
    qualified: bool, *, items_per_thread
) -> None:
    source = _values((3 * (_BLOCK_THREADS * items_per_thread) - 17), shift=107)
    destination = np.full(
        (3 * (_BLOCK_THREADS * items_per_thread) - 17), -1, dtype=np.int32
    )

    _logical_grid_stride_transpose_kernel(qualified)[
        (
            (
                3 * (_BLOCK_THREADS * items_per_thread)
                - 17
                + _BLOCK_THREADS * items_per_thread
                - 1
            )
            // (_BLOCK_THREADS * items_per_thread)
        ),
        _BLOCK_THREADS,
    ](source, destination, items_per_thread)

    np.testing.assert_array_equal(destination, source)


def _run_divergent_warp_probe(
    qualified: bool, *, items_per_thread
) -> subprocess.CompletedProcess[str]:
    """Run a one-warp primitive with a child-process timeout.

    A barrier that includes the inactive warp could stall the kernel. The
    child contains that failure and checks that it imports the same
    ``cuda.coop`` package as the test process before launching. Return its
    status and output for the test assertion.
    """

    if qualified:
        thread_data = "qualified_coop.ThreadData"
        group = "qualified_coop.this_warp()"
        load = "qualified_coop.load"
        algorithm = repr("transpose")
    else:
        thread_data = "root_coop.ThreadData"
        group = "root_coop.this_warp()"
        load = "root_coop.load"
        algorithm = repr("transpose")

    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from numba_cuda_mlir import types
from pathlib import Path

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(qualified_coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"divergent probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_WARP_THREADS = {_WARP_THREADS}
_BLOCK_THREADS = {_BLOCK_THREADS}
_ITEMS_PER_THREAD = {items_per_thread}

@cuda.jit
def kernel(source, observed, items_per_thread):
    thread = cuda.threadIdx.x
    warp = thread // _WARP_THREADS
    if warp == 0:
        payload = {thread_data}(items_per_thread, dtype=types.int32)
        {load}(
            {group},
            source,
            payload,
            algorithm={algorithm},
        )
        for item in range(items_per_thread):
            observed[thread * items_per_thread + item] = payload[item]

source = ((np.arange(_BLOCK_THREADS * _ITEMS_PER_THREAD, dtype=np.int64) * 3 + 97) % 211 - 101).astype(np.int32)
observed = np.full_like(source, -37)
expected = observed.copy()
expected[: _WARP_THREADS * _ITEMS_PER_THREAD] = source[: _WARP_THREADS * _ITEMS_PER_THREAD]
kernel[1, _BLOCK_THREADS](source, observed, _ITEMS_PER_THREAD)
cuda.synchronize()
np.testing.assert_array_equal(observed, expected)
"""  # noqa: E501 - Preserve embedded source bytes.
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_one_physical_warp_can_take_a_transpose_primitive_path(
    qualified: bool, *, items_per_thread
) -> None:
    result = _run_divergent_warp_probe(
        qualified, items_per_thread=items_per_thread
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_divergent_logical_warp_probe(
    qualified: bool, *, items_per_thread
) -> subprocess.CompletedProcess[str]:
    """Probe a nonzero logical group in each physical warp with a timeout.

    Only subgroup two participates, testing masks that must exclude other
    lanes. A child process bounds an incorrect barrier and checks the
    package's import origin. Return its status and captured output.
    """

    if qualified:
        thread_data = "qualified_coop.ThreadData"
        group = "qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)"
        load = "qualified_coop.load"
        algorithm = repr("transpose")
    else:
        thread_data = "root_coop.ThreadData"
        group = "root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS)"
        load = "root_coop.load"
        algorithm = repr("transpose")

    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from numba_cuda_mlir import types
from pathlib import Path

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(qualified_coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"divergent logical-warp probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_WARP_THREADS = {_WARP_THREADS}
_LOGICAL_WARP_THREADS = {_LOGICAL_WARP_THREADS}
_BLOCK_THREADS = {_BLOCK_THREADS}
_ITEMS_PER_THREAD = {items_per_thread}

@cuda.jit
def kernel(source, observed, items_per_thread):
    thread = cuda.threadIdx.x
    subgroup = (thread % _WARP_THREADS) // _LOGICAL_WARP_THREADS
    if subgroup == 2:
        payload = {thread_data}(items_per_thread, dtype=types.int32)
        {load}(
            {group},
            source,
            payload,
            algorithm={algorithm},
        )
        for item in range(items_per_thread):
            observed[thread * items_per_thread + item] = payload[item]

source = ((np.arange(_BLOCK_THREADS * _ITEMS_PER_THREAD, dtype=np.int64) * 3 + 113) % 211 - 101).astype(np.int32)
observed = np.full_like(source, -37)
expected = observed.copy()
for warp in range(_BLOCK_THREADS // _WARP_THREADS):
    begin = (
        warp * _WARP_THREADS + 2 * _LOGICAL_WARP_THREADS
    ) * _ITEMS_PER_THREAD
    end = begin + _LOGICAL_WARP_THREADS * _ITEMS_PER_THREAD
    expected[begin:end] = source[begin:end]
kernel[1, _BLOCK_THREADS](source, observed, _ITEMS_PER_THREAD)
cuda.synchronize()
np.testing.assert_array_equal(observed, expected)
"""  # noqa: E501 - Preserve embedded source bytes.
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_one_logical_warp_per_physical_warp_can_diverge_at_transpose(
    qualified: bool, *, items_per_thread
) -> None:
    result = _run_divergent_logical_warp_probe(
        qualified, items_per_thread=items_per_thread
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_invalid_runtime_valid_items_probe(
    operation: str,
    valid_items: int,
    *,
    logical_width: int | None = None,
    items_per_thread,
) -> subprocess.CompletedProcess[str]:
    """Run an invalid per-group count in a disposable CUDA context.

    ``logical_width`` selects a logical group; ``None`` uses a full physical
    warp. The child checks that its ``cuda.coop`` import origin matches the
    test process, then launches with ``items_per_thread`` items per lane.
    A device trap poisons its CUDA context. Return captured output and status
    so the test can require that fault while keeping its own context usable.
    """

    group = "root_coop.this_warp()"
    if logical_width is not None:
        group = f"root_coop.this_warp().group_by({logical_width})"
    operation_body = textwrap.indent(
        textwrap.dedent(
            {
                "load": f"""
            payload = root_coop.ThreadData(
                items_per_thread=_ITEMS_PER_THREAD, dtype=types.int32
            )
            root_coop.load(
                {group},
                source,
                payload,
                valid_items=valid_items,
            )
        """,
                "store": f"""
            payload = root_coop.ThreadData(
                items_per_thread=_ITEMS_PER_THREAD, dtype=types.int32
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            root_coop.store(
                {group},
                destination,
                payload,
                valid_items=valid_items,
            )
        """,
            }[operation]
        ).strip(),
        " " * 4,
    )
    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from numba_cuda_mlir import types
from pathlib import Path

import cuda.coop.numba_mlir as numba_coop
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(numba_coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"trap probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_THREADS = {_BLOCK_THREADS}
_ITEMS_PER_THREAD = {items_per_thread}

@cuda.jit
def kernel(source, destination, valid_items):
    thread = cuda.threadIdx.x
{operation_body}

source = np.arange(
    _THREADS * _ITEMS_PER_THREAD,
    dtype=np.int32,
)
destination = np.full_like(source, -1)
kernel[1, _THREADS](source, destination, np.int64({valid_items}))
cuda.synchronize()
raise AssertionError("invalid valid_items did not trap")
"""
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize(
    "valid_items",
    (
        pytest.param(-1, id="negative"),
        pytest.param(_WARP_TILE_ITEMS + 1, id="beyond-warp-tile"),
    ),
)
def test_runtime_valid_items_out_of_range_traps_in_an_isolated_context(
    operation: str, valid_items: int, *, items_per_thread
) -> None:
    valid_items = (
        valid_items if valid_items < 0 else _WARP_THREADS * items_per_thread + 1
    )
    result = _run_invalid_runtime_valid_items_probe(
        operation, valid_items, items_per_thread=items_per_thread
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output

    source = _values((_BLOCK_THREADS * items_per_thread), shift=109)
    observed = np.full((_BLOCK_THREADS * items_per_thread), -1, dtype=np.int32)
    _load_kernel("direct", False)[1, _BLOCK_THREADS](
        source,
        observed,
        np.int32(_WARP_THREADS * items_per_thread),
        np.int64(0),
        np.int32(-1),
        items_per_thread,
    )
    np.testing.assert_array_equal(observed, source)


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize(
    "valid_items",
    (
        pytest.param(-1, id="negative"),
        pytest.param(_LOGICAL_TILE_ITEMS + 1, id="beyond-logical-tile"),
    ),
)
def test_logical_runtime_valid_items_out_of_range_traps_in_isolated_context(
    operation: str, valid_items: int, *, items_per_thread
) -> None:
    valid_items = (
        valid_items
        if valid_items < 0
        else _LOGICAL_WARP_THREADS * items_per_thread + 1
    )
    result = _run_invalid_runtime_valid_items_probe(
        operation,
        valid_items,
        logical_width=_LOGICAL_WARP_THREADS,
        items_per_thread=items_per_thread,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output
