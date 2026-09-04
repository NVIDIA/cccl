# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

assert qualified_coop.__file__ is not None
_QUALIFIED_COOP_ORIGIN = Path(qualified_coop.__file__).resolve()
_SAFE_PATH_FLAG = "-P" if sys.version_info >= (3, 11) else "-I"

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_BLOCK_SHAPES = (_THREADS, (8, 4), (4, 4, 2))
_LOAD_OFFSET = 5
_STORE_OFFSET = 7
_GRID_STRIDE_BLOCKS = 3
_GRID_STRIDE_ITEMS = 7 * _TILE_ITEMS + 11

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


def _values(dtype: np.dtype, size: int, *, shift: int = 0) -> np.ndarray:
    values = (np.arange(size, dtype=np.int64) * 3 + shift) % 97
    if dtype.kind in {"i", "f"}:
        values = values - 48
    return values.astype(dtype)


def _sentinel(dtype: np.dtype) -> object:
    return dtype.type(211 if dtype.kind == "u" else -101)


@lru_cache(maxsize=None)
def _full_load_kernel(numba_dtype):
    @cuda.jit
    def kernel(source, observed):
        thread = cuda.threadIdx.x + cuda.blockDim.x * (
            cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
        )
        payload = root_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=numba_dtype,
        )
        loaded = root_coop.load(
            root_coop.this_block(),
            source,
            payload,
            algorithm="direct",
        )
        for item in range(_ITEMS_PER_THREAD):
            observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]

    return kernel


@lru_cache(maxsize=None)
def _full_store_kernel(numba_dtype):
    @cuda.jit
    def kernel(source, destination):
        thread = cuda.threadIdx.x + cuda.blockDim.x * (
            cuda.threadIdx.y + cuda.blockDim.y * cuda.threadIdx.z
        )
        payload = qualified_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=numba_dtype,
        )
        for item in range(_ITEMS_PER_THREAD):
            payload[item] = source[thread * _ITEMS_PER_THREAD + item]
        qualified_coop.store(
            qualified_coop.this_block(),
            destination,
            payload,
            algorithm=qualified_coop.BlockStoreAlgorithm.DIRECT,
        )

    return kernel


@pytest.mark.parametrize(("numpy_dtype", "numba_dtype"), _DTYPES)
def test_direct_load_matches_an_independent_oracle_for_every_dtype(
    numpy_dtype,
    numba_dtype,
):
    source = _values(numpy_dtype, _TILE_ITEMS)
    observed = np.full(_TILE_ITEMS, _sentinel(numpy_dtype), dtype=numpy_dtype)

    _full_load_kernel(numba_dtype)[1, _THREADS](source, observed)

    np.testing.assert_array_equal(observed, source)


@pytest.mark.parametrize(("numpy_dtype", "numba_dtype"), _DTYPES)
def test_direct_multi_item_store_matches_an_independent_oracle_for_every_dtype(
    numpy_dtype,
    numba_dtype,
):
    source = _values(numpy_dtype, _TILE_ITEMS, shift=11)
    destination = np.full(
        _TILE_ITEMS,
        _sentinel(numpy_dtype),
        dtype=numpy_dtype,
    )

    _full_store_kernel(numba_dtype)[1, _THREADS](source, destination)

    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("block_shape", _BLOCK_SHAPES, ids=("1d", "2d", "3d"))
def test_direct_load_uses_x_major_thread_order_for_exact_block_shape(block_shape):
    source = _values(np.dtype(np.int32), _TILE_ITEMS)
    observed = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _full_load_kernel(types.int32)[1, block_shape](source, observed)

    np.testing.assert_array_equal(observed, source)


@pytest.mark.parametrize("block_shape", _BLOCK_SHAPES, ids=("1d", "2d", "3d"))
def test_direct_store_uses_x_major_thread_order_for_exact_block_shape(block_shape):
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=7)
    destination = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _full_store_kernel(types.int32)[1, block_shape](source, destination)

    np.testing.assert_array_equal(destination, source)


@cuda.jit
def _load_preserving_invalid(source, observed, valid_items, source_offset):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    payload[0] = -17
    payload[1] = -17
    loaded = root_coop.load(
        root_coop.this_block(),
        source,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        offset=source_offset,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]


@pytest.mark.parametrize(
    "valid_items",
    (0, _TILE_ITEMS - 9, _TILE_ITEMS),
    ids=("zero", "partial", "full"),
)
def test_load_preserves_invalid_slots_and_applies_an_independent_offset(valid_items):
    source = _values(np.dtype(np.int32), _LOAD_OFFSET + _TILE_ITEMS)
    observed = np.full(_TILE_ITEMS, 37, dtype=np.int32)
    expected = np.full(_TILE_ITEMS, -17, dtype=np.int32)
    expected[:valid_items] = source[_LOAD_OFFSET : _LOAD_OFFSET + valid_items]

    _load_preserving_invalid[1, _THREADS](
        source,
        observed,
        np.int32(valid_items),
        np.int64(_LOAD_OFFSET),
    )

    np.testing.assert_array_equal(observed, expected)


@cuda.jit
def _load_defaulting_invalid(
    source,
    observed,
    valid_items,
    oob_default,
    source_offset,
):
    thread = cuda.threadIdx.x
    payload = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    payload[0] = 19
    payload[1] = 19
    loaded = qualified_coop.load(
        qualified_coop.this_block(),
        source,
        payload,
        algorithm=qualified_coop.BlockLoadAlgorithm.DIRECT,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=source_offset,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]


@pytest.mark.parametrize(
    "valid_items",
    (0, _TILE_ITEMS - 9),
    ids=("zero", "partial"),
)
def test_load_defaults_invalid_slots(valid_items):
    source = _values(np.dtype(np.int32), _LOAD_OFFSET + _TILE_ITEMS)
    observed = np.full(_TILE_ITEMS, 37, dtype=np.int32)
    expected = np.full(_TILE_ITEMS, -29, dtype=np.int32)
    expected[:valid_items] = source[_LOAD_OFFSET : _LOAD_OFFSET + valid_items]

    _load_defaulting_invalid[1, _THREADS](
        source,
        observed,
        np.int32(valid_items),
        np.int32(-29),
        np.int64(_LOAD_OFFSET),
    )

    np.testing.assert_array_equal(observed, expected)


@cuda.jit
def _store_valid_prefix(source, destination, valid_items, destination_offset):
    thread = cuda.threadIdx.x
    payload = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]
    qualified_coop.store(
        qualified_coop.this_block(),
        destination,
        payload,
        algorithm="direct",
        valid_items=valid_items,
        offset=destination_offset,
    )


@pytest.mark.parametrize(
    "valid_items",
    (0, _TILE_ITEMS - 9, _TILE_ITEMS),
    ids=("zero", "partial", "full"),
)
def test_store_writes_only_the_valid_prefix_at_an_independent_offset(valid_items):
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=17)
    destination = np.full(
        _STORE_OFFSET + _TILE_ITEMS + 3,
        -41,
        dtype=np.int32,
    )
    expected = destination.copy()
    expected[_STORE_OFFSET : _STORE_OFFSET + valid_items] = source[:valid_items]

    _store_valid_prefix[1, _THREADS](
        source,
        destination,
        np.int32(valid_items),
        np.int64(_STORE_OFFSET),
    )

    np.testing.assert_array_equal(destination, expected)


@cuda.jit
def _portable_grid_stride_load_store(source, destination):
    tile_offset = cuda.blockIdx.x * _TILE_ITEMS
    grid_stride = cuda.gridDim.x * _TILE_ITEMS
    while tile_offset < source.size:
        valid_items = source.size - tile_offset
        # Runtime counts trap rather than saturate. Clamp each grid-stride
        # remainder to the exact tile accepted by this block.
        if valid_items > _TILE_ITEMS:
            valid_items = _TILE_ITEMS
        payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        loaded = root_coop.load(
            root_coop.this_block(),
            source,
            payload,
            valid_items=valid_items,
            offset=tile_offset,
        )
        root_coop.store(
            root_coop.this_block(),
            destination,
            loaded,
            valid_items=valid_items,
            offset=tile_offset,
        )
        tile_offset += grid_stride


@cuda.jit
def _qualified_grid_stride_load_store(source, destination):
    tile_offset = cuda.blockIdx.x * _TILE_ITEMS
    grid_stride = cuda.gridDim.x * _TILE_ITEMS
    while tile_offset < source.size:
        valid_items = source.size - tile_offset
        # Runtime counts trap rather than saturate. Clamp each grid-stride
        # remainder to the exact tile accepted by this block.
        if valid_items > _TILE_ITEMS:
            valid_items = _TILE_ITEMS
        payload = qualified_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        loaded = qualified_coop.load(
            qualified_coop.this_block(),
            source,
            payload,
            valid_items=valid_items,
            offset=tile_offset,
        )
        qualified_coop.store(
            qualified_coop.this_block(),
            destination,
            loaded,
            valid_items=valid_items,
            offset=tile_offset,
        )
        tile_offset += grid_stride


@pytest.mark.parametrize(
    "kernel",
    (
        pytest.param(_portable_grid_stride_load_store, id="portable"),
        pytest.param(_qualified_grid_stride_load_store, id="qualified"),
    ),
)
def test_multi_block_grid_stride_load_store_handles_a_partial_tail(kernel):
    source = _values(np.dtype(np.int32), _GRID_STRIDE_ITEMS, shift=19)
    destination = np.full(_GRID_STRIDE_ITEMS, -1, dtype=np.int32)

    kernel[_GRID_STRIDE_BLOCKS, _THREADS](source, destination)

    np.testing.assert_array_equal(destination, source)


def _run_invalid_runtime_valid_items_probe(
    operation: str,
    valid_items: int,
) -> subprocess.CompletedProcess[str]:
    # A device trap poisons its CUDA context, so invalid launches must run in
    # disposable child processes rather than the pytest worker.
    operation_body = textwrap.indent(
        textwrap.dedent(
            {
                "load": """
            payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
            root_coop.load(
                root_coop.this_block(),
                source,
                payload,
                valid_items=valid_items,
            )
        """,
                "store": """
            payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            root_coop.store(
                root_coop.this_block(),
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

import cuda.coop.numba_mlir as _coop_numba_mlir
from cuda import coop as root_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(_coop_numba_mlir.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"trap probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

_THREADS = {_THREADS}
_ITEMS_PER_THREAD = {_ITEMS_PER_THREAD}

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


@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize(
    "valid_items",
    (
        pytest.param(-1, id="negative"),
        pytest.param(_TILE_ITEMS + 1, id="beyond-tile"),
    ),
)
def test_runtime_valid_items_out_of_range_traps_in_an_isolated_context(
    operation,
    valid_items,
):
    result = _run_invalid_runtime_valid_items_probe(operation, valid_items)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    # Drivers report PTX ``trap;`` as either an illegal instruction or the
    # more general launch failure. Both are deterministic device-side faults.
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output

    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=23)
    observed = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    _full_load_kernel(types.int32)[1, _THREADS](source, observed)
    np.testing.assert_array_equal(observed, source)


@cuda.jit
def _portable_scalar_store(source, destination):
    thread = cuda.threadIdx.x
    root_coop.store(
        root_coop.this_block(),
        destination,
        source[thread],
        algorithm="direct",
    )


def test_portable_scalar_store_matches_an_independent_oracle():
    source = _values(np.dtype(np.int32), _THREADS, shift=23)
    destination = np.full(_THREADS, -1, dtype=np.int32)

    _portable_scalar_store[1, _THREADS](source, destination)

    np.testing.assert_array_equal(destination, source)


@cuda.jit
def _qualified_untyped_load(source, observed):
    thread = cuda.threadIdx.x
    payload = qualified_coop.ThreadData(_ITEMS_PER_THREAD)
    loaded = qualified_coop.load(
        qualified_coop.this_block(),
        source,
        payload,
        algorithm=qualified_coop.BlockLoadAlgorithm.DIRECT,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]


def test_qualified_load_infers_an_untyped_payload():
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=29)
    observed = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _qualified_untyped_load[1, _THREADS](source, observed)

    np.testing.assert_array_equal(observed, source)


@lru_cache(maxsize=None)
def _storage_load_kernel(storage_mode: str):
    if storage_mode == "implicit":

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = qualified_coop.load(
                qualified_coop.this_block(),
                source,
                payload,
                algorithm="direct",
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]

    elif storage_mode == "shared":

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(sharing="shared")
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = qualified_coop.load(
                qualified_coop.this_block(),
                source,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]

    elif storage_mode == "exclusive":

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(
                4096,
                alignment=16,
                sharing="exclusive",
            )
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = qualified_coop.load(
                qualified_coop.this_block(),
                source,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]

    else:

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(128 * 1024, alignment=16)
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = qualified_coop.load(
                qualified_coop.this_block(),
                source,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]

    return kernel


@lru_cache(maxsize=None)
def _storage_store_kernel(storage_mode: str):
    if storage_mode == "implicit":

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            qualified_coop.store(
                qualified_coop.this_block(),
                destination,
                payload,
                algorithm="direct",
            )

    elif storage_mode == "shared":

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(sharing="shared")
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            qualified_coop.store(
                qualified_coop.this_block(),
                destination,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )

    elif storage_mode == "exclusive":

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(
                4096,
                alignment=16,
                sharing="exclusive",
            )
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            qualified_coop.store(
                qualified_coop.this_block(),
                destination,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )

    else:

        @cuda.jit
        def kernel(source, destination):
            thread = cuda.threadIdx.x
            storage = qualified_coop.TempStorage(128 * 1024, alignment=16)
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            qualified_coop.store(
                qualified_coop.this_block(),
                destination,
                payload,
                algorithm="direct",
                temp_storage=storage,
            )

    return kernel


@pytest.mark.parametrize(
    "storage_mode", ("implicit", "shared", "exclusive", "oversized")
)
def test_direct_load_accepts_each_temp_storage_descriptor_mode(storage_mode):
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=31)
    observed = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _storage_load_kernel(storage_mode)[1, _THREADS](source, observed)

    np.testing.assert_array_equal(observed, source)


@pytest.mark.parametrize(
    "storage_mode", ("implicit", "shared", "exclusive", "oversized")
)
def test_direct_store_accepts_each_temp_storage_descriptor_mode(storage_mode):
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=37)
    destination = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _storage_store_kernel(storage_mode)[1, _THREADS](source, destination)

    np.testing.assert_array_equal(destination, source)


@cuda.jit
def _divergent_direct_load_store(source, destination, observed):
    thread = cuda.threadIdx.x
    if thread & 1:
        payload = root_coop.ThreadData(1, dtype=types.int32)
        loaded = root_coop.load(root_coop.this_block(), source, payload)
        root_coop.store(root_coop.this_block(), destination, loaded)
        observed[thread] = loaded[0]


def test_direct_load_store_are_safe_in_divergent_control_flow():
    source = _values(np.dtype(np.int32), _THREADS, shift=43)
    destination = np.full(_THREADS, -1, dtype=np.int32)
    observed = np.full(_THREADS, -1, dtype=np.int32)
    expected = np.full(_THREADS, -1, dtype=np.int32)
    expected[1::2] = source[1::2]

    _divergent_direct_load_store[1, _THREADS](source, destination, observed)

    np.testing.assert_array_equal(destination, expected)
    np.testing.assert_array_equal(observed, expected)


@cuda.jit
def _repeated_shared_load(source_a, source_b, observed_a, observed_b):
    thread = cuda.threadIdx.x
    storage = qualified_coop.TempStorage(sharing="shared")
    payload_a = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    payload_b = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    loaded_a = qualified_coop.load(
        qualified_coop.this_block(),
        source_a,
        payload_a,
        algorithm="direct",
        temp_storage=storage,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed_a[thread * _ITEMS_PER_THREAD + item] = loaded_a[item]
    loaded_b = qualified_coop.load(
        qualified_coop.this_block(),
        source_b,
        payload_b,
        algorithm="direct",
        temp_storage=storage,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed_b[thread * _ITEMS_PER_THREAD + item] = loaded_b[item]


def test_direct_load_accepts_one_shared_descriptor_for_repeated_calls():
    source_a = _values(np.dtype(np.int32), _TILE_ITEMS, shift=41)
    source_b = _values(np.dtype(np.int32), _TILE_ITEMS, shift=53)
    observed_a = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    observed_b = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _repeated_shared_load[1, _THREADS](
        source_a,
        source_b,
        observed_a,
        observed_b,
    )

    np.testing.assert_array_equal(observed_a, source_a)
    np.testing.assert_array_equal(observed_b, source_b)


@cuda.jit
def _manual_sync_shared_load(source_a, source_b, observed_a, observed_b):
    thread = cuda.threadIdx.x
    storage = qualified_coop.TempStorage(
        sharing="shared",
        auto_sync=False,
    )
    payload_a = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    payload_b = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    loaded_a = qualified_coop.load(
        qualified_coop.this_block(),
        source_a,
        payload_a,
        algorithm="direct",
        temp_storage=storage,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed_a[thread * _ITEMS_PER_THREAD + item] = loaded_a[item]
    cuda.syncthreads()
    loaded_b = qualified_coop.load(
        qualified_coop.this_block(),
        source_b,
        payload_b,
        algorithm="direct",
        temp_storage=storage,
    )
    for item in range(_ITEMS_PER_THREAD):
        observed_b[thread * _ITEMS_PER_THREAD + item] = loaded_b[item]


def test_direct_load_allows_manual_sync_with_an_unused_shared_descriptor():
    source_a = _values(np.dtype(np.int32), _TILE_ITEMS, shift=71)
    source_b = _values(np.dtype(np.int32), _TILE_ITEMS, shift=73)
    observed_a = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    observed_b = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _manual_sync_shared_load[1, _THREADS](
        source_a,
        source_b,
        observed_a,
        observed_b,
    )

    np.testing.assert_array_equal(observed_a, source_a)
    np.testing.assert_array_equal(observed_b, source_b)


@cuda.jit
def _repeated_exclusive_load(source_a, source_b, observed_a, observed_b):
    thread = cuda.threadIdx.x
    storage = qualified_coop.TempStorage(sharing="exclusive")
    payload_a = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    payload_b = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    loaded_a = qualified_coop.load(
        qualified_coop.this_block(),
        source_a,
        payload_a,
        algorithm="direct",
        temp_storage=storage,
    )
    loaded_b = qualified_coop.load(
        qualified_coop.this_block(),
        source_b,
        payload_b,
        algorithm="direct",
        temp_storage=storage,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = thread * _ITEMS_PER_THREAD + item
        observed_a[index] = loaded_a[item]
        observed_b[index] = loaded_b[item]


def test_direct_load_accepts_one_exclusive_descriptor_for_repeated_calls():
    source_a = _values(np.dtype(np.int32), _TILE_ITEMS, shift=79)
    source_b = _values(np.dtype(np.int32), _TILE_ITEMS, shift=83)
    observed_a = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    observed_b = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _repeated_exclusive_load[1, _THREADS](
        source_a,
        source_b,
        observed_a,
        observed_b,
    )

    np.testing.assert_array_equal(observed_a, source_a)
    np.testing.assert_array_equal(observed_b, source_b)


@cuda.jit
def _looped_shared_store(source, destination, observed):
    thread = cuda.threadIdx.x
    storage = qualified_coop.TempStorage(sharing="shared")
    for iteration in range(2):
        payload = qualified_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        tile_offset = iteration * _TILE_ITEMS
        for item in range(_ITEMS_PER_THREAD):
            payload[item] = source[tile_offset + thread * _ITEMS_PER_THREAD + item]
        qualified_coop.store(
            qualified_coop.this_block(),
            destination,
            payload,
            algorithm="direct",
            temp_storage=storage,
        )
        for item in range(_ITEMS_PER_THREAD):
            index = thread * _ITEMS_PER_THREAD + item
            observed[tile_offset + index] = destination[index]


def test_direct_store_accepts_one_shared_descriptor_in_a_loop():
    source = _values(np.dtype(np.int32), 2 * _TILE_ITEMS, shift=59)
    destination = np.full(_TILE_ITEMS, -1, dtype=np.int32)
    observed = np.full(2 * _TILE_ITEMS, -1, dtype=np.int32)

    _looped_shared_store[1, _THREADS](source, destination, observed)

    np.testing.assert_array_equal(observed, source)
    np.testing.assert_array_equal(destination, source[_TILE_ITEMS:])


@cuda.jit(device=True, inline=True)
def _inlined_portable_load(source, observed):
    thread = cuda.threadIdx.x
    payload = root_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = root_coop.load(
        root_coop.this_block(),
        source,
        payload,
        algorithm="direct",
    )
    for item in range(_ITEMS_PER_THREAD):
        observed[thread * _ITEMS_PER_THREAD + item] = loaded[item]


@cuda.jit
def _load_through_inlined_device_helper(source, observed):
    _inlined_portable_load(source, observed)


def test_portable_load_is_planned_after_device_helper_inlining():
    source = _values(np.dtype(np.int32), _TILE_ITEMS, shift=67)
    observed = np.full(_TILE_ITEMS, -1, dtype=np.int32)

    _load_through_inlined_device_helper[1, _THREADS](source, observed)

    np.testing.assert_array_equal(observed, source)
