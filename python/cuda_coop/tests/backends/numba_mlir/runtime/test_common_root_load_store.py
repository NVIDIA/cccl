# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_WARP_ITEMS = 32 * _ITEMS_PER_THREAD
_BLOCK_VALID_ITEMS = _TILE_ITEMS - 9
_WARP_VALID_ITEMS = _WARP_ITEMS - 5
_LOAD_OFFSET = 3
_STORE_OFFSET = 5
_OOB_DEFAULT = -7
_SENTINEL = -999
_COMPLEX_SENTINEL = complex(-999.0, 999.0)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_root_load_store_kernel(
    d_input,
    d_block_loaded_out,
    d_block_full_out,
    d_block_partial_out_a,
    d_block_partial_out_b,
    d_warp_loaded_out,
    d_warp_full_out,
    d_warp_partial_out_a,
    d_warp_partial_out_b,
):
    block = coop.this_block()
    storage = coop.TempStorage()
    block_full_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded_block_full = coop.load(
        block,
        d_input,
        block_full_items,
        algorithm="transpose",
        temp_storage=storage,
    )
    block_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded_block = coop.load(
        block,
        d_input,
        block_items,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        oob_default=_OOB_DEFAULT,
        offset=_LOAD_OFFSET,
        temp_storage=storage,
    )
    coop.store(
        block,
        d_block_full_out,
        loaded_block_full,
        algorithm="transpose",
        temp_storage=storage,
    )
    coop.store(
        block,
        d_block_partial_out_a,
        loaded_block,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        offset=_STORE_OFFSET,
        temp_storage=storage,
    )
    tid = cuda.threadIdx.x
    # Store is specified as non-mutating at the group-first boundary. Observe
    # the payload after both a full and a partial transposing Store.
    block_begin = tid * _ITEMS_PER_THREAD
    d_block_loaded_out[block_begin] = block_items[0]
    d_block_loaded_out[block_begin + 1] = block_items[1]
    coop.store(
        block,
        d_block_partial_out_b,
        loaded_block,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        offset=_STORE_OFFSET,
        temp_storage=storage,
    )

    # Deferred warp storage is not portable across the certified backends.
    warp = coop.this_warp()
    warp_full_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded_warp_full = coop.load(
        warp,
        d_input,
        warp_full_items,
        algorithm="transpose",
    )
    warp_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded_warp = coop.load(
        warp,
        d_input,
        warp_items,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        oob_default=_OOB_DEFAULT,
        offset=_LOAD_OFFSET,
    )
    coop.store(
        warp,
        d_warp_full_out,
        loaded_warp_full,
        algorithm="transpose",
    )
    coop.store(
        warp,
        d_warp_partial_out_a,
        loaded_warp,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        offset=_STORE_OFFSET,
    )
    warp_id = tid // 32
    lane = tid - warp_id * 32
    warp_begin = warp_id * _WARP_ITEMS
    warp_thread_begin = warp_begin + lane * _ITEMS_PER_THREAD
    d_warp_loaded_out[warp_thread_begin] = warp_items[0]
    d_warp_loaded_out[warp_thread_begin + 1] = warp_items[1]
    coop.store(
        warp,
        d_warp_partial_out_b,
        loaded_warp,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        offset=_STORE_OFFSET,
    )


@pytest.mark.evidence_for("group.load", backend="numba_mlir", evidence="runtime")
@pytest.mark.evidence_for("group.store", backend="numba_mlir", evidence="runtime")
def test_common_root_load_store_runs_for_block_and_physical_warp(
    numba_mlir_cuda_available,
) -> None:
    del numba_mlir_cuda_available
    values = np.arange(_TILE_ITEMS, dtype=np.int32)
    outputs = [np.full_like(values, _SENTINEL) for _ in range(8)]

    _common_root_load_store_kernel[1, _BLOCK_THREADS](
        values,
        *outputs,
    )
    cuda.synchronize()

    expected_block_loaded = np.full_like(values, _OOB_DEFAULT)
    expected_block_loaded[:_BLOCK_VALID_ITEMS] = values[
        _LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS
    ]
    expected_block_partial = np.full_like(values, _SENTINEL)
    expected_block_partial[_STORE_OFFSET : _STORE_OFFSET + _BLOCK_VALID_ITEMS] = values[
        _LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS
    ]

    expected_warp_loaded = np.full_like(values, _OOB_DEFAULT)
    expected_warp_partial = np.full_like(values, _SENTINEL)
    for warp_begin in range(0, _TILE_ITEMS, _WARP_ITEMS):
        expected_warp_loaded[warp_begin : warp_begin + _WARP_VALID_ITEMS] = values[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]
        expected_warp_partial[
            warp_begin + _STORE_OFFSET : warp_begin + _STORE_OFFSET + _WARP_VALID_ITEMS
        ] = values[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]

    for output, expected in zip(
        outputs,
        (
            expected_block_loaded,
            values,
            expected_block_partial,
            expected_block_partial,
            expected_warp_loaded,
            values,
            expected_warp_partial,
            expected_warp_partial,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(output, expected)


@cuda.jit
def _qualified_complex_load_store_kernel(
    d_input,
    d_block_output,
    d_block_loaded,
    d_warp_output,
    d_warp_loaded,
):
    tid = cuda.threadIdx.x

    block = numba_coop.this_block()
    storage = numba_coop.TempStorage()
    block_items = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.complex128,
    )
    loaded_block = numba_coop.load(
        block,
        d_input,
        block_items,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        oob_default=d_input[0],
        offset=_LOAD_OFFSET,
        temp_storage=storage,
    )
    numba_coop.store(
        block,
        d_block_output,
        loaded_block,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        offset=_STORE_OFFSET,
        temp_storage=storage,
    )
    block_begin = tid * _ITEMS_PER_THREAD
    d_block_loaded[block_begin] = block_items[0]
    d_block_loaded[block_begin + 1] = block_items[1]

    warp = numba_coop.this_warp()
    warp_items = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.complex128,
    )
    loaded_warp = numba_coop.load(
        warp,
        d_input,
        warp_items,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        oob_default=d_input[0],
        offset=_LOAD_OFFSET,
    )
    numba_coop.store(
        warp,
        d_warp_output,
        loaded_warp,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        offset=_STORE_OFFSET,
    )
    warp_id = tid // 32
    lane = tid - warp_id * 32
    warp_begin = warp_id * _WARP_ITEMS
    warp_thread_begin = warp_begin + lane * _ITEMS_PER_THREAD
    d_warp_loaded[warp_thread_begin] = warp_items[0]
    d_warp_loaded[warp_thread_begin + 1] = warp_items[1]


def test_qualified_load_store_supports_complex128_aggregate_payloads(
    numba_mlir_cuda_available,
) -> None:
    del numba_mlir_cuda_available
    real = np.arange(_TILE_ITEMS, dtype=np.float64)
    values = (real + 1j * (real + 0.5)).astype(np.complex128)
    outputs = [np.full_like(values, _COMPLEX_SENTINEL) for _ in range(4)]

    _qualified_complex_load_store_kernel[1, _BLOCK_THREADS](values, *outputs)
    cuda.synchronize()

    expected_block_loaded = np.full_like(values, values[0])
    expected_block_loaded[:_BLOCK_VALID_ITEMS] = values[
        _LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS
    ]
    expected_block_output = np.full_like(values, _COMPLEX_SENTINEL)
    expected_block_output[_STORE_OFFSET : _STORE_OFFSET + _BLOCK_VALID_ITEMS] = values[
        _LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS
    ]

    expected_warp_loaded = np.full_like(values, values[0])
    expected_warp_output = np.full_like(values, _COMPLEX_SENTINEL)
    for warp_begin in range(0, _TILE_ITEMS, _WARP_ITEMS):
        expected_warp_loaded[warp_begin : warp_begin + _WARP_VALID_ITEMS] = values[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]
        expected_warp_output[
            warp_begin + _STORE_OFFSET : warp_begin + _STORE_OFFSET + _WARP_VALID_ITEMS
        ] = values[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]

    for output, expected in zip(
        outputs,
        (
            expected_block_output,
            expected_block_loaded,
            expected_warp_output,
            expected_warp_loaded,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(output, expected)
