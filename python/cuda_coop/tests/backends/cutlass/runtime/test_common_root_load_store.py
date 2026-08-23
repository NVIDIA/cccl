# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from cuda import coop

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")
# Module-level names so the DSL can evaluate stringized kernel annotations.
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32
Int64 = pytest.importorskip("cutlass.base_dsl.typing").Int64

from_dlpack = runtime.from_dlpack

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


@cute.kernel
def _common_root_load_store_kernel(
    values_in: cute.Tensor,
    block_loaded_out: cute.Tensor,
    block_partial_out: cute.Tensor,
    warp_loaded_out: cute.Tensor,
    warp_partial_out: cute.Tensor,
):
    block = coop.this_block()
    storage = coop.TempStorage()
    block_items = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded_block = coop.load(
        block,
        values_in,
        block_items,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        oob_default=_OOB_DEFAULT,
        offset=_LOAD_OFFSET,
        temp_storage=storage,
    )
    tidx, _, _ = cute.arch.thread_idx()
    # Observe the populated output independently of the partial Store below.
    block_begin = tidx * _ITEMS_PER_THREAD
    block_loaded_out[block_begin] = block_items[0]
    block_loaded_out[block_begin + 1] = block_items[1]
    coop.store(
        block,
        block_partial_out,
        loaded_block,
        algorithm="transpose",
        valid_items=_BLOCK_VALID_ITEMS,
        offset=_STORE_OFFSET,
        temp_storage=storage,
    )

    # Deferred warp storage is not portable across the certified backends.
    warp = coop.this_warp()
    warp_items = coop.ThreadData(_ITEMS_PER_THREAD)
    loaded_warp = coop.load(
        warp,
        values_in,
        warp_items,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        oob_default=_OOB_DEFAULT,
        offset=_LOAD_OFFSET,
    )
    warp_id = tidx // 32
    lane = tidx - warp_id * 32
    warp_begin = warp_id * _WARP_ITEMS
    warp_thread_begin = warp_begin + lane * _ITEMS_PER_THREAD
    warp_loaded_out[warp_thread_begin] = warp_items[0]
    warp_loaded_out[warp_thread_begin + 1] = warp_items[1]
    coop.store(
        warp,
        warp_partial_out,
        loaded_warp,
        algorithm="transpose",
        valid_items=_WARP_VALID_ITEMS,
        offset=_STORE_OFFSET,
    )


@cute.jit
def _run_common_root_load_store(
    values_in: cute.Tensor,
    block_loaded_out: cute.Tensor,
    block_partial_out: cute.Tensor,
    warp_loaded_out: cute.Tensor,
    warp_partial_out: cute.Tensor,
):
    _common_root_load_store_kernel(
        values_in,
        block_loaded_out,
        block_partial_out,
        warp_loaded_out,
        warp_partial_out,
    ).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


@pytest.mark.evidence_for("group.load", backend="cutlass", evidence="runtime")
@pytest.mark.evidence_for("group.store", backend="cutlass", evidence="runtime")
def test_common_root_load_store_runs_for_block_and_physical_warp(
    cutlass_runtime_available,
) -> None:
    del cutlass_runtime_available
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(_TILE_ITEMS, dtype=torch.int32)
    values_in = values_host.cuda()
    outputs = [
        torch.full(
            (_TILE_ITEMS,),
            _SENTINEL,
            dtype=torch.int32,
            device="cuda",
        )
        for _ in range(4)
    ]

    _run_common_root_load_store(
        from_dlpack(values_in),
        *(from_dlpack(output) for output in outputs),
    )
    torch.cuda.synchronize()

    expected_block_loaded = torch.full_like(values_host, _OOB_DEFAULT)
    expected_block_loaded[:_BLOCK_VALID_ITEMS] = values_host[
        _LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS
    ]
    expected_block_partial = torch.full_like(values_host, _SENTINEL)
    expected_block_partial[_STORE_OFFSET : _STORE_OFFSET + _BLOCK_VALID_ITEMS] = (
        values_host[_LOAD_OFFSET : _LOAD_OFFSET + _BLOCK_VALID_ITEMS]
    )

    expected_warp_loaded = torch.full_like(values_host, _OOB_DEFAULT)
    expected_warp_partial = torch.full_like(values_host, _SENTINEL)
    for warp_begin in range(0, _TILE_ITEMS, _WARP_ITEMS):
        expected_warp_loaded[warp_begin : warp_begin + _WARP_VALID_ITEMS] = values_host[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]
        expected_warp_partial[
            warp_begin + _STORE_OFFSET : warp_begin + _STORE_OFFSET + _WARP_VALID_ITEMS
        ] = values_host[
            warp_begin + _LOAD_OFFSET : warp_begin + _LOAD_OFFSET + _WARP_VALID_ITEMS
        ]

    for output, expected in zip(
        outputs,
        (
            expected_block_loaded,
            expected_block_partial,
            expected_warp_loaded,
            expected_warp_partial,
        ),
        strict=True,
    ):
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


def test_nested_compact_operands_and_declared_literals_run_with_runtime_controls(
    cutlass_runtime_available,
) -> None:
    del cutlass_runtime_available

    block_threads = 32
    items_per_thread = 2
    tile_items = block_threads * items_per_thread
    operand_items = 80
    sentinel = -999
    literal_values = (7, -3)

    @cute.kernel
    def copy_kernel(
        source: cute.Tensor,
        destination: cute.Tensor,
        literal_destination: cute.Tensor,
        load_valid_items: Int32,
        load_oob_default: Int32,
        load_offset: Int64,
        store_valid_items: Int32,
        store_offset: Int64,
    ):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(
            block,
            source,
            items,
            valid_items=load_valid_items,
            oob_default=load_oob_default,
            offset=load_offset,
        )
        coop.store(
            block,
            destination,
            items,
            valid_items=store_valid_items,
            offset=store_offset,
        )

        literals = coop.ThreadData(items_per_thread, dtype=Int32)
        literals[0] = literal_values[0]
        literals[1] = literal_values[1]
        coop.store(block, literal_destination, literals)

    @cute.jit
    def run(
        source: cute.Tensor,
        destination: cute.Tensor,
        literal_destination: cute.Tensor,
        load_valid_items: Int32,
        load_oob_default: Int32,
        load_offset: Int64,
        store_valid_items: Int32,
        store_offset: Int64,
    ):
        layout = cute.make_layout(((8, 5), 2), stride=((1, 8), 40))
        source_view = cute.make_tensor(source.iterator, layout)
        destination_view = cute.make_tensor(destination.iterator, layout)
        copy_kernel(
            source_view,
            destination_view,
            literal_destination,
            load_valid_items,
            load_oob_default,
            load_offset,
            store_valid_items,
            store_offset,
        ).launch(grid=(1, 1, 1), block=(block_threads, 1, 1))

    cutlass.cuda.initialize_cuda_context()
    source_host = torch.arange(operand_items, dtype=torch.int32)
    source = source_host.cuda()
    destination = torch.full_like(source, sentinel)
    literal_destination = torch.full(
        (tile_items,),
        sentinel,
        dtype=torch.int32,
        device="cuda",
    )
    load_valid_items = 51
    load_oob_default = -17
    load_offset = 5
    store_valid_items = 47
    store_offset = 7

    run(
        from_dlpack(source),
        from_dlpack(destination),
        from_dlpack(literal_destination),
        Int32(load_valid_items),
        Int32(load_oob_default),
        Int64(load_offset),
        Int32(store_valid_items),
        Int64(store_offset),
    )
    torch.cuda.synchronize()

    expected = torch.full_like(source_host, sentinel)
    expected[store_offset : store_offset + store_valid_items] = source_host[
        load_offset : load_offset + store_valid_items
    ]
    torch.testing.assert_close(destination.cpu(), expected, atol=0, rtol=0)
    assert literal_destination.cpu().tolist() == list(literal_values) * block_threads
