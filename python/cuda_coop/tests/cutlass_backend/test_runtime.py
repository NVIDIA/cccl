# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPU runtime tests for block Load and Store behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cuda.coop.cutlass", exc_type=ImportError)
torch = pytest.importorskip("torch")

from cuda.coop.cutlass import _provider  # noqa: E402

from ._compile_support import EXAMPLE_PATH  # noqa: E402


def test_executable_partial_copy_preserves_prefix_and_tail_sentinels() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")

    spec = importlib.util.spec_from_file_location(
        "cuda_coop_block_load_store_runtime",
        EXAMPLE_PATH,
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    values = example.run_example("root")
    artifacts = {
        path.name: path.read_bytes()
        for path in Path(_provider._ARTIFACTS.name).glob("*.ltoir")
    }
    assert artifacts
    assert example.run_example("root") == values
    assert example.run_example("qualified") == values
    assert {
        path.name: path.read_bytes()
        for path in Path(_provider._ARTIFACTS.name).glob("*.ltoir")
    } == artifacts
    assert values[: example.OUTPUT_OFFSET] == [example.SENTINEL] * (
        example.OUTPUT_OFFSET
    )
    assert values[
        example.OUTPUT_OFFSET + example.LOAD_VALID_ITEMS : example.OUTPUT_OFFSET
        + example.STORE_VALID_ITEMS
    ] == [example.OOB_DEFAULT] * (example.STORE_VALID_ITEMS - example.LOAD_VALID_ITEMS)
    assert values[example.OUTPUT_OFFSET + example.STORE_VALID_ITEMS :] == [
        example.SENTINEL
    ] * (len(values) - example.OUTPUT_OFFSET - example.STORE_VALID_ITEMS)


def test_full_tile_copy_uses_every_thread_item() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")

    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    from cuda import coop

    globals()["cute"] = cute

    block_threads = 32
    items_per_thread = 2
    tile_items = block_threads * items_per_thread
    sentinel = -999

    @cute.kernel
    def _full_copy(source: cute.Tensor, destination: cute.Tensor):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items)
        coop.store(block, destination, items)

    @cute.jit
    def _run(source: cute.Tensor, destination: cute.Tensor):
        _full_copy(source, destination).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
        )

    cutlass.cuda.initialize_cuda_context()
    source = torch.arange(tile_items, dtype=torch.int32, device="cuda")
    destination = torch.full(
        (tile_items + 4,),
        sentinel,
        dtype=torch.int32,
        device="cuda",
    )
    _run(from_dlpack(source), from_dlpack(destination))
    torch.cuda.synchronize()

    torch.testing.assert_close(destination[:tile_items], source, atol=0, rtol=0)
    assert destination[tile_items:].cpu().tolist() == [sentinel] * 4


def test_two_dimensional_uint8_copy_and_literal_store() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")

    import cutlass
    from cutlass import cute
    from cutlass.base_dsl.typing import Uint8
    from cutlass.cute.runtime import from_dlpack

    from cuda import coop

    globals()["cute"] = cute

    block_dim = (8, 4, 1)
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    items_per_thread = 2
    tile_items = block_threads * items_per_thread
    literal_values = (7, 250)

    @cute.kernel
    def _copy_and_store_literals(
        source: cute.Tensor,
        destination: cute.Tensor,
        literal_destination: cute.Tensor,
    ):
        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(block, source, items)
        coop.store(block, destination, items)

        literals = coop.ThreadData(items_per_thread, dtype=Uint8)
        literals[0] = literal_values[0]
        literals[1] = literal_values[1]
        coop.store(block, literal_destination, literals)

    @cute.jit
    def _run(
        source: cute.Tensor,
        destination: cute.Tensor,
        literal_destination: cute.Tensor,
    ):
        _copy_and_store_literals(
            source,
            destination,
            literal_destination,
        ).launch(grid=(1, 1, 1), block=block_dim)

    cutlass.cuda.initialize_cuda_context()
    source = torch.arange(tile_items, dtype=torch.uint8, device="cuda")
    destination = torch.empty_like(source)
    literal_destination = torch.empty_like(source)

    _run(
        from_dlpack(source),
        from_dlpack(destination),
        from_dlpack(literal_destination),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(destination, source, atol=0, rtol=0)
    assert literal_destination.cpu().tolist() == list(literal_values) * block_threads


def test_nested_compact_copy_uses_runtime_controls_and_literal_items() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-capable PyTorch runtime")

    import cutlass
    from cutlass import cute
    from cutlass.base_dsl.typing import Int32, Int64
    from cutlass.cute.runtime import from_dlpack

    from cuda import coop

    globals().update(
        {
            "Int32": Int32,
            "Int64": Int64,
            "cute": cute,
        }
    )

    block_threads = 32
    items_per_thread = 2
    tile_items = block_threads * items_per_thread
    operand_items = 80
    sentinel = -999
    literal_values = (7, -3)

    @cute.kernel
    def _runtime_copy(
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
    def _run(
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
        _runtime_copy(
            source_view,
            destination_view,
            literal_destination,
            load_valid_items,
            load_oob_default,
            load_offset,
            store_valid_items,
            store_offset,
        ).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
        )

    cutlass.cuda.initialize_cuda_context()
    source_host = torch.arange(operand_items, dtype=torch.int32)
    source = source_host.cuda()
    destination = torch.full(
        (operand_items,),
        sentinel,
        dtype=torch.int32,
        device="cuda",
    )
    literal_destination = torch.empty(
        (tile_items,),
        dtype=torch.int32,
        device="cuda",
    )
    load_valid_items = tile_items - 7
    load_oob_default = -17
    load_offset = 3
    store_valid_items = tile_items - 3
    store_offset = 5

    _run(
        from_dlpack(source),
        from_dlpack(destination),
        from_dlpack(literal_destination),
        load_valid_items,
        load_oob_default,
        load_offset,
        store_valid_items,
        store_offset,
    )
    torch.cuda.synchronize()

    expected = torch.full((operand_items,), sentinel, dtype=torch.int32)
    expected[store_offset : store_offset + load_valid_items] = source_host[
        load_offset : load_offset + load_valid_items
    ]
    expected[store_offset + load_valid_items : store_offset + store_valid_items] = (
        load_oob_default
    )
    torch.testing.assert_close(destination.cpu(), expected, atol=0, rtol=0)
    assert literal_destination.cpu().tolist() == list(literal_values) * block_threads
