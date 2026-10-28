# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stage a tile through shared memory and reload it with different ownership."""

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.memory import SmemAllocator

from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("reserved", (False, True), ids=("native", "reserved"))
def test_shared_source_and_destination(items_per_thread, reserved):
    import cuda.coop.cutlass as coop

    # docs: start cutlass-shared-load-store
    @cute.kernel
    def stage_and_reload(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
        reserved: cutlass.Constexpr,
    ):
        block = coop.this_block()
        tile_size = 64 * items_per_thread
        if cutlass.const_expr(reserved):
            workspace = coop.TempStorage(auto_sync=False)
            shared = workspace.reserve(tile_size, cutlass.Int32)
        else:
            shared = SmemAllocator().allocate_tensor(
                cutlass.Int32, cute.make_layout(tile_size)
            )

        values = coop.load(block, source, items_per_thread=items_per_thread)
        coop.store(block, shared, values)
        block.sync()

        # Striped reloads read values staged by other threads.
        reloaded = coop.load(
            block,
            shared,
            items_per_thread=items_per_thread,
            algorithm="striped",
        )
        coop.store(block, destination, reloaded)

        # A native register tensor receives values through its own store.
        registers = cute.make_rmem_tensor(items_per_thread, cutlass.Int32)
        register_alias = registers
        registers.store(reloaded.to_tensor_ssa())
        global_output = cute.make_tensor(
            destination, cute.make_layout(2 * tile_size)
        )
        for i in cutlass.range_constexpr(items_per_thread):
            index = block.rank() * items_per_thread + i
            global_output[tile_size + index] = register_alias[i]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
        reserved: cutlass.Constexpr,
    ):
        stage_and_reload(
            source, destination, items_per_thread, reserved
        ).launch(grid=1, block=64)

    # docs: end cutlass-shared-load-store
    tile_size = 64 * items_per_thread
    source = np.arange(tile_size, dtype=np.int32) * 3 - 17
    output = np.zeros(2 * tile_size, dtype=np.int32)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst, items_per_thread, reserved)
    expected = source.reshape(items_per_thread, 64).T.ravel()
    np.testing.assert_array_equal(output[:tile_size], expected)
    np.testing.assert_array_equal(output[tile_size:], expected)
