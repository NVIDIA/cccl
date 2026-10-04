# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Combine collective families in one kernel and reuse shared scratch.

Transpose Load, Store, and full/prefix CUB reductions share explicit scratch.
Their final Store checks that both reductions preserve the original payload.
The sort case adds Merge Sort and Scan to a repeated Load/Store pipeline.
Together they check provider registration and scratch reuse when several
families share one compilation.
"""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_mixed_primitives(api, items_per_thread):
    """Check both reductions and the later Store from the same loaded payload.

    The block root records both results. The prefix result uses the first
    item from each of the first 45 threads.
    The final copy verifies that the intervening reductions preserve payloads.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        copied: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        group = api.this_block()
        thread = group.rank()
        outputs = cute.make_tensor(observed, cute.make_layout(2))
        payload = api.ThreadData(items_per_thread)
        storage = api.TempStorage(sharing="shared", auto_sync=True)
        api.load(
            group, source, payload, algorithm="transpose", temp_storage=storage
        )
        total = api.sum(group, payload, temp_storage=storage)
        prefix = api.sum(
            group, payload[0], valid_items=45, temp_storage=storage
        )
        if thread == 0:
            outputs[0] = total
            outputs[1] = prefix
        api.store(
            group, copied, payload, algorithm="transpose", temp_storage=storage
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        copied: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, copied, observed, items_per_thread).launch(
            grid=1, block=(8, 4, 2)
        )

    source = values_for(np.int32, 64 * items_per_thread, shift=83)
    copied = np.zeros_like(source)
    observed = np.zeros(2, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(copied) as dst,
        device_array(observed) as out,
    ):
        launch(src, dst, out, items_per_thread)
    np.testing.assert_array_equal(copied, source)
    assert observed[0] == source.sum(dtype=np.int32)
    assert observed[1] == source[
        : 45 * items_per_thread : items_per_thread
    ].sum(dtype=np.int32)


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_sort_scan_shared_storage(api, items_per_thread):
    """Share one allocation across Load, MergeSort, Scan, and Store.

    Four runtime iterations process independent tiles with automatic reuse
    barriers. The host sorts each tile before computing exclusive prefixes,
    checking the whole pipeline across distinct provider scratch requirements.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        output: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        group = api.this_block()
        storage = api.TempStorage(alignment=128, auto_sync=True)
        for tile in range(tiles):
            keys = api.ThreadData(items_per_thread)
            api.load(
                group,
                source,
                keys,
                algorithm="transpose",
                offset=tile * 64 * items_per_thread,
                temp_storage=storage,
            )
            ordered = api.merge_sort_keys(group, keys, temp_storage=storage)
            prefixes = api.exclusive_sum(group, ordered, temp_storage=storage)
            api.store(
                group,
                output,
                prefixes,
                algorithm="transpose",
                offset=tile * 64 * items_per_thread,
                temp_storage=storage,
            )

    @cute.jit
    def launch(
        source: cute.Pointer,
        output: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, output, tiles, items_per_thread).launch(
            grid=1, block=(8, 4, 2)
        )

    source = values_for(np.int32, 256 * items_per_thread, shift=19)
    observed = np.zeros_like(source)
    expected = np.empty_like(source)
    for start in range(0, source.size, 64 * items_per_thread):
        ordered = np.sort(source[start : start + 64 * items_per_thread])
        expected[start : start + 64 * items_per_thread] = (
            np.cumsum(ordered) - ordered
        )
    with device_array(source) as src, device_array(observed) as dst:
        launch(src, dst, cutlass.Int32(4), items_per_thread)
    np.testing.assert_array_equal(observed, expected)
