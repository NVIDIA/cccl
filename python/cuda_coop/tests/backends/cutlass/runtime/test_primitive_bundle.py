# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Movement, hierarchy, and both reduction routes in one compiler trace."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_mixed_primitives(api):
    @cute.kernel
    def kernel(source: cute.Pointer, copied: cute.Pointer, observed: cute.Pointer):
        group = api.this_block()
        thread = group.rank()
        outputs = cute.make_tensor(observed, cute.make_layout(65))
        payload = api.ThreadData(2)
        storage = api.TempStorage(sharing="shared")
        api.load(group, source, payload, algorithm="transpose", temp_storage=storage)
        outputs[thread] = api.sum(group, payload)
        prefix = api.sum(group, payload[0], broadcast=False, valid_items=45)
        if thread == 0:
            outputs[64] = prefix
        api.store(group, copied, payload, algorithm="transpose", temp_storage=storage)

    @cute.jit
    def launch(source: cute.Pointer, copied: cute.Pointer, observed: cute.Pointer):
        kernel(source, copied, observed).launch(grid=1, block=(8, 4, 2))

    source = values_for(np.int32, 128, shift=83)
    copied = np.zeros_like(source)
    observed = np.zeros(65, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(copied) as dst,
        device_array(observed) as out,
    ):
        launch(src, dst, out)
    np.testing.assert_array_equal(copied, source)
    np.testing.assert_array_equal(
        observed[:64], np.full(64, source.sum(dtype=np.int32))
    )
    assert observed[64] == source[:90:2].sum(dtype=np.int32)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_sort_scan_shared_storage(api):
    @cute.kernel
    def kernel(source: cute.Pointer, output: cute.Pointer, tiles: cutlass.Int32):
        group = api.this_block()
        storage = api.TempStorage(alignment=128)
        for tile in range(tiles):
            keys = api.ThreadData(2)
            api.load(
                group,
                source,
                keys,
                algorithm="transpose",
                offset=tile * 128,
                temp_storage=storage,
            )
            ordered = api.merge_sort_keys(group, keys, temp_storage=storage)
            prefixes = api.exclusive_sum(group, ordered, temp_storage=storage)
            api.store(
                group,
                output,
                prefixes,
                algorithm="transpose",
                offset=tile * 128,
                temp_storage=storage,
            )

    @cute.jit
    def launch(source: cute.Pointer, output: cute.Pointer, tiles: cutlass.Int32):
        kernel(source, output, tiles).launch(grid=1, block=(8, 4, 2))

    source = values_for(np.int32, 512, shift=19)
    observed = np.zeros_like(source)
    expected = np.empty_like(source)
    for start in range(0, source.size, 128):
        ordered = np.sort(source[start : start + 128])
        expected[start : start + 128] = np.cumsum(ordered) - ordered
    with device_array(source) as src, device_array(observed) as dst:
        launch(src, dst, cutlass.Int32(4))
    np.testing.assert_array_equal(observed, expected)
