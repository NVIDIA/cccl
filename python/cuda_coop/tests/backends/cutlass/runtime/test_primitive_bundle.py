# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Combine group queries, Load, Reduce, and Store in one compiled kernel.

Transpose Load and Store share explicit scratch. A full payload reduction
uses CUDAX, while a scalar prefix reduction selects CUB. The checks expose
missing provider registration or interference between operations that
share one compilation. The final Store also checks that both reductions
leave the original payload unchanged.
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

    Every member records the full reduction. Only the block root records the
    prefix result, using the first item from each of the first 45 threads.
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
        outputs = cute.make_tensor(observed, cute.make_layout(65))
        payload = api.ThreadData(items_per_thread)
        storage = api.TempStorage(sharing="shared", auto_sync=True)
        api.load(
            group, source, payload, algorithm="transpose", temp_storage=storage
        )
        outputs[thread] = api.sum(group, payload)
        prefix = api.sum(group, payload[0], broadcast=False, valid_items=45)
        if thread == 0:
            outputs[64] = prefix
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
    observed = np.zeros(65, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(copied) as dst,
        device_array(observed) as out,
    ):
        launch(src, dst, out, items_per_thread)
    np.testing.assert_array_equal(copied, source)
    np.testing.assert_array_equal(
        observed[:64], np.full(64, source.sum(dtype=np.int32))
    )
    assert observed[64] == source[
        : 45 * items_per_thread : items_per_thread
    ].sum(dtype=np.int32)
