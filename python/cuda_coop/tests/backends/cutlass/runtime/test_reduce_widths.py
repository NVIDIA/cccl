# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Logical reduction widths, excluded tails, and CUB prefix edge cases."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize(
    "width,prefix",
    ((1, False), (3, False), (32, False), (1, True), (32, True)),
)
def test_logical_width(api, width, prefix):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + 8 * (y + 4 * z)
        inputs = cute.make_tensor(source, cute.make_layout(128))
        outputs = cute.make_tensor(observed, cute.make_layout(128))
        group = api.this_warp().group_by(width, exhaustive=False)
        if cutlass.const_expr(prefix):
            result = api.sum(group, inputs[thread], broadcast=False, valid_items=1)
            if thread % width == 0:
                outputs[thread] = result
        else:
            result = api.sum(group, inputs[thread])
            if group.is_member():
                outputs[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=(8, 4, 4))

    source = values_for(np.int32, 128, shift=89)
    observed = np.full_like(source, -101)
    expected = observed.copy()
    for warp in range(4):
        for group in range(32 // width):
            start = warp * 32 + group * width
            if prefix:
                expected[start] = source[start]
            else:
                expected[start : start + width] = source[start : start + width].sum(
                    dtype=np.int32
                )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, expected)
