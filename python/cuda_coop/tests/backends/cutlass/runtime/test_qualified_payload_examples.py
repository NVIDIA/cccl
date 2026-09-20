# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Executable register conversion example for the qualified API reference."""

import numpy as np
import pytest

pytest.importorskip("cutlass")

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_register_conversion_example():
    # example-begin
    import cutlass
    from cutlass import cute

    from cuda.coop import cutlass as cutlass_coop

    @cute.kernel
    def convert_registers(destination: cute.Pointer):
        thread = cutlass_coop.this_block().rank()
        original = cute.make_rmem_tensor(2, cutlass.Int32)
        original[0] = thread * 2
        original[1] = thread * 2 + 1

        items = cutlass_coop.ThreadData.from_register_tensor(original)
        items[0] = items[0] + 10
        vector = items.to_tensor_ssa()
        copied = items.to_register_tensor()
        copied[1] = copied[1] + 100

        output = cute.make_tensor(destination, cute.make_layout(64 * 6))
        for item in cutlass.range_constexpr(2):
            output[thread * 6 + item] = original[item]
            output[thread * 6 + 2 + item] = vector[item]
            output[thread * 6 + 4 + item] = copied[item]

    @cute.jit
    def launch(destination: cute.Pointer):
        convert_registers(destination).launch(grid=1, block=64)

    # example-end

    observed = np.zeros((64, 6), dtype=np.int32)
    original = np.arange(128, dtype=np.int32).reshape(64, 2)
    changed = original + np.array([10, 0], dtype=np.int32)
    copied = changed + np.array([0, 100], dtype=np.int32)
    expected = np.concatenate((original, changed, copied), axis=1)
    with device_array(observed) as destination:
        launch(destination)
    np.testing.assert_array_equal(observed, expected)
