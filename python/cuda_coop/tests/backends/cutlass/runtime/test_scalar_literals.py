# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Ordinary integer literals follow the shared floating-payload contract."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")
from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import cutlass_dtype, device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_integer_default_and_seed(api, dtype):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, output: cute.Pointer):
        group = api.this_block()
        values = api.ThreadData(1, dtype=value_type)
        api.load(group, source, values, valid_items=51, oob_default=0)
        scanned = api.exclusive_scan(group, values, initial_value=2)
        api.store(group, output, scanned)

    @cute.jit
    def launch(source: cute.Pointer, output: cute.Pointer):
        kernel(source, output).launch(grid=1, block=64)

    source = np.arange(64, dtype=dtype) / dtype(4)
    observed = np.zeros_like(source)
    values = source.copy()
    values[51:] = 0
    expected = np.concatenate((np.zeros(1, dtype=dtype), np.cumsum(values)[:-1])) + 2
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, expected)
