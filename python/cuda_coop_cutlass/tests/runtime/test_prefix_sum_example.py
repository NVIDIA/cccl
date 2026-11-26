# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the prefix-sum example and check its 512 running sums."""

import importlib.util

import numpy as np
import pytest

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_prefix_sum_example():
    """Check the documented case of 128 threads with 4 items each."""

    path = PACKAGE_ROOT / "examples/prefix_sum.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_prefix_sum_example", path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    result = example.run_example()
    expected = np.cumsum(np.arange(512, dtype=np.int32), dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
