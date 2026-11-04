# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the documented Tensor Core/TopK example and its numerical checks."""

import importlib.util

import pytest

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("batches,windows,keep", [(1, 1, 1), (2, 4, 129)])
def test_tensor_core_windows_with_topk(batches, windows, keep):
    path = PACKAGE_ROOT / "examples/tensor_memory_topk.py"
    spec = importlib.util.spec_from_file_location("tensor_memory_topk", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    if not example.is_supported_device():
        pytest.skip("tcgen05 TMEM example requires SM100 or SM103")
    example.run_example(batches=batches, windows=windows, keep=keep)
