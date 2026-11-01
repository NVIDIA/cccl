# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile native staging for blocks and logical warps without launch."""

import importlib.util

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize("width", (8, 64))
@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("striped", (False, True))
def test_native_shared_staging_compile(width, items_per_thread, striped):
    path = PACKAGE_ROOT / "examples/shared_staging.py"
    spec = importlib.util.spec_from_file_location("shared_staging", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    assert (
        cute.compile[(GPUArch("sm_80"),)](
            example.launch,
            pointer,
            pointer,
            cutlass.Int32(width * items_per_thread - 3),
            width,
            items_per_thread,
            striped,
        )
        is not None
    )
