# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as coop
from cuda.coop._core.thread_group import ThreadGroup


def test_qualified_api_exposes_only_the_block_reduction_slice():
    assert coop.__all__ == ["ThreadGroup", "this_block", "reduce", "sum"]
    assert coop.ThreadGroup is ThreadGroup
    assert coop.this_block.__cuda_coop_backend_member__ == "this_block"
    assert coop.reduce.__cuda_coop_backend_member__ == "reduce"
    assert coop.sum.__cuda_coop_backend_member__ == "sum"


def test_qualified_block_descriptor_is_compiler_free():
    group = coop.this_block()

    assert group.kind == "block"
    assert group.is_current
    assert group.static_size is None


def test_reduction_markers_fail_outside_kernel_compilation():
    group = coop.this_block()

    with pytest.raises(RuntimeError, match="kernel compile-time construct"):
        coop.sum(group, 1)
    with pytest.raises(RuntimeError, match="kernel compile-time construct"):
        coop.reduce(group, 1, binary_op="max")
