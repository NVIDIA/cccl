# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as coop
from cuda.coop._core.thread_group import ThreadGroup
from cuda.coop.numba_mlir._compiler._operations import group_operation_name


def test_qualified_api_exposes_only_the_block_reduction_slice():
    assert coop.__all__ == ["ThreadGroup", "this_block", "reduce", "sum"]
    assert coop.ThreadGroup is ThreadGroup
    assert coop.this_block.__cuda_coop_backend_member__ == "this_block"
    assert coop.reduce.__cuda_coop_backend_member__ == "reduce"
    assert coop.sum.__cuda_coop_backend_member__ == "sum"
    assert coop.this_block.__module__ == "cuda.coop.numba_mlir._thread_group"
    assert coop.reduce.__module__ == "cuda.coop.numba_mlir._group_reduce"
    assert coop.sum.__module__ == "cuda.coop.numba_mlir._group_reduce"


def test_public_markers_are_recognized_only_by_exact_callable_identity():
    def reduce(*args, **kwargs):
        del args, kwargs

    assert group_operation_name(coop.reduce) == "reduce"
    assert group_operation_name(coop.sum) == "sum"
    assert group_operation_name(coop.this_block) == "this_block"
    assert group_operation_name(reduce) is None


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
