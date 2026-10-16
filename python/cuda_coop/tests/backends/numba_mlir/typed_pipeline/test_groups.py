# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native group typing preserves launch and partition constraints."""

from types import SimpleNamespace

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from ._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = [
    requires_typed_pipeline,
    pytest.mark.backend_numba_mlir,
    pytest.mark.compile,
]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (12, 0) if as_type is tuple else CHIP,
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(12, 0)),
    )


def test_constructor_only_kernel_retains_exact_launch_bounds():
    def kernel(output):
        coop.this_block()
        output[0] = 1

    compiled = testing.compile_kernel(
        kernel, types.void(types.int32[::1]), block=BLOCK, chip=CHIP
    )
    assert "launch_bounds 128" in compiled.plan


def test_different_partitions_cannot_merge_into_one_group():
    def kernel(output, condition):
        if condition:
            group = coop.this_warp().group_by(8)
        else:
            group = coop.this_warp().group_by(16)
        output[0] = group.count()

    with pytest.raises(TypingError, match="Cannot unify coop.group"):
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1], types.boolean),
            block=BLOCK,
            chip=CHIP,
        )


def test_captured_group_and_keyword_queries_compile():
    group = coop.this_warp().group_by(8)

    def kernel(output):
        alias = group
        output[0] = alias.rank_as(level="thread", dtype=types.int32)
        output[1] = alias.count(level="warp")

    compiled = testing.compile_kernel(
        kernel, types.void(types.int32[::1]), block=BLOCK, chip=CHIP
    )
    assert "call @cuda_coop_numba_mlir_group_" in compiled.mlir


def test_partition_count_requests_literal_specialization():
    @cuda.jit(chip=CHIP)
    def kernel(output, count):
        group = coop.this_warp().group_by(count=count)
        output[0] = group.count()

    key = (
        ("grid", (1, 1, 1)),
        ("block", BLOCK),
        ("sharedmem", 0),
        ("cluster", None),
    )
    # The compiler requests the launch argument's value, rather than
    # silently accepting a runtime partition that cannot have one type.
    from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

    with pytest.raises(ForceLiteralArg) as error:
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1], types.int64),
            block=BLOCK,
            chip=CHIP,
        )
    assert error.value.requested_args == {1}
    result = kernel._compile_launch_config_signature(
        types.void(types.int32[::1], types.IntegerLiteral(8)), key
    )
    assert result.metadata["cubin"]


@pytest.mark.parametrize("mapped", [False, True])
def test_sync_requires_supported_launch_and_barrier_contract(mapped):
    def kernel(output):
        if mapped:
            group = coop.this_block().group_by(2)
        else:
            group = coop.this_grid()
        group.sync()
        output[0] = 1

    message = "owned barrier lifetime" if mapped else "cooperative launch"
    with pytest.raises(TypingError, match=message):
        testing.compile_kernel(
            kernel, types.void(types.int32[::1]), block=BLOCK, chip=CHIP
        )
