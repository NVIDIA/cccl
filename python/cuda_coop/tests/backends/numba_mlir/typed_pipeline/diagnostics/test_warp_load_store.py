# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp transfer controls are bounded by each group's own memory tile."""

from __future__ import annotations

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline


@pytest.fixture(autouse=True)
def _fixed_compiler_target(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (12, 0) if as_type is tuple else CHIP,
    )


@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_warp_static_offset_reserves_space_for_each_tile(
    width, items_per_thread
):
    def kernel(source):
        value = coop.ThreadData(items_per_thread)
        coop.load(
            coop.this_warp().group_by(width),
            source,
            value,
            offset=(1 << 63) - 1,
        )

    with pytest.raises(
        TypedPipelineError,
        match="static offset plus the warp-group tile origin must fit",
    ):
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1]),
            block=(16, 2, 2),
            chip=CHIP,
        )


@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_warp_static_valid_count_uses_the_group_tile(width, items_per_thread):
    valid_items = width * items_per_thread + 1

    def kernel(source):
        value = coop.ThreadData(items_per_thread)
        coop.load(
            coop.this_warp().group_by(width),
            source,
            value,
            valid_items=valid_items,
        )

    with pytest.raises(
        TypedPipelineError,
        match=rf"group tile size \({width * items_per_thread}\)",
    ):
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1]),
            block=(64, 1, 1),
            chip=CHIP,
        )


def test_warp_explicit_temp_storage_is_rejected():
    def kernel(source):
        value = coop.ThreadData(items_per_thread=4)
        storage = coop.TempStorage()
        coop.load(
            coop.this_warp(),
            source,
            value,
            algorithm="transpose",
            temp_storage=storage,
        )

    with pytest.raises(
        TypingError, match="temp_storage is not supported for Warp"
    ):
        testing.compile_kernel(
            kernel,
            types.void(types.int32[::1]),
            block=(64, 1, 1),
            chip=CHIP,
        )
