# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject ambiguous Load modes and incompatible returned payloads."""

import os
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)
from cuda.coop.numba_mlir._compiler._rewrite import (
    CoopSinglePhaseRewriteError,
)

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, *arg_types):
    return kernel._compile_launch_config_signature(
        types.void(*arg_types),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_returning_load_requires_positive_static_integer(api, count):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=count)
        api.store(block, destination, values)

    with pytest.raises(ValueError, match="positive compile-time integer"):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("with_output", [False, True])
def test_load_requires_exactly_one_sizing_route(api, with_output):
    @cuda.jit
    def kernel(source):
        block = api.this_block()
        if with_output:
            values = api.ThreadData(items_per_thread=2)
            api.load(block, source, values, items_per_thread=2)
        else:
            api.load(block, source)

    message = (
        "output with items_per_thread must be"
        if with_output
        else "requires output"
    )
    with pytest.raises(TypeError, match=message):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_load_positional_integer_is_not_an_item_count(api):
    @cuda.jit
    def kernel(source):
        api.load(api.this_block(), source, 4)

    with pytest.raises(TypeError, match="output.*fixed-size"):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_rejects_data_dependent_extent(api):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=source[0])
        api.store(block, destination, values)

    with pytest.raises(GroupRewriteError, match="compile-time"):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("alignment", [1, 64])
def test_returning_load_materializes_minimum_payload_alignment(api, alignment):
    @cuda.jit
    def kernel(source, destination, index):
        block = api.this_block()
        values = api.load(
            block, source, items_per_thread=4, alignment=alignment
        )
        # Runtime indexing keeps a local allocation in the final PTX.
        destination[cuda.threadIdx.x] = values[index]

    result = _compile(kernel, types.int32[::1], types.int32[::1], types.int32)
    local_alignments = re.findall(r"\.local\s+\.align\s+(\d+)", result.ptx)
    assert local_alignments
    assert all(int(value) >= alignment for value in local_alignments)


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize(
    ("alignment", "message"),
    [
        (True, "alignment must be an integer or None"),
        (1.5, "alignment must be an integer or None"),
        (0, "alignment must be a positive integer"),
        (-1, "alignment must be a positive integer"),
        (3, "alignment must be a power of 2"),
    ],
)
def test_returning_load_rejects_invalid_alignment(api, alignment, message):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.load(
            block, source, items_per_thread=4, alignment=alignment
        )
        api.store(block, destination, values)

    with pytest.raises(CoopSinglePhaseRewriteError, match=message):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_rejects_data_dependent_alignment(api):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.load(
            block, source, items_per_thread=4, alignment=source[0]
        )
        api.store(block, destination, values)

    with pytest.raises(
        CoopSinglePhaseRewriteError, match="alignment must be a compile-time"
    ):
        _compile(kernel, types.int32[::1], types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("with_output", [False, True])
def test_load_alignment_requires_returning_form(api, with_output):
    @cuda.jit
    def kernel(source):
        block = api.this_block()
        if with_output:
            values = api.ThreadData(items_per_thread=4, alignment=64)
            api.load(block, source, values, alignment=64)
        else:
            api.load(block, source, alignment=64)

    message = (
        "alignment requires items_per_thread and no output"
        if with_output
        else "requires output or items_per_thread"
    )
    with pytest.raises(TypeError, match=message):
        _compile(kernel, types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_rejects_mismatched_branch_extents(api):
    @cuda.jit
    def kernel(source, destination, choose_second):
        block = api.this_block()
        if choose_second:
            values = api.load(block, source, items_per_thread=2)
        else:
            values = api.load(block, source, items_per_thread=4)
        api.store(block, destination, values)

    with pytest.raises(GroupRewriteError, match="inconsistent.*extent"):
        _compile(kernel, types.int32[::1], types.int32[::1], types.boolean)


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_retains_source_dtype_at_store(api):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=2)
        api.store(block, destination, values)

    with pytest.raises(TypeError, match="dtype"):
        _compile(kernel, types.int16[::1], types.int32[::1])


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_rejects_mismatched_branch_dtypes(api):
    @cuda.jit
    def kernel(integers, floats, destination, choose_second):
        block = api.this_block()
        if choose_second:
            values = api.load(block, integers, items_per_thread=2)
        else:
            values = api.load(block, floats, items_per_thread=2)
        api.store(block, destination, values)

    with pytest.raises(GroupRewriteError, match="inconsistent.*dtype"):
        _compile(
            kernel,
            types.int32[::1],
            types.float32[::1],
            types.int32[::1],
            types.boolean,
        )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_output_load_result_is_still_none(api):
    @cuda.jit
    def kernel(source, destination):
        block = api.this_block()
        values = api.ThreadData(items_per_thread=2)
        result = api.load(block, source, values)
        api.store(block, destination, result)

    with pytest.raises(TypingError, match="store value dtype none"):
        _compile(kernel, types.int32[::1], types.int32[::1])
