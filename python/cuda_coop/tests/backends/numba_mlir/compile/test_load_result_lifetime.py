# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject returned payload aliases that overlap a later dynamic Load."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
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


def _compile(kernel):
    return kernel._compile_launch_config_signature(
        types.void(types.int32[::1], types.int32[::1], types.int32),
        (
            ("grid", (1, 1, 1)),
            ("block", (32, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_rejects_previous_iteration_payload(api):
    @cuda.jit(chip="sm_90")
    def kernel(source, output, tiles):
        block = api.this_block()
        previous = api.load(block, source, items_per_thread=1)
        for tile in range(1, tiles):
            current = api.load(
                block, source, items_per_thread=1, offset=tile * 32
            )
            output[(tile - 1) * 32 + cuda.threadIdx.x] = (
                current[0] - previous[0]
            )
            previous = current

    with pytest.raises(GroupRewriteError, match="earlier result live") as error:
        _compile(kernel)
    assert "two explicit ThreadData payloads" in str(error.value)
    assert "current = api.load(" in str(error.value)


@pytest.mark.parametrize("dynamic_index", [False, True])
def test_returning_load_rejects_tuple_alias_from_inlined_helper(dynamic_index):
    @cuda.jit(device=True, inline="always")
    def load_pair(block, source, offset):
        values = coop.load(block, source, items_per_thread=1, offset=offset)
        return ((values, values), 9)

    @cuda.jit(chip="sm_90")
    def kernel(source, output, tiles):
        block = coop.this_block()
        initial = coop.load(block, source, items_per_thread=1)
        saved = ((initial, initial), 9)
        for tile in range(1, tiles):
            current = load_pair(block, source, tile * 32)
            index = tile % 2 if dynamic_index else 0
            output[(tile - 1) * 32 + cuda.threadIdx.x] = (
                current[0][index][0] - saved[0][index][0]
            )
            saved = current

    with pytest.raises(GroupRewriteError, match="earlier result live"):
        _compile(kernel)


def test_returning_load_rejects_alias_retained_on_one_branch():
    @cuda.jit(chip="sm_90")
    def kernel(source, output, tiles):
        block = coop.this_block()
        saved = coop.load(block, source, items_per_thread=1)
        for tile in range(1, tiles):
            current = coop.load(
                block, source, items_per_thread=1, offset=tile * 32
            )
            output[(tile - 1) * 32 + cuda.threadIdx.x] = current[0] - saved[0]
            if tile % 2:
                saved = current

    with pytest.raises(GroupRewriteError, match="earlier result live"):
        _compile(kernel)


def test_returning_load_rejects_alias_across_outer_loop_backedge():
    @cuda.jit(chip="sm_90")
    def kernel(source, output, tiles):
        block = coop.this_block()
        saved = coop.load(block, source, items_per_thread=1)
        for tile in range(1, tiles):
            for inner in range(2):
                current = coop.load(
                    block, source, items_per_thread=1, offset=tile * 32
                )
                output[(tile - 1) * 32 + cuda.threadIdx.x] = (
                    current[0] - saved[0] + inner
                )
            saved = current

    with pytest.raises(GroupRewriteError, match="earlier result live"):
        _compile(kernel)


def _slice_alias(current, tiles):
    return current[:]


def _dynamic_slice_alias(current, tiles):
    return current[:tiles]


def _view_alias(current, tiles):
    return current.view(np.int32)


def _ravel_alias(current, tiles):
    return current.ravel()


def _tuple_slice_alias(current, tiles):
    return (current,)[0:1][0]


def _tuple_concat_alias(current, tiles):
    return ((current,) + (current,))[1]


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize(
    "make_alias",
    [
        _slice_alias,
        _dynamic_slice_alias,
        _view_alias,
        _ravel_alias,
        _tuple_slice_alias,
        _tuple_concat_alias,
    ],
    ids=[
        "slice",
        "dynamic_slice",
        "view",
        "ravel",
        "tuple_slice",
        "tuple_concat",
    ],
)
def test_returning_load_rejects_native_alias_in_while_helper(api, make_alias):
    alias = cuda.jit(device=True, inline="always")(make_alias)

    @cuda.jit(device=True, inline="always")
    def copy_differences(source, output, tiles):
        block = api.this_block()
        saved = source
        tile = 0
        while tile < tiles:
            current = api.load(
                block, source, items_per_thread=1, offset=tile * 32
            )
            if tile > 0:
                output[(tile - 1) * 32 + cuda.threadIdx.x] = (
                    current[0] - saved[0]
                )
            if tile == tiles - 1:
                break
            saved = alias(current, tiles)
            if tile % 2 == 0:
                tile += 2
                continue
            tile += 1

    @cuda.jit(chip="sm_90")
    def kernel(source, output, tiles):
        copy_differences(source, output, tiles)

    with pytest.raises(GroupRewriteError, match="earlier result live"):
        _compile(kernel)
