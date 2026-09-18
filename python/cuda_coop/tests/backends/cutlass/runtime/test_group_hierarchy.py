# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Hierarchy queries and synchronization against flat-coordinate oracles."""

import re
import shutil
import subprocess

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import NUMPY_DTYPES, cutlass_dtype, device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_BLOCK = (8, 4, 3)
_GRID = (2, 2, 1)
_BLOCK_THREADS = 96
_THREADS = 4 * _BLOCK_THREADS
_FIELDS = 23


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
def test_queries(api):
    @cute.kernel
    def kernel(observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        thread_index = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        block_index = bx + _GRID[0] * (by + _GRID[1] * bz)
        index = block_index * _BLOCK_THREADS + thread_index
        outputs = cute.make_tensor(observed, cute.make_layout(_FIELDS * _THREADS))
        thread = api.this_thread()
        warp = api.this_warp()
        block = api.this_block()
        cluster = api.this_cluster()
        grid = api.this_grid()
        lanes = warp.group_by(8)
        warps = block.group_by(2, exhaustive=False)
        thread.sync()
        warp.sync_aligned()
        lanes.sync()
        lanes.sync_aligned()
        block.sync()
        block.sync_aligned()
        member = warps.is_member()
        assert isinstance(member, cutlass.Uint8)
        block_rank = block.rank()
        grid_rank = grid.rank()
        grid_count = block.count("grid")
        assert isinstance(block_rank, cutlass.Uint32)
        assert isinstance(grid_rank, cutlass.Uint64)
        assert isinstance(grid_count, cutlass.Uint64)
        outputs[0 * _THREADS + index] = thread.rank("block")
        outputs[1 * _THREADS + index] = thread.count()
        outputs[2 * _THREADS + index] = warp.rank()
        outputs[3 * _THREADS + index] = warp.count("block")
        outputs[4 * _THREADS + index] = block_rank
        outputs[5 * _THREADS + index] = grid_count
        outputs[6 * _THREADS + index] = lanes.rank()
        outputs[7 * _THREADS + index] = lanes.count("warp")
        outputs[8 * _THREADS + index] = warps.count("thread")
        outputs[9 * _THREADS + index] = member
        if member:
            outputs[10 * _THREADS + index] = warps.rank("warp")
            outputs[11 * _THREADS + index] = warps.rank("block")
            outputs[12 * _THREADS + index] = warps.count("block")
        outputs[13 * _THREADS + index] = thread.is_member()
        outputs[14 * _THREADS + index] = warp.is_member()
        outputs[15 * _THREADS + index] = block.is_member()
        outputs[16 * _THREADS + index] = lanes.is_member()
        outputs[17 * _THREADS + index] = grid_rank
        outputs[18 * _THREADS + index] = grid.count()
        outputs[19 * _THREADS + index] = cluster.count("block")
        outputs[20 * _THREADS + index] = cluster.rank("thread")
        outputs[21 * _THREADS + index] = block.rank("gpu_thread")
        outputs[22 * _THREADS + index] = grid.count("cluster")

    @cute.jit
    def launch(observed: cute.Pointer):
        kernel(observed).launch(grid=_GRID, block=_BLOCK)

    observed = np.full(_FIELDS * _THREADS, -1, dtype=np.int64)
    with device_array(observed) as out:
        launch(out)
    local = np.tile(np.arange(_BLOCK_THREADS), 4)
    lane = local % 32
    member = local < 64

    def full(value):
        return np.full(_THREADS, value, dtype=np.int64)

    expected = np.stack(
        (
            local,
            full(1),
            lane,
            full(3),
            local,
            full(4),
            lane % 8,
            full(4),
            full(64),
            member,
            np.where(member, local // 32, -1),
            np.where(member, 0, -1),
            np.where(member, 1, -1),
            full(1),
            full(1),
            full(1),
            full(1),
            np.arange(_THREADS),
            full(_THREADS),
            full(1),
            local,
            local,
            full(4),
        )
    )
    np.testing.assert_array_equal(observed.reshape(_FIELDS, _THREADS), expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES[:8])
def test_query_dtype(api, dtype):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(64)), value_type
        )
        group = api.this_block()
        rank = group.rank_as(value_type)
        count = group.count_as(value_type)
        assert isinstance(rank, value_type)
        assert isinstance(count, value_type)
        outputs[thread] = rank
        outputs[32 + thread] = count

    @cute.jit
    def launch(observed: cute.Pointer):
        kernel(observed).launch(grid=1, block=32)

    observed = np.zeros(64, dtype=dtype)
    with device_array(observed) as out:
        launch(out)
    np.testing.assert_array_equal(
        observed, np.concatenate((np.arange(32), np.full(32, 32))).astype(dtype)
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
def test_partial_warp(api):
    @cute.kernel
    def kernel(observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        outputs = cute.make_tensor(observed, cute.make_layout(96))
        group = api.this_block()
        outputs[thread] = group.rank("warp")
        outputs[48 + thread] = group.count("warp")

    @cute.jit
    def launch(observed: cute.Pointer):
        kernel(observed).launch(grid=1, block=48)

    observed = np.zeros(96, dtype=np.int32)
    with device_array(observed) as out:
        launch(out)
    np.testing.assert_array_equal(
        observed, np.concatenate((np.arange(48) // 32, np.full(48, 2)))
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
def test_nonpower_mapping(api):
    @cute.kernel
    def kernel(observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        outputs = cute.make_tensor(observed, cute.make_layout(4 * 64))
        group = api.this_warp().group_by(3, exhaustive=False)
        member = group.is_member()
        outputs[thread] = member
        outputs[64 + thread] = group.count()
        if member:
            group.sync()
            outputs[128 + thread] = group.rank()
            outputs[192 + thread] = group.rank("warp")

    @cute.jit
    def launch(observed: cute.Pointer):
        kernel(observed).launch(grid=1, block=64)

    observed = np.full(4 * 64, -1, dtype=np.int32)
    with device_array(observed) as out:
        launch(out)
    lane = np.arange(64) % 32
    member = lane < 30
    expected = np.stack(
        (
            member,
            np.full(64, 3),
            np.where(member, lane % 3, -1),
            np.where(member, lane // 3, -1),
        )
    )
    np.testing.assert_array_equal(observed.reshape(4, 64), expected)


def test_mapped_query_cubin(tmp_path):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        outputs = cute.make_tensor(observed, cute.make_layout(96))
        if thread < 32:
            group = cutlass_coop.this_block().group_by(2, exhaustive=False)
            outputs[thread] = group.rank() + group.count()

    @cute.jit
    def launch(observed: cute.Pointer):
        kernel(observed).launch(grid=1, block=96)

    observed = np.full(96, -1, dtype=np.int32)
    with device_array(observed) as out:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, out)
        compiled(out)
    expected = np.full_like(observed, -1)
    expected[:32] = np.arange(32) + 64
    np.testing.assert_array_equal(observed, expected)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        resources = subprocess.check_output(
            [cuobjdump, "--dump-resource-usage", str(cubin)], text=True
        )
        cubin.with_suffix(".sass").write_text(sass)
        cubin.with_suffix(".resources").write_text(resources)
        assert re.search(r"\b(?:BAR|CALL)(?:\.[A-Z0-9_]+)*\b", sass) is None
        assert "cuda_coop_cutlass_" not in sass
        assert "WARPSYNC" not in sass
        assert "SHARED:0" in resources
