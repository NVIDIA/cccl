# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

from ....support.paths import REPO_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

if not callable(getattr(cute, "_get_launch_facts", None)):
    pytest.skip(
        "requires CUTLASS DSL launch-facts support",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip(
        "requires a CUDA-capable PyTorch runtime",
        allow_module_level=True,
    )

coop = pytest.importorskip("cuda.coop.cutlass")

from_dlpack = runtime.from_dlpack

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_LOCAL_BLOCK_THREADS = 128
_LOCAL_QUERY_FIELDS = 15


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-group-runtime")
    env_values = {
        "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT": "ltoir",
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR": os.fspath(cache_dir),
    }
    if os.environ.get("CUDA_COOP_CUTLASS_FINAL_LINK_TEST") != "1":
        env_values["CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT"] = os.fspath(REPO_ROOT)
    original = {name: os.environ.get(name) for name in env_values}
    os.environ.update(env_values)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@cute.kernel
def _local_groups_kernel(query_out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    thread = coop.this_thread()
    warp = coop.this_warp()
    block = coop.this_block()
    lanes = warp.group_by(8)
    warps = block.group_by(2)

    thread.sync()
    warp.sync_aligned()
    lanes.sync()
    warps.sync_aligned()
    block.sync()

    query_out[0 * _LOCAL_BLOCK_THREADS + tidx] = thread.rank("block")
    query_out[1 * _LOCAL_BLOCK_THREADS + tidx] = thread.count("thread")
    query_out[2 * _LOCAL_BLOCK_THREADS + tidx] = warp.rank("thread")
    query_out[3 * _LOCAL_BLOCK_THREADS + tidx] = warp.count("block")
    query_out[4 * _LOCAL_BLOCK_THREADS + tidx] = block.rank("thread")
    query_out[5 * _LOCAL_BLOCK_THREADS + tidx] = block.count("grid")
    query_out[6 * _LOCAL_BLOCK_THREADS + tidx] = lanes.rank("thread")
    query_out[7 * _LOCAL_BLOCK_THREADS + tidx] = lanes.count("warp")
    query_out[8 * _LOCAL_BLOCK_THREADS + tidx] = warps.rank("warp")
    query_out[9 * _LOCAL_BLOCK_THREADS + tidx] = warps.count("thread")
    query_out[10 * _LOCAL_BLOCK_THREADS + tidx] = thread.is_member()
    query_out[11 * _LOCAL_BLOCK_THREADS + tidx] = warp.is_member()
    query_out[12 * _LOCAL_BLOCK_THREADS + tidx] = block.is_member()
    query_out[13 * _LOCAL_BLOCK_THREADS + tidx] = lanes.is_member()
    query_out[14 * _LOCAL_BLOCK_THREADS + tidx] = warps.is_member()


@cute.jit
def _run_local_groups(query_out: cute.Tensor):
    _local_groups_kernel(query_out).launch(
        grid=(1, 1, 1),
        block=(_LOCAL_BLOCK_THREADS, 1, 1),
    )


_REMAINDER_BLOCK_THREADS = 224
_REMAINDER_FIELDS = 2


@cute.kernel
def _remainder_groups_kernel(membership_out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    lanes = coop.this_warp().group_by(12, exhaustive=False)
    warps = coop.this_block().group_by(3, exhaustive=False)

    membership_out[0 * _REMAINDER_BLOCK_THREADS + tidx] = lanes.is_member()
    membership_out[1 * _REMAINDER_BLOCK_THREADS + tidx] = warps.is_member()


@cute.jit
def _run_remainder_groups(membership_out: cute.Tensor):
    _remainder_groups_kernel(membership_out).launch(
        grid=(1, 1, 1),
        block=(_REMAINDER_BLOCK_THREADS, 1, 1),
    )


_WIDE_GROUP_BLOCK_THREADS = 64
_WIDE_GROUP_BLOCKS = 2
_WIDE_GROUP_THREADS = _WIDE_GROUP_BLOCK_THREADS * _WIDE_GROUP_BLOCKS
_WIDE_QUERY_FIELDS = 3


@cute.kernel
def _cluster_group_kernel(query_out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * _WIDE_GROUP_BLOCK_THREADS + tidx
    cluster = coop.this_cluster()

    cluster.sync_aligned()
    query_out[0 * _WIDE_GROUP_THREADS + index] = cluster.rank("block")
    query_out[1 * _WIDE_GROUP_THREADS + index] = cluster.count("grid")
    query_out[2 * _WIDE_GROUP_THREADS + index] = cluster.is_member()


@cute.jit
def _run_cluster_group(query_out: cute.Tensor):
    _cluster_group_kernel(query_out).launch(
        grid=(_WIDE_GROUP_BLOCKS, 1, 1),
        block=(_WIDE_GROUP_BLOCK_THREADS, 1, 1),
        cluster=(_WIDE_GROUP_BLOCKS, 1, 1),
    )


@cute.kernel
def _grid_group_kernel(query_out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * _WIDE_GROUP_BLOCK_THREADS + tidx
    grid = coop.this_grid()

    grid.sync()
    grid.sync_aligned()
    query_out[0 * _WIDE_GROUP_THREADS + index] = grid.rank("block")
    query_out[1 * _WIDE_GROUP_THREADS + index] = grid.count("thread")
    query_out[2 * _WIDE_GROUP_THREADS + index] = grid.is_member()


@cute.jit
def _run_grid_group(query_out: cute.Tensor):
    _grid_group_kernel(query_out).launch(
        grid=(_WIDE_GROUP_BLOCKS, 1, 1),
        block=(_WIDE_GROUP_BLOCK_THREADS, 1, 1),
        cooperative=True,
    )


def _expected_wide_queries(*, cluster: bool) -> torch.Tensor:
    return torch.stack(
        (
            torch.arange(_WIDE_GROUP_BLOCKS, dtype=torch.int32).repeat_interleave(
                _WIDE_GROUP_BLOCK_THREADS
            ),
            torch.full(
                (_WIDE_GROUP_THREADS,),
                1 if cluster else _WIDE_GROUP_THREADS,
                dtype=torch.int32,
            ),
            torch.ones((_WIDE_GROUP_THREADS,), dtype=torch.int32),
        )
    )


def test_local_physical_and_exhaustive_mapped_groups_runtime():
    cutlass.cuda.initialize_cuda_context()

    query_out = torch.empty(
        (_LOCAL_QUERY_FIELDS * _LOCAL_BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_local_groups(from_dlpack(query_out))
    torch.cuda.synchronize()

    tidx = torch.arange(_LOCAL_BLOCK_THREADS, dtype=torch.int32)
    lane = tidx % 32
    expected_queries = torch.stack(
        (
            tidx,
            torch.ones_like(tidx),
            lane,
            torch.full_like(tidx, 4),
            tidx,
            torch.ones_like(tidx),
            lane % 8,
            torch.full_like(tidx, 4),
            (tidx % 64) // 32,
            torch.full_like(tidx, 64),
            *([torch.ones_like(tidx)] * 5),
        )
    )
    torch.testing.assert_close(
        query_out.cpu().reshape(_LOCAL_QUERY_FIELDS, _LOCAL_BLOCK_THREADS),
        expected_queries,
        atol=0,
        rtol=0,
    )


def test_non_exhaustive_mapped_group_membership_runtime():
    cutlass.cuda.initialize_cuda_context()

    membership_out = torch.empty(
        (_REMAINDER_FIELDS * _REMAINDER_BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_remainder_groups(from_dlpack(membership_out))
    torch.cuda.synchronize()

    expected_membership = torch.zeros(
        (_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        dtype=torch.int32,
    )
    for index in range(_REMAINDER_BLOCK_THREADS):
        lane = index % 32
        warp = index // 32
        if lane < 24:
            expected_membership[0, index] = 1
        if warp < 6:
            expected_membership[1, index] = 1

    torch.testing.assert_close(
        membership_out.cpu().reshape(_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        expected_membership,
        atol=0,
        rtol=0,
    )


def test_cluster_group_runtime():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("thread-block clusters require compute capability 9.0 or newer")

    cutlass.cuda.initialize_cuda_context()
    query_out = torch.empty(
        (_WIDE_QUERY_FIELDS * _WIDE_GROUP_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_cluster_group(from_dlpack(query_out))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        query_out.cpu().reshape(_WIDE_QUERY_FIELDS, _WIDE_GROUP_THREADS),
        _expected_wide_queries(cluster=True),
        atol=0,
        rtol=0,
    )


def test_cooperative_grid_group_runtime():
    cutlass.cuda.initialize_cuda_context()
    query_out = torch.empty(
        (_WIDE_QUERY_FIELDS * _WIDE_GROUP_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_grid_group(from_dlpack(query_out))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        query_out.cpu().reshape(_WIDE_QUERY_FIELDS, _WIDE_GROUP_THREADS),
        _expected_wide_queries(cluster=False),
        atol=0,
        rtol=0,
    )
