# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import pytest

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")
typing = pytest.importorskip("cutlass.base_dsl.typing")

if not callable(getattr(cute, "_get_launch_facts", None)):
    pytest.skip(
        "requires CUTLASS DSL launch-facts support",
        allow_module_level=True,
    )

import cuda.coop.cutlass as coop

from_dlpack = runtime.from_dlpack
Int32 = typing.Int32

pytestmark = pytest.mark.usefixtures("cutlass_cuda_available")

_LOCAL_BLOCK_THREADS = 128
_LOCAL_QUERY_FIELDS = 15
_LOCAL_REDUCE_FIELDS = 13


@cute.kernel
def _local_groups_kernel(
    values: cute.Tensor,
    query_out: cute.Tensor,
    reduce_out: cute.Tensor,
):
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

    value = values[tidx]
    items = coop.ThreadData.from_values(
        value,
        values[_LOCAL_BLOCK_THREADS + tidx],
        dtype=Int32,
    )
    reduce_out[0 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(thread, value)
    reduce_out[1 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(warp, value)
    reduce_out[2 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(block, value)
    reduce_out[3 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(lanes, value)
    reduce_out[4 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(warps, value)
    reduce_out[5 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(
        thread, value, broadcast=False
    )
    reduce_out[6 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(
        warp, value, broadcast=False
    )
    reduce_out[7 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(
        block, value, broadcast=False
    )
    reduce_out[8 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(
        lanes, value, broadcast=False
    )
    reduce_out[9 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(
        warps, value, broadcast=False
    )
    reduce_out[10 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(thread, items)
    reduce_out[11 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(lanes, items)
    reduce_out[12 * _LOCAL_BLOCK_THREADS + tidx] = coop.reduce(warps, items)


@cute.jit
def _run_local_groups(
    values: cute.Tensor,
    query_out: cute.Tensor,
    reduce_out: cute.Tensor,
):
    _local_groups_kernel(values, query_out, reduce_out).launch(
        grid=(1, 1, 1),
        block=(_LOCAL_BLOCK_THREADS, 1, 1),
    )


_REMAINDER_BLOCK_THREADS = 224
_REMAINDER_FIELDS = 2


@cute.kernel
def _remainder_groups_kernel(
    values: cute.Tensor,
    membership_out: cute.Tensor,
    broadcast_out: cute.Tensor,
    root_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    lanes = coop.this_warp().group_by(12, exhaustive=False)
    warps = coop.this_block().group_by(3, exhaustive=False)
    value = values[tidx]

    membership_out[0 * _REMAINDER_BLOCK_THREADS + tidx] = lanes.is_member()
    membership_out[1 * _REMAINDER_BLOCK_THREADS + tidx] = warps.is_member()
    broadcast_out[0 * _REMAINDER_BLOCK_THREADS + tidx] = coop.reduce(lanes, value)
    broadcast_out[1 * _REMAINDER_BLOCK_THREADS + tidx] = coop.reduce(warps, value)
    root_out[0 * _REMAINDER_BLOCK_THREADS + tidx] = coop.reduce(
        lanes, value, broadcast=False
    )
    root_out[1 * _REMAINDER_BLOCK_THREADS + tidx] = coop.reduce(
        warps, value, broadcast=False
    )


@cute.jit
def _run_remainder_groups(
    values: cute.Tensor,
    membership_out: cute.Tensor,
    broadcast_out: cute.Tensor,
    root_out: cute.Tensor,
):
    _remainder_groups_kernel(
        values,
        membership_out,
        broadcast_out,
        root_out,
    ).launch(
        grid=(1, 1, 1),
        block=(_REMAINDER_BLOCK_THREADS, 1, 1),
    )


_WIDE_GROUP_BLOCK_THREADS = 64
_WIDE_GROUP_BLOCKS = 2
_WIDE_GROUP_THREADS = _WIDE_GROUP_BLOCK_THREADS * _WIDE_GROUP_BLOCKS
_WIDE_QUERY_FIELDS = 3
_WIDE_REDUCE_FIELDS = 4


@cute.kernel
def _cluster_group_kernel(
    values: cute.Tensor,
    query_out: cute.Tensor,
    reduce_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * _WIDE_GROUP_BLOCK_THREADS + tidx
    cluster = coop.this_cluster()

    cluster.sync_aligned()
    query_out[0 * _WIDE_GROUP_THREADS + index] = cluster.rank("block")
    query_out[1 * _WIDE_GROUP_THREADS + index] = cluster.count("grid")
    query_out[2 * _WIDE_GROUP_THREADS + index] = cluster.is_member()

    value = values[index]
    items = coop.ThreadData.from_values(
        value,
        values[_WIDE_GROUP_THREADS + index],
        dtype=Int32,
    )
    reduce_out[0 * _WIDE_GROUP_THREADS + index] = coop.reduce(cluster, value)
    reduce_out[1 * _WIDE_GROUP_THREADS + index] = coop.reduce(
        cluster, value, broadcast=False
    )
    reduce_out[2 * _WIDE_GROUP_THREADS + index] = coop.reduce(cluster, items)
    reduce_out[3 * _WIDE_GROUP_THREADS + index] = coop.reduce(
        cluster, items, broadcast=False
    )


@cute.jit
def _run_cluster_group(
    values: cute.Tensor,
    query_out: cute.Tensor,
    reduce_out: cute.Tensor,
):
    _cluster_group_kernel(values, query_out, reduce_out).launch(
        grid=(_WIDE_GROUP_BLOCKS, 1, 1),
        block=(_WIDE_GROUP_BLOCK_THREADS, 1, 1),
        cluster=(_WIDE_GROUP_BLOCKS, 1, 1),
    )


@cute.kernel
def _grid_group_kernel(
    query_out: cute.Tensor,
):
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
def _run_grid_group(
    query_out: cute.Tensor,
):
    _grid_group_kernel(query_out).launch(
        grid=(_WIDE_GROUP_BLOCKS, 1, 1),
        block=(_WIDE_GROUP_BLOCK_THREADS, 1, 1),
        cooperative=True,
    )


def _broadcast_segments(values: torch.Tensor, segment_size: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                (segment_size,),
                int(values[start : start + segment_size].sum().item()),
                dtype=torch.int32,
            )
            for start in range(0, len(values), segment_size)
        ]
    )


def _root_segments(values: torch.Tensor, segment_size: int) -> torch.Tensor:
    result = torch.zeros_like(values)
    for start in range(0, len(values), segment_size):
        result[start] = values[start : start + segment_size].sum()
    return result


def test_local_physical_and_exhaustive_mapped_groups_runtime():
    cutlass.cuda.initialize_cuda_context()

    first = torch.arange(1, _LOCAL_BLOCK_THREADS + 1, dtype=torch.int32)
    second = torch.arange(1001, 1001 + _LOCAL_BLOCK_THREADS, dtype=torch.int32)
    values = torch.cat((first, second)).cuda()
    query_out = torch.empty(
        (_LOCAL_QUERY_FIELDS * _LOCAL_BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    reduce_out = torch.empty(
        (_LOCAL_REDUCE_FIELDS * _LOCAL_BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_local_groups(
        from_dlpack(values),
        from_dlpack(query_out),
        from_dlpack(reduce_out),
    )
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

    pair_values = first + second
    expected_reductions = torch.stack(
        (
            first,
            _broadcast_segments(first, 32),
            torch.full_like(first, int(first.sum().item())),
            _broadcast_segments(first, 8),
            _broadcast_segments(first, 64),
            first,
            _root_segments(first, 32),
            _root_segments(first, _LOCAL_BLOCK_THREADS),
            _root_segments(first, 8),
            _root_segments(first, 64),
            pair_values,
            _broadcast_segments(pair_values, 8),
            _broadcast_segments(pair_values, 64),
        )
    )
    torch.testing.assert_close(
        reduce_out.cpu().reshape(_LOCAL_REDUCE_FIELDS, _LOCAL_BLOCK_THREADS),
        expected_reductions,
        atol=0,
        rtol=0,
    )


def test_non_exhaustive_mapped_group_membership_and_reduce_runtime():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, _REMAINDER_BLOCK_THREADS + 1, dtype=torch.int32)
    values = values_host.cuda()
    membership_out = torch.empty(
        (_REMAINDER_FIELDS * _REMAINDER_BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    broadcast_out = torch.empty_like(membership_out)
    root_out = torch.empty_like(membership_out)

    _run_remainder_groups(
        from_dlpack(values),
        from_dlpack(membership_out),
        from_dlpack(broadcast_out),
        from_dlpack(root_out),
    )
    torch.cuda.synchronize()

    expected_membership = torch.zeros(
        (_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        dtype=torch.int32,
    )
    expected_broadcast = torch.zeros_like(expected_membership)
    expected_root = torch.zeros_like(expected_membership)
    for index in range(_REMAINDER_BLOCK_THREADS):
        lane = index % 32
        warp = index // 32
        if lane < 24:
            expected_membership[0, index] = 1
            start = index - lane + (lane // 12) * 12
            total = values_host[start : start + 12].sum()
            expected_broadcast[0, index] = total
            if lane % 12 == 0:
                expected_root[0, index] = total
        if warp < 6:
            expected_membership[1, index] = 1
            start = (warp // 3) * 3 * 32
            total = values_host[start : start + 3 * 32].sum()
            expected_broadcast[1, index] = total
            if index == start:
                expected_root[1, index] = total

    torch.testing.assert_close(
        membership_out.cpu().reshape(_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        expected_membership,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        broadcast_out.cpu().reshape(_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        expected_broadcast,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        root_out.cpu().reshape(_REMAINDER_FIELDS, _REMAINDER_BLOCK_THREADS),
        expected_root,
        atol=0,
        rtol=0,
    )


def _assert_wide_group_results(
    values_host: torch.Tensor,
    query_out: torch.Tensor,
    reduce_out: torch.Tensor,
    *,
    cluster: bool,
) -> None:
    first = values_host[:_WIDE_GROUP_THREADS]
    second = values_host[_WIDE_GROUP_THREADS:]
    expected_queries = torch.stack(
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
    torch.testing.assert_close(
        query_out.cpu().reshape(_WIDE_QUERY_FIELDS, _WIDE_GROUP_THREADS),
        expected_queries,
        atol=0,
        rtol=0,
    )

    pair_values = first + second
    expected_reductions = torch.stack(
        (
            torch.full_like(first, int(first.sum().item())),
            _root_segments(first, _WIDE_GROUP_THREADS),
            torch.full_like(pair_values, int(pair_values.sum().item())),
            _root_segments(pair_values, _WIDE_GROUP_THREADS),
        )
    )
    torch.testing.assert_close(
        reduce_out.cpu().reshape(_WIDE_REDUCE_FIELDS, _WIDE_GROUP_THREADS),
        expected_reductions,
        atol=0,
        rtol=0,
    )


@pytest.mark.usefixtures("cutlass_cluster_runtime_available")
def test_cluster_group_runtime():
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, 2 * _WIDE_GROUP_THREADS + 1, dtype=torch.int32)
    values = values_host.cuda()
    query_out = torch.empty(
        (_WIDE_QUERY_FIELDS * _WIDE_GROUP_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    reduce_out = torch.empty(
        (_WIDE_REDUCE_FIELDS * _WIDE_GROUP_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_cluster_group(
        from_dlpack(values),
        from_dlpack(query_out),
        from_dlpack(reduce_out),
    )
    torch.cuda.synchronize()
    _assert_wide_group_results(
        values_host,
        query_out,
        reduce_out,
        cluster=True,
    )


def test_cooperative_grid_group_runtime():
    cutlass.cuda.initialize_cuda_context()

    query_out = torch.empty(
        (_WIDE_QUERY_FIELDS * _WIDE_GROUP_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )

    _run_grid_group(
        from_dlpack(query_out),
    )
    torch.cuda.synchronize()
    expected_queries = torch.stack(
        (
            torch.arange(_WIDE_GROUP_BLOCKS, dtype=torch.int32).repeat_interleave(
                _WIDE_GROUP_BLOCK_THREADS
            ),
            torch.full(
                (_WIDE_GROUP_THREADS,),
                _WIDE_GROUP_THREADS,
                dtype=torch.int32,
            ),
            torch.ones((_WIDE_GROUP_THREADS,), dtype=torch.int32),
        )
    )
    torch.testing.assert_close(
        query_out.cpu().reshape(_WIDE_QUERY_FIELDS, _WIDE_GROUP_THREADS),
        expected_queries,
        atol=0,
        rtol=0,
    )
