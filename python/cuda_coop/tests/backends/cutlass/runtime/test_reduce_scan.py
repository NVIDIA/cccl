# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

from cuda import coop as common_coop

from ....support.paths import REPO_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA-capable PyTorch runtime", allow_module_level=True)

coop = pytest.importorskip("cuda.coop.cutlass")

from_dlpack = runtime.from_dlpack
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32

pytestmark = [
    pytest.mark.backend_cutlass,
    pytest.mark.runtime,
    pytest.mark.gpu,
]

_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
_SEGMENTS = 15
_SENTINEL = -999


@pytest.fixture(scope="module", autouse=True)
def _isolated_provider_cache(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("cuda-coop-cutlass-reduce-scan-runtime")
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


def _store_items(output: cute.Tensor, segment: int, items) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    offset = segment * _TILE_ITEMS + tidx * _ITEMS_PER_THREAD
    output[offset] = items[0]
    output[offset + 1] = items[1]


def _store_scalar(output: cute.Tensor, segment: int, value) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    output[segment * _TILE_ITEMS + tidx] = value


@cute.kernel
def _reduce_scan_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    block = common_coop.this_block()
    items = common_coop.ThreadData(_ITEMS_PER_THREAD)
    items[0] = values[tidx * _ITEMS_PER_THREAD]
    items[1] = values[tidx * _ITEMS_PER_THREAD + 1]

    fixed_storage = coop.TempStorage(4096, alignment=16)
    block_aggregate = coop.ThreadData(1, dtype=Int32)
    exclusive = coop.exclusive_sum(
        coop.this_block(),
        items,
        temp_storage=fixed_storage,
        aggregate_output=block_aggregate,
    )
    inclusive = common_coop.inclusive_sum(block, items)
    full_sum = common_coop.sum(block, items)
    direct_items_sum = coop.sum(
        coop.this_block(),
        items,
        broadcast=False,
        algorithm="raking",
    )
    partial_block_sum = common_coop.sum(
        block,
        values[tidx],
        broadcast=False,
        valid_items=47,
        algorithm="raking",
    )

    thread_sum = coop.sum(coop.this_thread(), values[tidx])
    warp_sum = coop.sum(coop.this_warp(), values[tidx])
    mapped_sum = coop.sum(coop.this_block().group_by(1), values[tidx])
    logical = coop.this_warp().group_by(8)
    logical_sum = coop.sum(logical, values[tidx])
    logical_partial_sum = common_coop.sum(
        logical,
        values[tidx],
        broadcast=False,
        valid_items=5,
    )
    logical_aggregate = coop.ThreadData(1, dtype=Int32)
    logical_scan = coop.inclusive_sum(
        logical,
        values[tidx],
        valid_items=5,
        aggregate_output=logical_aggregate,
    )
    logical_exclusive = coop.exclusive_scan(
        logical,
        values[tidx],
        scan_op="sum",
        initial_value=Int32(0),
        valid_items=5,
    )

    # Read the input payload only after both scans to verify non-mutation.
    _store_items(output, 0, items)
    _store_items(output, 1, exclusive)
    _store_items(output, 2, inclusive)
    _store_scalar(output, 3, full_sum)
    _store_scalar(output, 4, mapped_sum)
    _store_scalar(output, 6, logical_scan)
    _store_scalar(output, 7, logical_aggregate[0])
    _store_scalar(output, 8, block_aggregate[0])
    _store_scalar(output, 11, logical_exclusive)
    _store_scalar(output, 12, thread_sum)
    _store_scalar(output, 13, warp_sum)
    _store_scalar(output, 14, logical_sum)
    if tidx == 0:
        output[9 * _TILE_ITEMS] = direct_items_sum
        output[10 * _TILE_ITEMS] = partial_block_sum
    if tidx % 8 == 0:
        output[5 * _TILE_ITEMS + tidx] = logical_partial_sum


@cute.jit
def _run_reduce_scan(values: cute.Tensor, output: cute.Tensor):
    _reduce_scan_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_reduce_scan_group_routes_match_independent_oracles() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = (torch.arange(_TILE_ITEMS, dtype=torch.int32) * 17 % 97) - 48
    values = values_host.cuda()
    output = torch.full(
        (_SEGMENTS * _TILE_ITEMS,),
        _SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )

    _run_reduce_scan(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()
    segments = output.cpu().reshape(_SEGMENTS, _TILE_ITEMS)

    inclusive = torch.cumsum(values_host.to(torch.int64), dim=0).to(torch.int32)
    exclusive = inclusive - values_host
    torch.testing.assert_close(segments[0], values_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], exclusive, atol=0, rtol=0)
    torch.testing.assert_close(segments[2], inclusive, atol=0, rtol=0)

    total = int(values_host.sum())
    torch.testing.assert_close(
        segments[3, :_BLOCK_THREADS],
        torch.full((_BLOCK_THREADS,), total, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    warp_values = values_host[:_BLOCK_THREADS].reshape(2, 32)
    mapped_expected = warp_values.sum(dim=1).to(torch.int32).repeat_interleave(32)
    torch.testing.assert_close(
        segments[4, :_BLOCK_THREADS],
        mapped_expected,
        atol=0,
        rtol=0,
    )

    for begin in range(0, _BLOCK_THREADS, 8):
        group = values_host[begin : begin + 8]
        valid = group[:5]
        assert segments[5, begin].item() == int(valid.sum())
        torch.testing.assert_close(
            segments[6, begin : begin + 5],
            torch.cumsum(valid.to(torch.int64), dim=0).to(torch.int32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            segments[11, begin : begin + 5],
            (torch.cumsum(valid.to(torch.int64), dim=0) - valid.to(torch.int64)).to(
                torch.int32
            ),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            segments[6, begin + 5 : begin + 8],
            group[5:8],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            segments[11, begin + 5 : begin + 8],
            group[5:8],
            atol=0,
            rtol=0,
        )
        assert segments[7, begin].item() == int(valid.sum())

    torch.testing.assert_close(
        segments[8, :_BLOCK_THREADS],
        torch.full((_BLOCK_THREADS,), total, dtype=torch.int32),
        atol=0,
        rtol=0,
    )
    assert segments[9, 0].item() == total
    assert segments[10, 0].item() == int(values_host[:47].sum())
    torch.testing.assert_close(
        segments[12, :_BLOCK_THREADS],
        values_host[:_BLOCK_THREADS],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        segments[13, :_BLOCK_THREADS],
        warp_values.sum(dim=1).to(torch.int32).repeat_interleave(32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        segments[14, :_BLOCK_THREADS],
        values_host[:_BLOCK_THREADS]
        .reshape(8, 8)
        .sum(dim=1)
        .to(torch.int32)
        .repeat_interleave(8),
        atol=0,
        rtol=0,
    )


@cute.kernel
def _register_tensor_reduce_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    fragment = cute.make_rmem_tensor((_ITEMS_PER_THREAD,), Int32)
    fragment[0] = values[tidx * _ITEMS_PER_THREAD]
    fragment[1] = values[tidx * _ITEMS_PER_THREAD + 1]

    tensor_sum = coop.sum(
        coop.this_block(),
        fragment,
        broadcast=False,
        algorithm="raking",
    )
    tensor_ssa_max = coop.reduce(
        coop.this_block(),
        fragment.load(),
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )
    if tidx == 0:
        output[0] = tensor_sum
        output[1] = tensor_ssa_max


@cute.jit
def _run_register_tensor_reduce(values: cute.Tensor, output: cute.Tensor):
    _register_tensor_reduce_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_reduce_accepts_real_rmem_tensor_and_tensor_ssa_payloads() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = (torch.arange(_TILE_ITEMS, dtype=torch.int32) * 13 % 101) - 50
    values = values_host.cuda()
    output = torch.full((2,), _SENTINEL, dtype=torch.int32, device="cuda")

    _run_register_tensor_reduce(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.cpu(),
        torch.tensor(
            [int(values_host.sum()), int(values_host.max())],
            dtype=torch.int32,
        ),
        atol=0,
        rtol=0,
    )


@cute.kernel
def _non_power_of_two_warp_reduce_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    group = coop.this_warp().group_by(12, exhaustive=False)
    reduced = coop.sum(group, values[tidx], broadcast=False)
    lane = tidx % 32
    if lane == 0 or lane == 12:
        output[(tidx // 32) * 2 + lane // 12] = reduced


@cute.jit
def _run_non_power_of_two_warp_reduce(
    values: cute.Tensor,
    output: cute.Tensor,
):
    _non_power_of_two_warp_reduce_kernel(values, output).launch(
        grid=(1, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
    )


def test_non_power_of_two_warp_reduce_uses_stable_membership_masks() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(_BLOCK_THREADS, dtype=torch.int32) - 19
    values = values_host.cuda()
    output = torch.full((4,), _SENTINEL, dtype=torch.int32, device="cuda")

    _run_non_power_of_two_warp_reduce(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    expected = torch.tensor(
        [
            int(values_host[0:12].sum()),
            int(values_host[12:24].sum()),
            int(values_host[32:44].sum()),
            int(values_host[44:56].sum()),
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@cute.kernel
def _cluster_reduce_kernel(values: cute.Tensor, output: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * _BLOCK_THREADS + tidx
    output[index] = coop.sum(coop.this_cluster(), values[index])


@cute.jit
def _run_cluster_reduce(values: cute.Tensor, output: cute.Tensor):
    _cluster_reduce_kernel(values, output).launch(
        grid=(2, 1, 1),
        block=(_BLOCK_THREADS, 1, 1),
        cluster=(2, 1, 1),
    )


def test_cluster_reduce_broadcasts_across_cluster_members() -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("thread-block clusters require compute capability 9.0 or newer")

    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(2 * _BLOCK_THREADS, dtype=torch.int32) - 41
    values = values_host.cuda()
    output = torch.empty_like(values)

    _run_cluster_reduce(from_dlpack(values), from_dlpack(output))
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.cpu(),
        torch.full_like(values_host, int(values_host.sum())),
        atol=0,
        rtol=0,
    )
