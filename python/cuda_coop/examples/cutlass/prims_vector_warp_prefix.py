# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS warp-prefix example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
WARP_THREADS = 32
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
VALID_WARP_LANES = 19


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    return cutlass, cute, torch, from_dlpack, coop


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the Prims warp JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop

    @cute.kernel
    def _vector_warp_prefix_kernel(
        values_in: cute.Tensor,
        prefix_out: cute.Tensor,
        valid_prefix_out: cute.Tensor,
        valid_prefix_aggregate_by_lane_out: cute.Tensor,
        warp_totals_by_lane_out: cute.Tensor,
        valid_warp_totals_by_lane_out: cute.Tensor,
        warp_min_by_lane_out: cute.Tensor,
        warp_max_by_lane_out: cute.Tensor,
        valid_warp_max_by_lane_out: cute.Tensor,
        warp_xor_by_lane_out: cute.Tensor,
        direct_copy_out: cute.Tensor,
        exchange_out: cute.Tensor,
        valid_lanes: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = cute.arch.lane_idx()
        warp = coop.this_warp()
        block = coop.this_block()
        values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        coop.load(
            warp,
            values_in,
            values_vec,
        )

        prefix_values = coop.exclusive_sum(warp, values_vec)
        valid_prefix_aggregate = coop.ThreadData(1)
        valid_prefix_values = coop.exclusive_sum(
            warp,
            values_vec,
            valid_items=valid_lanes,
            aggregate_output=valid_prefix_aggregate,
        )
        coop.store(
            warp,
            prefix_out,
            prefix_values,
            algorithm="direct",
        )
        coop.store(
            warp,
            valid_prefix_out,
            valid_prefix_values,
            algorithm="direct",
        )
        coop.store(
            warp,
            valid_prefix_aggregate_by_lane_out,
            valid_prefix_aggregate,
            algorithm="direct",
        )
        warp_totals = coop.sum(warp, values_vec)
        local_sum = values_vec[0] + values_vec[1]
        valid_warp_totals = coop.sum(
            warp,
            local_sum,
            valid_items=valid_lanes,
            broadcast=False,
        )
        warp_min = coop.reduce(
            warp,
            values_vec,
            binary_op="min",
        )
        warp_max = coop.reduce(
            warp,
            values_vec,
            binary_op="max",
        )
        local_max = cutlass.max(values_vec[0], values_vec[1])
        valid_warp_max = coop.reduce(
            warp,
            local_max,
            valid_items=valid_lanes,
            binary_op="max",
            broadcast=False,
        )
        warp_xor = coop.reduce(
            warp,
            values_vec,
            binary_op="bit_xor",
        )
        warp_totals_by_lane_out[tidx] = warp_totals
        warp_min_by_lane_out[tidx] = warp_min
        warp_max_by_lane_out[tidx] = warp_max
        warp_xor_by_lane_out[tidx] = warp_xor
        if lane_id == 0:
            valid_warp_totals_by_lane_out[tidx] = valid_warp_totals
            valid_warp_max_by_lane_out[tidx] = valid_warp_max
        coop.store(block, direct_copy_out, values_vec)

        striped_values = coop.exchange(
            warp,
            values_vec,
            mode="blocked_to_striped",
        )
        coop.store(
            warp,
            exchange_out,
            striped_values,
            algorithm="direct",
        )

    @cute.jit
    def _run_vector_warp_prefix(
        values_in: cute.Tensor,
        prefix_out: cute.Tensor,
        valid_prefix_out: cute.Tensor,
        valid_prefix_aggregate_by_lane_out: cute.Tensor,
        warp_totals_by_lane_out: cute.Tensor,
        valid_warp_totals_by_lane_out: cute.Tensor,
        warp_min_by_lane_out: cute.Tensor,
        warp_max_by_lane_out: cute.Tensor,
        valid_warp_max_by_lane_out: cute.Tensor,
        warp_xor_by_lane_out: cute.Tensor,
        direct_copy_out: cute.Tensor,
        exchange_out: cute.Tensor,
        valid_lanes: cutlass.Int32,
    ):
        _vector_warp_prefix_kernel(
            values_in,
            prefix_out,
            valid_prefix_out,
            valid_prefix_aggregate_by_lane_out,
            warp_totals_by_lane_out,
            valid_warp_totals_by_lane_out,
            warp_min_by_lane_out,
            warp_max_by_lane_out,
            valid_warp_max_by_lane_out,
            warp_xor_by_lane_out,
            direct_copy_out,
            exchange_out,
            valid_lanes,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_vector_warp_prefix, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_prefix(values: Any, *, torch: Any) -> Any:
    expected = torch.empty_like(values)
    tile_items = WARP_THREADS * ITEMS_PER_THREAD
    for tile_base in range(0, TOTAL_ITEMS, tile_items):
        warp_values = values[tile_base : tile_base + tile_items]
        expected[tile_base : tile_base + tile_items] = (
            torch.cumsum(warp_values, dim=0) - warp_values
        )
    return expected.to(values.dtype)


def _expected_valid_prefix(values: Any, valid_lanes: int, *, torch: Any) -> Any:
    expected = torch.zeros_like(values)
    tile_items = WARP_THREADS * ITEMS_PER_THREAD
    valid_items = valid_lanes * ITEMS_PER_THREAD
    for tile_base in range(0, TOTAL_ITEMS, tile_items):
        valid_values = values[tile_base : tile_base + valid_items]
        expected[tile_base : tile_base + valid_items] = (
            torch.cumsum(valid_values, dim=0) - valid_values
        )
    return expected.to(values.dtype)


def _expected_warp_totals(values: Any, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        warp_values = values[tile_base : tile_base + WARP_THREADS * ITEMS_PER_THREAD]
        expected[warp_id * WARP_THREADS : (warp_id + 1) * WARP_THREADS] = torch.sum(
            warp_values
        )
    return expected


def _expected_valid_warp_totals(values: Any, valid_lanes: int, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        valid_values = values[tile_base : tile_base + valid_lanes * ITEMS_PER_THREAD]
        expected[warp_id * WARP_THREADS] = torch.sum(valid_values)
    return expected


def _expected_warp_min(values: Any, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        warp_values = values[tile_base : tile_base + WARP_THREADS * ITEMS_PER_THREAD]
        expected[warp_id * WARP_THREADS : (warp_id + 1) * WARP_THREADS] = torch.min(
            warp_values
        )
    return expected


def _expected_warp_max(values: Any, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        warp_values = values[tile_base : tile_base + WARP_THREADS * ITEMS_PER_THREAD]
        expected[warp_id * WARP_THREADS : (warp_id + 1) * WARP_THREADS] = torch.max(
            warp_values
        )
    return expected


def _expected_valid_warp_max(values: Any, valid_lanes: int, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        valid_values = values[tile_base : tile_base + valid_lanes * ITEMS_PER_THREAD]
        expected[warp_id * WARP_THREADS] = torch.max(valid_values)
    return expected


def _expected_warp_xor(values: Any, *, torch: Any) -> Any:
    expected = torch.zeros((BLOCK_THREADS,), dtype=values.dtype)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        running_xor = 0
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        warp_values = values[tile_base : tile_base + WARP_THREADS * ITEMS_PER_THREAD]
        for value in warp_values.tolist():
            running_xor ^= int(value)
        expected[warp_id * WARP_THREADS : (warp_id + 1) * WARP_THREADS] = running_xor
    return expected


def _expected_blocked_to_striped(values: Any, *, torch: Any) -> Any:
    expected = torch.empty_like(values)
    for warp_id in range(BLOCK_THREADS // WARP_THREADS):
        tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
        for lane in range(WARP_THREADS):
            direct_base = tile_base + lane * ITEMS_PER_THREAD
            expected[direct_base] = values[tile_base + lane]
            expected[direct_base + 1] = values[tile_base + WARP_THREADS + lane]
    return expected


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_vector_warp_prefix, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    values_host = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    valid_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    valid_prefix_aggregate_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    warp_totals_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    valid_warp_totals_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    warp_min_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    warp_max_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    valid_warp_max_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    warp_xor_by_lane_out = torch.zeros(
        (BLOCK_THREADS,),
        dtype=torch.int32,
        device="cuda",
    )
    direct_copy_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    exchange_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    values_arg = from_dlpack(values_in)
    prefix_arg = from_dlpack(prefix_out)
    valid_prefix_arg = from_dlpack(valid_prefix_out)
    valid_prefix_aggregate_arg = from_dlpack(valid_prefix_aggregate_by_lane_out)
    totals_arg = from_dlpack(warp_totals_by_lane_out)
    valid_totals_arg = from_dlpack(valid_warp_totals_by_lane_out)
    min_arg = from_dlpack(warp_min_by_lane_out)
    max_arg = from_dlpack(warp_max_by_lane_out)
    valid_max_arg = from_dlpack(valid_warp_max_by_lane_out)
    xor_arg = from_dlpack(warp_xor_by_lane_out)
    direct_copy_arg = from_dlpack(direct_copy_out)
    exchange_arg = from_dlpack(exchange_out)

    def step() -> None:
        run_vector_warp_prefix(
            values_arg,
            prefix_arg,
            valid_prefix_arg,
            valid_prefix_aggregate_arg,
            totals_arg,
            valid_totals_arg,
            min_arg,
            max_arg,
            valid_max_arg,
            xor_arg,
            direct_copy_arg,
            exchange_arg,
            cutlass.Int32(VALID_WARP_LANES),
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        torch.testing.assert_close(
            prefix_out.cpu(),
            _expected_prefix(values_host, torch=torch),
            atol=0,
            rtol=0,
        )
        expected_valid_prefix = _expected_valid_prefix(
            values_host,
            VALID_WARP_LANES,
            torch=torch,
        )
        valid_prefix_cpu = valid_prefix_out.cpu()
        for warp_id in range(BLOCK_THREADS // WARP_THREADS):
            tile_base = warp_id * WARP_THREADS * ITEMS_PER_THREAD
            valid_end = tile_base + VALID_WARP_LANES * ITEMS_PER_THREAD
            torch.testing.assert_close(
                valid_prefix_cpu[tile_base:valid_end],
                expected_valid_prefix[tile_base:valid_end],
                atol=0,
                rtol=0,
            )
        expected_totals = _expected_warp_totals(values_host, torch=torch)
        lane0_indices = torch.arange(
            0,
            BLOCK_THREADS,
            WARP_THREADS,
            dtype=torch.int64,
        )
        torch.testing.assert_close(
            warp_totals_by_lane_out.cpu(),
            expected_totals,
            atol=0,
            rtol=0,
        )
        expected_valid_totals = _expected_valid_warp_totals(
            values_host,
            VALID_WARP_LANES,
            torch=torch,
        )
        valid_prefix_aggregate_cpu = valid_prefix_aggregate_by_lane_out.cpu()
        for warp_id in range(BLOCK_THREADS // WARP_THREADS):
            lane_base = warp_id * WARP_THREADS
            valid_lane_end = lane_base + VALID_WARP_LANES
            expected_aggregate = int(expected_valid_totals[lane_base].item())
            torch.testing.assert_close(
                valid_prefix_aggregate_cpu[lane_base:valid_lane_end],
                torch.full(
                    (VALID_WARP_LANES,),
                    expected_aggregate,
                    dtype=torch.int32,
                ),
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(
            valid_warp_totals_by_lane_out.cpu()[lane0_indices],
            expected_valid_totals[lane0_indices],
            atol=0,
            rtol=0,
        )
        expected_min = _expected_warp_min(values_host, torch=torch)
        torch.testing.assert_close(
            warp_min_by_lane_out.cpu(),
            expected_min,
            atol=0,
            rtol=0,
        )
        expected_max = _expected_warp_max(values_host, torch=torch)
        torch.testing.assert_close(
            warp_max_by_lane_out.cpu(),
            expected_max,
            atol=0,
            rtol=0,
        )
        expected_valid_max = _expected_valid_warp_max(
            values_host,
            VALID_WARP_LANES,
            torch=torch,
        )
        torch.testing.assert_close(
            valid_warp_max_by_lane_out.cpu()[lane0_indices],
            expected_valid_max[lane0_indices],
            atol=0,
            rtol=0,
        )
        expected_xor = _expected_warp_xor(values_host, torch=torch)
        torch.testing.assert_close(
            warp_xor_by_lane_out.cpu(),
            expected_xor,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            direct_copy_out.cpu(),
            values_host,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            exchange_out.cpu(),
            _expected_blocked_to_striped(values_host, torch=torch),
            atol=0,
            rtol=0,
        )

        return {
            "prefix_out": [int(x) for x in prefix_out.cpu().tolist()],
            "valid_prefix_first_warp": [
                int(x)
                for x in valid_prefix_out.cpu()[
                    : VALID_WARP_LANES * ITEMS_PER_THREAD
                ].tolist()
            ],
            "valid_prefix_aggregate_first_warp": [
                int(x)
                for x in valid_prefix_aggregate_by_lane_out.cpu()[
                    :VALID_WARP_LANES
                ].tolist()
            ],
            "warp_totals": [
                int(x) for x in warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "valid_warp_totals": [
                int(x)
                for x in valid_warp_totals_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "warp_min": [
                int(x) for x in warp_min_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "warp_max": [
                int(x) for x in warp_max_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "valid_warp_max": [
                int(x) for x in valid_warp_max_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "warp_xor": [
                int(x) for x in warp_xor_by_lane_out.cpu()[lane0_indices].tolist()
            ],
            "direct_copy": [int(x) for x in direct_copy_out.cpu().tolist()],
            "exchange_out": [int(x) for x in exchange_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the Prims vector warp-prefix example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
