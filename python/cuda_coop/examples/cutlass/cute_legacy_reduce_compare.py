# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block- and warp-group reduction comparison for cuda.coop.cutlass."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    block_temp_storage = coop.TempStorage(size_in_bytes=4096, sharing="shared")

    @cute.kernel
    def _legacy_reduce_kernel(
        values: cute.Tensor,
        block_sum_out: cute.Tensor,
        block_items_sum_out: cute.Tensor,
        warp_max_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        warp = coop.this_warp()

        value = values[tidx]
        block_sum_out[tidx] = coop.reduce(
            block,
            value,
        )

        item_base = tidx * ITEMS_PER_THREAD
        items = coop.ThreadData.from_values(
            values[item_base],
            values[item_base + 1],
            dtype=Int32,
        )
        block_items_sum = coop.reduce(
            block,
            items,
        )
        block_items_sum_out[tidx] = block_items_sum

        warp_max_out[tidx] = coop.reduce(
            warp,
            value,
            binary_op="max",
        )

    @cute.jit
    def _run_legacy_reduce(
        values: cute.Tensor,
        block_sum_out: cute.Tensor,
        block_items_sum_out: cute.Tensor,
        warp_max_out: cute.Tensor,
    ):
        _legacy_reduce_kernel(
            values,
            block_sum_out,
            block_items_sum_out,
            warp_max_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_legacy_reduce, torch, from_dlpack, cutlass, block_temp_storage


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]
    measure_cuda_event_us: Callable[..., float]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_legacy_reduce, torch, from_dlpack, cutlass, block_temp_storage = make_runner()
    cutlass.cuda.initialize_cuda_context()
    block_temp_storage.reset_uses()

    host_values = torch.arange(1, TOTAL_ITEMS + 1, dtype=torch.int32)
    values = host_values.cuda()
    block_sum_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    block_items_sum_out = torch.zeros(
        (BLOCK_THREADS,), dtype=torch.int32, device="cuda"
    )
    warp_max_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    values_arg = from_dlpack(values)
    block_sum_arg = from_dlpack(block_sum_out)
    block_items_sum_arg = from_dlpack(block_items_sum_out)
    warp_max_arg = from_dlpack(warp_max_out)

    def step() -> None:
        run_legacy_reduce(
            values_arg,
            block_sum_arg,
            block_items_sum_arg,
            warp_max_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_block_sum = torch.full(
            (BLOCK_THREADS,),
            int(torch.sum(host_values[:BLOCK_THREADS]).item()),
            dtype=torch.int32,
        )
        expected_block_items_sum = torch.full(
            (BLOCK_THREADS,),
            int(torch.sum(host_values).item()),
            dtype=torch.int32,
        )
        expected_warp_max = torch.cat(
            (
                torch.full((32,), int(host_values[31].item()), dtype=torch.int32),
                torch.full((32,), int(host_values[63].item()), dtype=torch.int32),
            )
        )

        torch.testing.assert_close(
            block_sum_out.cpu(), expected_block_sum, atol=0, rtol=0
        )
        torch.testing.assert_close(
            block_items_sum_out.cpu(),
            expected_block_items_sum,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            warp_max_out.cpu(), expected_warp_max, atol=0, rtol=0
        )

        return {
            "block_sum": [int(x) for x in block_sum_out.cpu().tolist()],
            "block_items_sum": [int(x) for x in block_items_sum_out.cpu().tolist()],
            "warp_max": [int(x) for x in warp_max_out.cpu().tolist()],
        }

    def measure_cuda_event_us(*, warmup_iters: int, measure_iters: int) -> float:
        for _ in range(max(0, int(warmup_iters))):
            step()
        synchronize()

        iterations = max(1, int(measure_iters))
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            step()
        stop_event.record()
        stop_event.synchronize()
        return float(start_event.elapsed_time(stop_event) * 1.0e3 / iterations)

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
        measure_cuda_event_us=measure_cuda_event_us,
    )


def run_example() -> dict[str, Any]:
    """Run the legacy reduction comparison and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
