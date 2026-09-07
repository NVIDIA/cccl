# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Current thread-group query example for cuda.coop.cutlass."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 64


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _thread_group_query_kernel(
        thread_rank_out: cute.Tensor,
        thread_count_out: cute.Tensor,
        warp_rank_out: cute.Tensor,
        warp_count_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        current_block = coop.this_block()
        current_block.sync()
        thread_rank_out[tidx] = current_block.rank("thread")
        thread_count_out[tidx] = current_block.count("thread")

        current_warp = coop.this_warp()
        warp_rank_out[tidx] = current_warp.rank("block")
        warp_count_out[tidx] = current_warp.count("block")

    @cute.jit
    def _run_thread_group_query(
        thread_rank_out: cute.Tensor,
        thread_count_out: cute.Tensor,
        warp_rank_out: cute.Tensor,
        warp_count_out: cute.Tensor,
    ):
        _thread_group_query_kernel(
            thread_rank_out,
            thread_count_out,
            warp_rank_out,
            warp_count_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_thread_group_query, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]
    measure_cuda_event_us: Callable[..., float]


def prepare_example() -> PreparedExample:
    """Prepare reusable outputs and a launch-only step for the example."""

    run_thread_group_query, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    thread_rank_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    thread_count_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    warp_rank_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    warp_count_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    thread_rank_arg = from_dlpack(thread_rank_out)
    thread_count_arg = from_dlpack(thread_count_out)
    warp_rank_arg = from_dlpack(warp_rank_out)
    warp_count_arg = from_dlpack(warp_count_out)

    def step() -> None:
        run_thread_group_query(
            thread_rank_arg,
            thread_count_arg,
            warp_rank_arg,
            warp_count_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_thread_rank = torch.arange(0, BLOCK_THREADS, dtype=torch.int32)
        expected_thread_count = torch.full(
            (BLOCK_THREADS,),
            BLOCK_THREADS,
            dtype=torch.int32,
        )
        expected_warp_rank = torch.cat(
            (
                torch.zeros((32,), dtype=torch.int32),
                torch.ones((32,), dtype=torch.int32),
            )
        )
        expected_warp_count = torch.full((BLOCK_THREADS,), 2, dtype=torch.int32)

        torch.testing.assert_close(
            thread_rank_out.cpu(), expected_thread_rank, atol=0, rtol=0
        )
        torch.testing.assert_close(
            thread_count_out.cpu(), expected_thread_count, atol=0, rtol=0
        )
        torch.testing.assert_close(
            warp_rank_out.cpu(), expected_warp_rank, atol=0, rtol=0
        )
        torch.testing.assert_close(
            warp_count_out.cpu(), expected_warp_count, atol=0, rtol=0
        )

        return {
            "thread_rank": [int(x) for x in thread_rank_out.cpu().tolist()],
            "thread_count": [int(x) for x in thread_count_out.cpu().tolist()],
            "warp_rank": [int(x) for x in warp_rank_out.cpu().tolist()],
            "warp_count": [int(x) for x in warp_count_out.cpu().tolist()],
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
    """Run the current thread-group query example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
