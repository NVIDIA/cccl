# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scheduler-style prefix example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    return require_runtime()


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop

    @cute.kernel
    def _scheduler_kernel(
        counts: cute.Tensor,
        target_linear_idx: cutlass.Int32,
        prefix_end: cute.Tensor,
        selected_group_idx: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = cute.arch.lane_idx()

        cur_count = counts[tidx]
        prefix_exclusive = coop.exclusive_sum(coop.this_block(), cur_count)
        prefix_end_val = prefix_exclusive + cur_count
        prefix_end[tidx] = prefix_end_val

        group_not_in_window = target_linear_idx >= prefix_end_val
        hitted_group_idx = cute.arch.popc(
            cute.arch.vote_ballot_sync(group_not_in_window)
        )
        if lane_id == 0:
            selected_group_idx[0] = hitted_group_idx.to(cutlass.Int32)

    @cute.jit
    def _run_scheduler(
        counts: cute.Tensor,
        target_linear_idx: cutlass.Int32,
        prefix_end: cute.Tensor,
        selected_group_idx: cute.Tensor,
    ):
        _scheduler_kernel(
            counts,
            target_linear_idx,
            prefix_end,
            selected_group_idx,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1))

    return _run_scheduler, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example(target_linear_idx: int = 57) -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_scheduler, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    host_counts = torch.tensor([(i % 7) + 1 for i in range(32)], dtype=torch.int32)
    counts = host_counts.cuda()
    prefix_end = torch.zeros((32,), dtype=torch.int32, device="cuda")
    selected_group_idx = torch.zeros((1,), dtype=torch.int32, device="cuda")
    counts_arg = from_dlpack(counts)
    prefix_end_arg = from_dlpack(prefix_end)
    selected_group_idx_arg = from_dlpack(selected_group_idx)
    target_arg = cutlass.Int32(target_linear_idx)

    def step() -> None:
        run_scheduler(
            counts_arg,
            target_arg,
            prefix_end_arg,
            selected_group_idx_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_prefix_end = torch.cumsum(host_counts, dim=0).to(torch.int32)
        expected_group_idx = int(
            torch.sum(expected_prefix_end <= target_linear_idx).item()
        )
        expected_group_idx = min(expected_group_idx, 32)

        torch.testing.assert_close(
            prefix_end.cpu(), expected_prefix_end, atol=0, rtol=0
        )
        actual_group_idx = int(selected_group_idx.cpu()[0].item())
        assert actual_group_idx == expected_group_idx

        return {
            "target_linear_idx": int(target_linear_idx),
            "selected_group_idx": actual_group_idx,
            "prefix_end": [int(x) for x in prefix_end.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example(target_linear_idx: int = 57) -> dict[str, Any]:
    """Run the scheduler-prefix example and validate the result."""

    prepared = prepare_example(target_linear_idx)
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
