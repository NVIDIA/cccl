# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp prefix, reduction, and exchange example for cuda.coop.cutlass root."""

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
    from cutlass.base_dsl.typing import Int32

    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _warp_prefix_reduce_kernel(
        values: cute.Tensor,
        prefix_out: cute.Tensor,
        warp_totals: cute.Tensor,
        exchange_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_id = cute.arch.lane_idx()
        warp_id = tidx // cutlass.Int32(32)
        warp = coop.this_warp()

        striped_items = coop.ThreadData(2, Int32)
        coop.load(
            warp,
            values,
            striped_items,
            algorithm="striped",
        )
        value = striped_items[0]
        prefix_value = coop.ThreadData.from_values(
            coop.exclusive_sum(warp, value),
            dtype=Int32,
        )
        coop.store(warp, prefix_out, prefix_value)

        warp_total = coop.sum(warp, value)
        if lane_id == 0:
            warp_totals[warp_id] = warp_total

        blocked_items = coop.exchange(warp, striped_items, mode="striped_to_blocked")
        coop.store(
            warp,
            exchange_out,
            blocked_items,
            algorithm="direct",
        )

    @cute.jit
    def _run_warp_prefix_reduce(
        values: cute.Tensor,
        prefix_out: cute.Tensor,
        warp_totals: cute.Tensor,
        exchange_out: cute.Tensor,
    ):
        _warp_prefix_reduce_kernel(
            values,
            prefix_out,
            warp_totals,
            exchange_out,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1))

    return _run_warp_prefix_reduce, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_warp_prefix_reduce, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    host_values = torch.arange(1, 129, dtype=torch.int32)
    values = host_values.cuda()
    prefix_out = torch.zeros((64,), dtype=torch.int32, device="cuda")
    warp_totals = torch.zeros((2,), dtype=torch.int32, device="cuda")
    exchange_out = torch.zeros((128,), dtype=torch.int32, device="cuda")
    values_arg = from_dlpack(values)
    prefix_arg = from_dlpack(prefix_out)
    totals_arg = from_dlpack(warp_totals)
    exchange_arg = from_dlpack(exchange_out)

    def step() -> None:
        run_warp_prefix_reduce(
            values_arg,
            prefix_arg,
            totals_arg,
            exchange_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        first_warp_values = host_values[:32]
        second_warp_values = host_values[64:96]
        expected_prefix = torch.cat(
            (
                torch.cumsum(first_warp_values, dim=0) - first_warp_values,
                torch.cumsum(second_warp_values, dim=0) - second_warp_values,
            )
        ).to(torch.int32)
        expected_totals = torch.stack(
            [torch.sum(first_warp_values), torch.sum(second_warp_values)]
        ).to(torch.int32)

        torch.testing.assert_close(prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(warp_totals.cpu(), expected_totals, atol=0, rtol=0)
        torch.testing.assert_close(exchange_out.cpu(), host_values, atol=0, rtol=0)

        return {
            "prefix_out": [int(x) for x in prefix_out.cpu().tolist()],
            "warp_totals": [int(x) for x in warp_totals.cpu().tolist()],
            "exchange_out": [int(x) for x in exchange_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the warp-prefix/reduce example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


if __name__ == "__main__":
    print(run_example())
