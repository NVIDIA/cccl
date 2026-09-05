# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Multi-item ThreadData sort-and-segment example for cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime
from examples.cutlass.cute_sort_and_segment import _expected_radix_order

BLOCK_THREADS = 32
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
    temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")

    @cute.kernel
    def _sort_and_segment_thread_data_kernel(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
        sorted_lane_out: cute.Tensor,
        head_out: cute.Tensor,
        run_id_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        base_lane = tidx.to(Int32) * Int32(items_per_thread)
        block = coop.this_block()

        keys = coop.ThreadData(items_per_thread, Int32)
        coop.load(
            block,
            keys_in,
            keys,
        )
        lanes = coop.ThreadData.from_fn(
            items_per_thread,
            lambda item: base_lane + Int32(item),
            dtype=Int32,
        )

        sorted_keys, sorted_lanes = coop.radix_sort_pairs(
            block,
            keys,
            lanes,
            begin_bit=0,
            end_bit=32,
            descending=False,
            temp_storage=temp_storage,
        )
        heads = coop.discontinuity(
            block,
            sorted_keys,
            mode="heads",
            temp_storage=temp_storage,
        )
        run_ids = coop.exclusive_sum(
            block,
            heads,
            temp_storage=temp_storage,
        )

        coop.store(
            block,
            sorted_key_out,
            sorted_keys,
        )
        coop.store(
            block,
            sorted_lane_out,
            sorted_lanes,
        )
        coop.store(
            block,
            head_out,
            heads,
        )
        coop.store(
            block,
            run_id_out,
            run_ids,
        )

    @cute.jit
    def _run_sort_and_segment_thread_data(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
        sorted_lane_out: cute.Tensor,
        head_out: cute.Tensor,
        run_id_out: cute.Tensor,
    ):
        _sort_and_segment_thread_data_kernel(
            keys_in,
            sorted_key_out,
            sorted_lane_out,
            head_out,
            run_id_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_sort_and_segment_thread_data, torch, from_dlpack, cutlass, temp_storage


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_sort_and_segment, torch, from_dlpack, cutlass, temp_storage = make_runner()
    cutlass.cuda.initialize_cuda_context()
    temp_storage.reset_uses()

    values_host = torch.tensor(
        [((idx * 19 + 7) % 47) - 23 for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    quantized = torch.remainder(torch.abs(values_host), 11).to(torch.int32)
    keys_in = quantized.cuda()
    sorted_key_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    sorted_lane_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    head_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    run_id_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    sorted_key_arg = from_dlpack(sorted_key_out)
    sorted_lane_arg = from_dlpack(sorted_lane_out)
    head_arg = from_dlpack(head_out)
    run_id_arg = from_dlpack(run_id_out)

    def step() -> None:
        run_sort_and_segment(
            keys_arg,
            sorted_key_arg,
            sorted_lane_arg,
            head_arg,
            run_id_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        lanes = torch.arange(TOTAL_ITEMS, dtype=torch.int32)
        expected_key, expected_lane = _expected_radix_order(
            quantized,
            lanes,
            begin_bit=0,
            end_bit=32,
            descending=False,
        )
        expected_head = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32)
        expected_head[0] = 1
        for idx in range(1, TOTAL_ITEMS):
            expected_head[idx] = int(expected_key[idx] != expected_key[idx - 1])
        expected_run_id = (
            torch.cumsum(expected_head.to(torch.int64), dim=0)
            - expected_head.to(torch.int64)
        ).to(torch.int32)

        torch.testing.assert_close(sorted_key_out.cpu(), expected_key, atol=0, rtol=0)
        torch.testing.assert_close(sorted_lane_out.cpu(), expected_lane, atol=0, rtol=0)
        torch.testing.assert_close(head_out.cpu(), expected_head, atol=0, rtol=0)
        torch.testing.assert_close(run_id_out.cpu(), expected_run_id, atol=0, rtol=0)

        return {
            "sorted_key": [int(x) for x in sorted_key_out.cpu().tolist()],
            "sorted_lane": [int(x) for x in sorted_lane_out.cpu().tolist()],
            "run_id": [int(x) for x in run_id_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the multi-item sort-and-segment example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
