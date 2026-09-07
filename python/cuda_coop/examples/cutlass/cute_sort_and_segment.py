# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix-sort plus segment-boundary example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 1


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


def _expected_radix_order(
    keys: Any,
    values: Any,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
) -> tuple[Any, Any]:
    import torch

    span = max(0, int(end_bit) - int(begin_bit))
    shifted = keys.to(torch.int64) >> int(begin_bit)
    if span == 0:
        key_sig = torch.zeros_like(shifted).tolist()
    elif span >= 64:
        key_sig = shifted.tolist()
    else:
        key_sig = (shifted & ((1 << span) - 1)).tolist()
    order = list(range(len(key_sig)))
    if descending:
        order = sorted(order, key=lambda i: (-int(key_sig[i]), i))
    else:
        order = sorted(order, key=lambda i: (int(key_sig[i]), i))
    index = values.new_tensor(order).long()
    return keys[index], values[index]


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    temp_storage = coop.TempStorage(size_in_bytes=4096, sharing="shared")

    @cute.kernel
    def _sort_and_segment_kernel(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
        sorted_lane_out: cute.Tensor,
        head_out: cute.Tensor,
        run_id_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx.to(Int32)
        block = coop.this_block()
        keys = coop.ThreadData(ITEMS_PER_THREAD, Int32)
        coop.load(
            block,
            keys_in,
            keys,
        )
        lanes = coop.ThreadData.from_values(lane, dtype=Int32)

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
    def _run_sort_and_segment(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
        sorted_lane_out: cute.Tensor,
        head_out: cute.Tensor,
        run_id_out: cute.Tensor,
    ):
        _sort_and_segment_kernel(
            keys_in,
            sorted_key_out,
            sorted_lane_out,
            head_out,
            run_id_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_sort_and_segment, torch, from_dlpack, cutlass, temp_storage


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
        [
            12,
            -7,
            9,
            4,
            -18,
            22,
            15,
            -2,
            11,
            10,
            -5,
            3,
            -14,
            6,
            8,
            -9,
            1,
            13,
            -20,
            17,
            19,
            -6,
            7,
            -1,
            21,
            -11,
            5,
            -16,
            14,
            -3,
            2,
            -4,
        ],
        dtype=torch.int32,
    )
    quantized = torch.remainder(torch.abs(values_host), 8).to(torch.int32)
    keys_in = quantized.cuda()
    sorted_key_out = torch.zeros((32,), dtype=torch.int32, device="cuda")
    sorted_lane_out = torch.zeros((32,), dtype=torch.int32, device="cuda")
    head_out = torch.zeros((32,), dtype=torch.int32, device="cuda")
    run_id_out = torch.zeros((32,), dtype=torch.int32, device="cuda")
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
        lanes = torch.arange(32, dtype=torch.int32)
        expected_key, expected_lane = _expected_radix_order(
            quantized,
            lanes,
            begin_bit=0,
            end_bit=32,
            descending=False,
        )
        expected_head = torch.zeros((32,), dtype=torch.int32)
        expected_head[0] = 1
        for i in range(1, 32):
            expected_head[i] = int(expected_key[i] != expected_key[i - 1])
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
    """Run the sort-and-segment example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
