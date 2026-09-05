# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS block prefix/segment example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
SHUFFLE_DISTANCE = 1


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    return cutlass, cute, torch, from_dlpack, coop


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Build and return the Prims block-prefix runner plus helpers."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    scan_temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")
    reduce_temp_storage = coop.TempStorage(size_in_bytes=4096, sharing="shared")
    segment_temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")
    shuffle_temp_storage = coop.TempStorage(size_in_bytes=4096, sharing="shared")

    @cute.kernel
    def _vector_block_prefix_segment_kernel(
        values_in: cute.Tensor,
        segments_in: cute.Tensor,
        exclusive_out: cute.Tensor,
        inclusive_out: cute.Tensor,
        xor_prefix_out: cute.Tensor,
        sum_aggregate_out: cute.Tensor,
        xor_aggregate_out: cute.Tensor,
        sum_out: cute.Tensor,
        xor_out: cute.Tensor,
        diff_out: cute.Tensor,
        diff_right_out: cute.Tensor,
        head_out: cute.Tensor,
        tail_out: cute.Tensor,
        head_pair_out: cute.Tensor,
        tail_pair_out: cute.Tensor,
        shuffle_down_out: cute.Tensor,
        shuffle_prefix_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        segments_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        coop.load(
            block,
            values_in,
            values_vec,
        )
        coop.load(
            block,
            segments_in,
            segments_vec,
        )
        sum_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)
        xor_aggregate = coop.ThreadData(1, dtype=cutlass.Int32)

        exclusive = coop.exclusive_sum(
            block,
            values_vec,
            temp_storage=scan_temp_storage,
        )
        inclusive = coop.inclusive_sum(
            block,
            values_vec,
            aggregate_output=sum_aggregate,
            temp_storage=scan_temp_storage,
        )
        xor_prefix = coop.inclusive_scan(
            block,
            values_vec,
            scan_op="bit_xor",
            aggregate_output=xor_aggregate,
            temp_storage=scan_temp_storage,
        )
        total = coop.sum(block, values_vec)
        xor_total = coop.reduce(
            block,
            values_vec,
            binary_op="bit_xor",
        )
        diff = coop.adjacent_difference(
            block,
            segments_vec,
            temp_storage=segment_temp_storage,
        )
        diff_right = coop.adjacent_difference(
            block,
            segments_vec,
            direction="right",
            temp_storage=segment_temp_storage,
        )
        head = coop.discontinuity(
            block,
            segments_vec,
            mode="heads",
            temp_storage=segment_temp_storage,
        )
        tail = coop.discontinuity(
            block,
            segments_vec,
            mode="tails",
            temp_storage=segment_temp_storage,
        )
        head_pair, tail_pair = coop.discontinuity(
            block,
            segments_vec,
            mode="heads_and_tails",
            temp_storage=segment_temp_storage,
        )
        shuffle_prefix = coop.ThreadData(1, dtype=cutlass.Int32)
        shuffled = coop.shuffle(
            block,
            values_vec,
            mode="down",
            distance=SHUFFLE_DISTANCE,
            block_prefix=shuffle_prefix,
        )

        coop.store(block, exclusive_out, exclusive)
        coop.store(block, inclusive_out, inclusive)
        coop.store(block, xor_prefix_out, xor_prefix)
        coop.store(block, sum_aggregate_out, sum_aggregate)
        coop.store(block, xor_aggregate_out, xor_aggregate)
        sum_out[tidx] = total
        xor_out[tidx] = xor_total
        coop.store(block, diff_out, diff)
        coop.store(block, diff_right_out, diff_right)
        coop.store(block, head_out, head)
        coop.store(block, tail_out, tail)
        coop.store(block, head_pair_out, head_pair)
        coop.store(block, tail_pair_out, tail_pair)
        coop.store(block, shuffle_down_out, shuffled)
        coop.store(block, shuffle_prefix_out, shuffle_prefix)

    @cute.jit
    def _run_vector_block_prefix_segment(
        values_in: cute.Tensor,
        segments_in: cute.Tensor,
        exclusive_out: cute.Tensor,
        inclusive_out: cute.Tensor,
        xor_prefix_out: cute.Tensor,
        sum_aggregate_out: cute.Tensor,
        xor_aggregate_out: cute.Tensor,
        sum_out: cute.Tensor,
        xor_out: cute.Tensor,
        diff_out: cute.Tensor,
        diff_right_out: cute.Tensor,
        head_out: cute.Tensor,
        tail_out: cute.Tensor,
        head_pair_out: cute.Tensor,
        tail_pair_out: cute.Tensor,
        shuffle_down_out: cute.Tensor,
        shuffle_prefix_out: cute.Tensor,
    ):
        _vector_block_prefix_segment_kernel(
            values_in,
            segments_in,
            exclusive_out,
            inclusive_out,
            xor_prefix_out,
            sum_aggregate_out,
            xor_aggregate_out,
            sum_out,
            xor_out,
            diff_out,
            diff_right_out,
            head_out,
            tail_out,
            head_pair_out,
            tail_pair_out,
            shuffle_down_out,
            shuffle_prefix_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_block_prefix_segment,
        torch,
        from_dlpack,
        cutlass,
        scan_temp_storage,
        reduce_temp_storage,
        segment_temp_storage,
        shuffle_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_scan(values: Any, *, torch: Any) -> tuple[Any, Any]:
    inclusive = torch.cumsum(values.to(torch.int64), dim=0).to(torch.int32)
    exclusive = inclusive - values
    return exclusive, inclusive


def _expected_xor_scan(values: Any, *, torch: Any) -> Any:
    running_xor = 0
    result = []
    for value in values.tolist():
        running_xor ^= int(value)
        result.append(running_xor)
    return torch.tensor(result, dtype=torch.int32)


def _expected_reduce(values: Any, *, block_threads: int, torch: Any) -> tuple[Any, Any]:
    running_xor = 0
    for value in values.tolist():
        running_xor ^= int(value)
    total = torch.full(
        (block_threads,),
        int(values.to(torch.int64).sum().item()),
        dtype=torch.int32,
    )
    xor_total = torch.full((block_threads,), running_xor, dtype=torch.int32)
    return total, xor_total


def _expected_diff_heads_tails(values: Any, *, torch: Any) -> tuple[Any, Any, Any, Any]:
    diff = values.clone()
    diff[1:] = values[1:] - values[:-1]
    diff_right = values.clone()
    diff_right[:-1] = values[:-1] - values[1:]
    heads = torch.zeros((int(values.numel()),), dtype=torch.int32)
    tails = torch.zeros((int(values.numel()),), dtype=torch.int32)
    heads[0] = 1
    tails[-1] = 1
    for idx in range(1, int(values.numel())):
        heads[idx] = int(values[idx] != values[idx - 1])
    for idx in range(0, int(values.numel()) - 1):
        tails[idx] = int(values[idx] != values[idx + 1])
    return diff, diff_right, heads, tails


def _expected_shuffle_down(values: Any, *, distance: int) -> Any:
    shuffled = values.clone()
    shuffled[:-distance] = values[distance:]
    return shuffled


def _expected_shuffle_prefix(values: Any, *, block_threads: int, torch: Any) -> Any:
    return torch.full((block_threads,), int(values[0].item()), dtype=torch.int32)


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_block_prefix_segment,
        torch,
        from_dlpack,
        cutlass,
        scan_temp_storage,
        reduce_temp_storage,
        segment_temp_storage,
        shuffle_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    scan_temp_storage.reset_uses()
    reduce_temp_storage.reset_uses()
    segment_temp_storage.reset_uses()
    shuffle_temp_storage.reset_uses()

    values_host = ((torch.arange(TOTAL_ITEMS, dtype=torch.int64) % 17) + 1).to(
        torch.int32
    )
    segments_host = torch.tensor(
        [((idx // 3) + (idx % 11 == 0)) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )

    values_in = values_host.cuda()
    segments_in = segments_host.cuda()
    exclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    inclusive_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    xor_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    sum_aggregate_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    xor_aggregate_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    sum_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    xor_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    diff_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    diff_right_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    head_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    tail_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    head_pair_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    tail_pair_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    shuffle_down_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    shuffle_prefix_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")

    values_arg = from_dlpack(values_in)
    segments_arg = from_dlpack(segments_in)
    exclusive_arg = from_dlpack(exclusive_out)
    inclusive_arg = from_dlpack(inclusive_out)
    xor_prefix_arg = from_dlpack(xor_prefix_out)
    sum_aggregate_arg = from_dlpack(sum_aggregate_out)
    xor_aggregate_arg = from_dlpack(xor_aggregate_out)
    sum_arg = from_dlpack(sum_out)
    xor_arg = from_dlpack(xor_out)
    diff_arg = from_dlpack(diff_out)
    diff_right_arg = from_dlpack(diff_right_out)
    head_arg = from_dlpack(head_out)
    tail_arg = from_dlpack(tail_out)
    head_pair_arg = from_dlpack(head_pair_out)
    tail_pair_arg = from_dlpack(tail_pair_out)
    shuffle_down_arg = from_dlpack(shuffle_down_out)
    shuffle_prefix_arg = from_dlpack(shuffle_prefix_out)

    def step() -> None:
        run_vector_block_prefix_segment(
            values_arg,
            segments_arg,
            exclusive_arg,
            inclusive_arg,
            xor_prefix_arg,
            sum_aggregate_arg,
            xor_aggregate_arg,
            sum_arg,
            xor_arg,
            diff_arg,
            diff_right_arg,
            head_arg,
            tail_arg,
            head_pair_arg,
            tail_pair_arg,
            shuffle_down_arg,
            shuffle_prefix_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_exclusive, expected_inclusive = _expected_scan(
            values_host,
            torch=torch,
        )
        expected_xor_prefix = _expected_xor_scan(values_host, torch=torch)
        expected_sum, expected_xor = _expected_reduce(
            values_host,
            block_threads=BLOCK_THREADS,
            torch=torch,
        )
        expected_diff, expected_diff_right, expected_head, expected_tail = (
            _expected_diff_heads_tails(
                segments_host,
                torch=torch,
            )
        )
        expected_shuffle_down = _expected_shuffle_down(
            values_host,
            distance=SHUFFLE_DISTANCE,
        )
        expected_shuffle_prefix = _expected_shuffle_prefix(
            values_host,
            block_threads=BLOCK_THREADS,
            torch=torch,
        )

        torch.testing.assert_close(
            exclusive_out.cpu(), expected_exclusive, atol=0, rtol=0
        )
        torch.testing.assert_close(
            inclusive_out.cpu(), expected_inclusive, atol=0, rtol=0
        )
        torch.testing.assert_close(
            xor_prefix_out.cpu(),
            expected_xor_prefix,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            sum_aggregate_out.cpu(),
            expected_sum,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            xor_aggregate_out.cpu(),
            expected_xor,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(sum_out.cpu(), expected_sum, atol=0, rtol=0)
        torch.testing.assert_close(xor_out.cpu(), expected_xor, atol=0, rtol=0)
        torch.testing.assert_close(diff_out.cpu(), expected_diff, atol=0, rtol=0)
        torch.testing.assert_close(
            diff_right_out.cpu(),
            expected_diff_right,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(head_out.cpu(), expected_head, atol=0, rtol=0)
        torch.testing.assert_close(tail_out.cpu(), expected_tail, atol=0, rtol=0)
        torch.testing.assert_close(head_pair_out.cpu(), expected_head, atol=0, rtol=0)
        torch.testing.assert_close(tail_pair_out.cpu(), expected_tail, atol=0, rtol=0)
        torch.testing.assert_close(
            shuffle_down_out.cpu(),
            expected_shuffle_down,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            shuffle_prefix_out.cpu(),
            expected_shuffle_prefix,
            atol=0,
            rtol=0,
        )

        return {
            "exclusive": [int(x) for x in exclusive_out[:8].cpu().tolist()],
            "inclusive": [int(x) for x in inclusive_out[:8].cpu().tolist()],
            "xor_prefix": [int(x) for x in xor_prefix_out[:8].cpu().tolist()],
            "sum_aggregate": int(sum_aggregate_out[0].cpu().item()),
            "xor_aggregate": int(xor_aggregate_out[0].cpu().item()),
            "sum": int(sum_out[0].cpu().item()),
            "xor": int(xor_out[0].cpu().item()),
            "diff": [int(x) for x in diff_out[:8].cpu().tolist()],
            "diff_right": [int(x) for x in diff_right_out[:8].cpu().tolist()],
            "heads": [int(x) for x in head_out[:8].cpu().tolist()],
            "tails": [int(x) for x in tail_out[:8].cpu().tolist()],
            "shuffle_down": [int(x) for x in shuffle_down_out[:8].cpu().tolist()],
            "shuffle_prefix": int(shuffle_prefix_out[0].cpu().item()),
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the Prims vector scan/reduce/segment example and validate it."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
