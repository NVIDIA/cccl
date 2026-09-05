# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS radix-rank/merge-sort example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
MERGE_OOB_DEFAULT = -1000000


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)
    return cutlass, cute, torch, from_dlpack, coop, Int32


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build and return the Prims rank/merge JIT runner plus helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    radix_temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")
    merge_temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")

    @cute.kernel
    def _vector_rank_merge_kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        rank_out: cute.Tensor,
        prefix_out: cute.Tensor,
        merge_keys_out: cute.Tensor,
        merge_values_out: cute.Tensor,
        merge_keys_only_out: cute.Tensor,
        valid_items: cutlass.Int32,
        begin_bit: cutlass.Constexpr,
        end_bit: cutlass.Constexpr,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys_vec = coop.ThreadData(items_per_thread, Int32)
        values_vec = coop.ThreadData(items_per_thread, Int32)
        coop.load(
            block,
            keys_in,
            keys_vec,
        )
        coop.load(
            block,
            values_in,
            values_vec,
        )

        exclusive_digit_prefix = coop.ThreadData(1, dtype=Int32)
        ranks = coop.radix_rank(
            block,
            keys_vec,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            exclusive_digit_prefix=exclusive_digit_prefix,
        )
        merge_keys, merge_values = coop.merge_sort_pairs(
            block,
            keys_vec,
            values_vec,
            descending=True,
            valid_items=valid_items,
            oob_default=Int32(MERGE_OOB_DEFAULT),
            temp_storage=merge_temp_storage,
        )
        merge_keys_only = coop.merge_sort_keys(
            block,
            keys_vec,
            descending=True,
            valid_items=valid_items,
            oob_default=Int32(MERGE_OOB_DEFAULT),
            temp_storage=merge_temp_storage,
        )

        coop.store(block, rank_out, ranks)
        coop.store(block, prefix_out, exclusive_digit_prefix)
        coop.store(block, merge_keys_out, merge_keys)
        coop.store(block, merge_values_out, merge_values)
        coop.store(block, merge_keys_only_out, merge_keys_only)

    @cute.jit
    def _run_vector_rank_merge(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        rank_out: cute.Tensor,
        prefix_out: cute.Tensor,
        merge_keys_out: cute.Tensor,
        merge_values_out: cute.Tensor,
        merge_keys_only_out: cute.Tensor,
        valid_items: cutlass.Int32,
        begin_bit: cutlass.Constexpr,
        end_bit: cutlass.Constexpr,
    ):
        _vector_rank_merge_kernel(
            keys_in,
            values_in,
            rank_out,
            prefix_out,
            merge_keys_out,
            merge_values_out,
            merge_keys_only_out,
            valid_items,
            begin_bit,
            end_bit,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_rank_merge,
        torch,
        from_dlpack,
        cutlass,
        radix_temp_storage,
        merge_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_radix_ranks(
    keys: Any,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    torch: Any,
) -> Any:
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = (
        0 if keys.dtype in {torch.uint32, torch.uint64} else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    mask = (1 << (end_bit - begin_bit)) - 1
    digits = [int((value >> begin_bit) & mask) for value in ordered]
    ranks = torch.empty((len(digits),), dtype=torch.int32)
    for idx, digit in enumerate(digits):
        rank = 0
        for peer_idx, peer_digit in enumerate(digits):
            before = peer_digit > digit if descending else peer_digit < digit
            if before or (peer_digit == digit and peer_idx < idx):
                rank += 1
        ranks[idx] = rank
    return ranks


def _expected_radix_digit_prefix(
    keys: Any,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    block_threads: int,
    bins_per_thread: int,
    torch: Any,
) -> Any:
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = (
        0 if keys.dtype in {torch.uint32, torch.uint64} else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    radix_digits = 1 << (end_bit - begin_bit)
    mask = radix_digits - 1
    digits = [int((value >> begin_bit) & mask) for value in ordered]

    counts = [0 for _ in range(radix_digits)]
    for digit in digits:
        counts[digit] += 1

    prefix = [0 for _ in range(radix_digits)]
    running = 0
    digit_iter = range(radix_digits - 1, -1, -1) if descending else range(radix_digits)
    for digit in digit_iter:
        prefix[digit] = running
        running += counts[digit]

    expected = torch.full(
        (block_threads, bins_per_thread),
        -1,
        dtype=torch.int32,
    )
    for tid in range(block_threads):
        for track in range(bins_per_thread):
            bin_idx = tid * bins_per_thread + track
            if block_threads == radix_digits or bin_idx < radix_digits:
                expected[tid, track] = prefix[bin_idx]
    return expected.reshape(-1)


def _gather_cpu_tensor(tensor: Any, indices: list[int], *, torch: Any) -> Any:
    values = tensor.tolist()
    return torch.tensor([values[index] for index in indices], dtype=tensor.dtype)


def _expected_merge_order(
    keys: Any,
    values: Any,
    *,
    descending: bool,
    torch: Any,
) -> tuple[Any, Any]:
    key_values = keys.tolist()
    idx = list(range(len(key_values)))
    if descending:
        idx = sorted(idx, key=lambda i: (-int(key_values[i]), i))
    else:
        idx = sorted(idx, key=lambda i: (int(key_values[i]), i))
    return (
        _gather_cpu_tensor(keys, idx, torch=torch),
        _gather_cpu_tensor(values, idx, torch=torch),
    )


def _expected_merge_order_partial(
    keys: Any,
    values: Any,
    *,
    descending: bool,
    valid_items: int,
    oob_default: int,
    torch: Any,
) -> tuple[Any, Any]:
    expected_keys = torch.full_like(keys, oob_default)
    expected_values = torch.empty_like(values)
    sorted_keys, sorted_values = _expected_merge_order(
        keys[:valid_items],
        values[:valid_items],
        descending=descending,
        torch=torch,
    )
    expected_keys[:valid_items] = sorted_keys
    expected_values[:valid_items] = sorted_values
    expected_values[valid_items:] = values[valid_items:]
    return expected_keys, expected_values


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_rank_merge,
        torch,
        from_dlpack,
        cutlass,
        radix_temp_storage,
        merge_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    radix_temp_storage.reset_uses()
    merge_temp_storage.reset_uses()

    begin_bit = 0
    end_bit = 4
    valid_items = TOTAL_ITEMS - 9
    keys_host = torch.tensor(
        [((idx * 11 + (idx % 7) * 5) % 53) - 26 for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 7 + 3
    keys_in = keys_host.cuda()
    values_in = values_host.cuda()
    rank_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    prefix_out = torch.zeros((BLOCK_THREADS,), dtype=torch.int32, device="cuda")
    merge_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    merge_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    merge_keys_only_out = torch.zeros(
        (TOTAL_ITEMS,),
        dtype=torch.int32,
        device="cuda",
    )
    keys_arg = from_dlpack(keys_in)
    values_arg = from_dlpack(values_in)
    rank_arg = from_dlpack(rank_out)
    prefix_arg = from_dlpack(prefix_out)
    merge_keys_arg = from_dlpack(merge_keys_out)
    merge_values_arg = from_dlpack(merge_values_out)
    merge_keys_only_arg = from_dlpack(merge_keys_only_out)

    def step() -> None:
        run_vector_rank_merge(
            keys_arg,
            values_arg,
            rank_arg,
            prefix_arg,
            merge_keys_arg,
            merge_values_arg,
            merge_keys_only_arg,
            cutlass.Int32(valid_items),
            begin_bit,
            end_bit,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_ranks = _expected_radix_ranks(
            keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )
        expected_prefix = _expected_radix_digit_prefix(
            keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            block_threads=BLOCK_THREADS,
            bins_per_thread=1,
            torch=torch,
        )
        expected_merge_keys, expected_merge_values = _expected_merge_order_partial(
            keys_host,
            values_host,
            descending=True,
            valid_items=valid_items,
            oob_default=MERGE_OOB_DEFAULT,
            torch=torch,
        )
        torch.testing.assert_close(rank_out.cpu(), expected_ranks, atol=0, rtol=0)
        torch.testing.assert_close(prefix_out.cpu(), expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(
            merge_keys_out[:valid_items].cpu(),
            expected_merge_keys[:valid_items],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            merge_keys_only_out[:valid_items].cpu(),
            expected_merge_keys[:valid_items],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            merge_values_out[:valid_items].cpu(),
            expected_merge_values[:valid_items],
            atol=0,
            rtol=0,
        )

        return {
            "ranks": [int(x) for x in rank_out[:8].cpu().tolist()],
            "prefix": [int(x) for x in prefix_out[:8].cpu().tolist()],
            "merge_pairs": [
                (int(key), int(value))
                for key, value in zip(
                    merge_keys_out[:8].cpu().tolist(),
                    merge_values_out[:8].cpu().tolist(),
                    strict=True,
                )
            ],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the Prims vector rank/merge example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
