# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first key/value sorting example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
TOPK_K = 7
TOPK_VALID_ITEMS = TOTAL_ITEMS - 11


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Build and return the pair sort/TopK JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    topk_temp_storage = coop.TempStorage(size_in_bytes=16384, sharing="shared")

    @cute.kernel
    def _vector_pair_sort_topk_kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        sorted_keys_out: cute.Tensor,
        sorted_values_out: cute.Tensor,
        top_pair_keys_out: cute.Tensor,
        top_pair_values_out: cute.Tensor,
        topk_k: Int32,
        num_valid: Int32,
        begin_bit: Int32,
        end_bit: Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys_vec = coop.ThreadData(
            items_per_thread=items_per_thread,
            dtype=Int32,
        )
        values_vec = coop.ThreadData(
            items_per_thread=items_per_thread,
            dtype=Int32,
        )
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

        # docs: start cutlass-radix-sort-pairs
        sorted_keys, sorted_values = coop.radix_sort_pairs(
            block,
            keys_vec,
            values_vec,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
        )
        # docs: end cutlass-radix-sort-pairs
        # docs: start cutlass-topk-partial
        top_pair_keys, top_pair_values = coop.topk_min_pairs(
            block,
            keys_vec,
            values_vec,
            topk_k,
            valid_items=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )
        # docs: end cutlass-topk-partial

        coop.store(block, sorted_keys_out, sorted_keys)
        coop.store(block, sorted_values_out, sorted_values)
        coop.store(block, top_pair_keys_out, top_pair_keys)
        coop.store(block, top_pair_values_out, top_pair_values)

    @cute.jit
    def _run_vector_pair_sort_topk(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        sorted_keys_out: cute.Tensor,
        sorted_values_out: cute.Tensor,
        top_pair_keys_out: cute.Tensor,
        top_pair_values_out: cute.Tensor,
        topk_k: Int32,
        num_valid: Int32,
        begin_bit: Int32,
        end_bit: Int32,
    ):
        _vector_pair_sort_topk_kernel(
            keys_in,
            values_in,
            sorted_keys_out,
            sorted_values_out,
            top_pair_keys_out,
            top_pair_values_out,
            topk_k,
            num_valid,
            begin_bit,
            end_bit,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_pair_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        topk_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_radix_pairs(
    keys: Any,
    values: Any,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    torch: Any,
) -> tuple[Any, Any]:
    width_bits = 64 if keys.dtype in {torch.int64, torch.uint64} else 32
    full_mask = (1 << width_bits) - 1
    raw_unsigned = [(int(key) & full_mask) for key in keys.tolist()]
    sign_flip = (
        0 if keys.dtype in {torch.uint32, torch.uint64} else 1 << (width_bits - 1)
    )
    ordered = [value ^ sign_flip for value in raw_unsigned]
    mask = (1 << (end_bit - begin_bit)) - 1
    key_sig = [int((value >> begin_bit) & mask) for value in ordered]
    idx = list(range(len(key_sig)))
    if descending:
        idx = sorted(idx, key=lambda i: (-int(key_sig[i]), i))
    else:
        idx = sorted(idx, key=lambda i: (int(key_sig[i]), i))
    index = values.new_tensor(idx).long()
    return keys[index], values[index]


def _assert_topk_pairs_unordered(
    actual_keys: Any,
    actual_values: Any,
    expected_keys: Any,
    expected_values: Any,
) -> None:
    actual_pairs = sorted(
        zip(actual_keys.cpu().tolist(), actual_values.cpu().tolist(), strict=True)
    )
    expected_pairs = sorted(
        zip(expected_keys.cpu().tolist(), expected_values.cpu().tolist(), strict=True)
    )
    assert actual_pairs == expected_pairs


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_pair_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        topk_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    topk_temp_storage.reset_uses()

    begin_bit = 0
    end_bit = 8
    keys_host = torch.tensor(
        [((idx * 17 + 23) % 251) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 11 + 5
    keys_in = keys_host.cuda()
    values_in = values_host.cuda()
    sorted_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    sorted_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    top_pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    top_pair_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    values_arg = from_dlpack(values_in)
    sorted_keys_arg = from_dlpack(sorted_keys_out)
    sorted_values_arg = from_dlpack(sorted_values_out)
    top_pair_keys_arg = from_dlpack(top_pair_keys_out)
    top_pair_values_arg = from_dlpack(top_pair_values_out)

    def step() -> None:
        run_vector_pair_sort_topk(
            keys_arg,
            values_arg,
            sorted_keys_arg,
            sorted_values_arg,
            top_pair_keys_arg,
            top_pair_values_arg,
            cutlass.Int32(TOPK_K),
            cutlass.Int32(TOPK_VALID_ITEMS),
            cutlass.Int32(begin_bit),
            cutlass.Int32(end_bit),
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_sorted_keys, expected_sorted_values = _expected_radix_pairs(
            keys_host,
            values_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )
        torch.testing.assert_close(
            sorted_keys_out.cpu(),
            expected_sorted_keys,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            sorted_values_out.cpu(),
            expected_sorted_values,
            atol=0,
            rtol=0,
        )
        expected_top_keys, expected_top_values = _expected_radix_pairs(
            keys_host[:TOPK_VALID_ITEMS],
            values_host[:TOPK_VALID_ITEMS],
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )
        _assert_topk_pairs_unordered(
            top_pair_keys_out[:TOPK_K],
            top_pair_values_out[:TOPK_K],
            expected_top_keys[:TOPK_K],
            expected_top_values[:TOPK_K],
        )

        return {
            "topk_valid_items": TOPK_VALID_ITEMS,
            "sorted_pairs": [
                (int(key), int(value))
                for key, value in zip(
                    sorted_keys_out.cpu().tolist(),
                    sorted_values_out.cpu().tolist(),
                    strict=True,
                )
            ],
            "top_pairs": [
                (int(key), int(value))
                for key, value in zip(
                    top_pair_keys_out[:TOPK_K].cpu().tolist(),
                    top_pair_values_out[:TOPK_K].cpu().tolist(),
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
    """Run the pair sort/TopK example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


if __name__ == "__main__":
    print(run_example())
