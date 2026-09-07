# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS sort/TopK example mixing register payload types."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime
from examples.cutlass.prims_vector_sort_topk import (
    _assert_topk_keys_unordered,
    _expected_radix_keys,
)

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
TOPK_K = 5


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)
    return cutlass, cute, torch, from_dlpack, coop, Int32


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Build and return the mixed-payload JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    vector_sort_temp_storage = coop.TempStorage(
        size_in_bytes=8192,
        sharing="shared",
    )
    vector_topk_temp_storage = coop.TempStorage(
        size_in_bytes=16384,
        sharing="shared",
    )
    fragment_sort_temp_storage = coop.TempStorage(
        size_in_bytes=8192,
        sharing="shared",
    )

    @cute.kernel
    def _mixed_payload_sort_topk_kernel(
        vector_keys_in: cute.Tensor,
        fragment_keys_in: cute.Tensor,
        sorted_vector_keys_out: cute.Tensor,
        top_vector_keys_out: cute.Tensor,
        sorted_fragment_keys_out: cute.Tensor,
        topk_k: cutlass.Int32,
        num_valid: cutlass.Int32,
        begin_bit: cutlass.Int32,
        end_bit: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        cute_tidx, _, _ = cute.arch.thread_idx()
        fragment_base = cute_tidx * items_per_thread
        block = coop.this_block()

        vector_keys = coop.ThreadData(items_per_thread, Int32)
        coop.load(
            block,
            vector_keys_in,
            vector_keys,
        )
        fragment_keys = cute.make_rmem_tensor((items_per_thread,), Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            fragment_keys[item] = fragment_keys_in[fragment_base + item]

        sorted_vector_keys = coop.radix_sort_keys(
            block,
            vector_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            temp_storage=vector_sort_temp_storage,
        )
        top_vector_keys = coop.topk_max_keys(
            block,
            vector_keys,
            topk_k,
            valid_items=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=vector_topk_temp_storage,
        )
        sorted_fragment_keys = coop.radix_sort_keys(
            block,
            fragment_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            temp_storage=fragment_sort_temp_storage,
        )

        coop.store(
            block,
            sorted_vector_keys_out,
            sorted_vector_keys,
        )
        coop.store(
            block,
            top_vector_keys_out,
            top_vector_keys,
        )
        coop.store(block, sorted_fragment_keys_out, sorted_fragment_keys)

    @cute.jit
    def _run_mixed_payload_sort_topk(
        vector_keys_in: cute.Tensor,
        fragment_keys_in: cute.Tensor,
        sorted_vector_keys_out: cute.Tensor,
        top_vector_keys_out: cute.Tensor,
        sorted_fragment_keys_out: cute.Tensor,
        topk_k: cutlass.Int32,
        num_valid: cutlass.Int32,
        begin_bit: cutlass.Int32,
        end_bit: cutlass.Int32,
    ):
        _mixed_payload_sort_topk_kernel(
            vector_keys_in,
            fragment_keys_in,
            sorted_vector_keys_out,
            top_vector_keys_out,
            sorted_fragment_keys_out,
            topk_k,
            num_valid,
            begin_bit,
            end_bit,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_mixed_payload_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        vector_sort_temp_storage,
        vector_topk_temp_storage,
        fragment_sort_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_mixed_payload_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        vector_sort_temp_storage,
        vector_topk_temp_storage,
        fragment_sort_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    vector_sort_temp_storage.reset_uses()
    vector_topk_temp_storage.reset_uses()
    fragment_sort_temp_storage.reset_uses()

    begin_bit = 0
    end_bit = 8
    vector_keys_host = torch.tensor(
        [((idx * 37 + (idx % 13) * 5) % 251) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    fragment_keys_host = torch.tensor(
        [((idx * 19 + 7) % 239) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    vector_keys_in = vector_keys_host.cuda()
    fragment_keys_in = fragment_keys_host.cuda()
    sorted_vector_keys_out = torch.zeros(
        (TOTAL_ITEMS,),
        dtype=torch.int32,
        device="cuda",
    )
    top_vector_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    sorted_fragment_keys_out = torch.zeros(
        (TOTAL_ITEMS,),
        dtype=torch.int32,
        device="cuda",
    )
    vector_keys_arg = from_dlpack(vector_keys_in)
    fragment_keys_arg = from_dlpack(fragment_keys_in)
    sorted_vector_keys_arg = from_dlpack(sorted_vector_keys_out)
    top_vector_keys_arg = from_dlpack(top_vector_keys_out)
    sorted_fragment_keys_arg = from_dlpack(sorted_fragment_keys_out)

    def step() -> None:
        run_mixed_payload_sort_topk(
            vector_keys_arg,
            fragment_keys_arg,
            sorted_vector_keys_arg,
            top_vector_keys_arg,
            sorted_fragment_keys_arg,
            cutlass.Int32(TOPK_K),
            cutlass.Int32(TOTAL_ITEMS),
            cutlass.Int32(begin_bit),
            cutlass.Int32(end_bit),
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_vector_sorted = _expected_radix_keys(
            vector_keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )
        expected_vector_top = _expected_radix_keys(
            vector_keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=True,
            torch=torch,
        )
        expected_fragment_sorted = _expected_radix_keys(
            fragment_keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )

        torch.testing.assert_close(
            sorted_vector_keys_out.cpu(),
            expected_vector_sorted,
            atol=0,
            rtol=0,
        )
        _assert_topk_keys_unordered(
            top_vector_keys_out[:TOPK_K],
            expected_vector_top[:TOPK_K],
            torch=torch,
        )
        torch.testing.assert_close(
            sorted_fragment_keys_out.cpu(),
            expected_fragment_sorted,
            atol=0,
            rtol=0,
        )

        return {
            "sorted_vector_keys": [
                int(x) for x in sorted_vector_keys_out.cpu().tolist()
            ],
            "top_vector_keys": [
                int(x) for x in top_vector_keys_out[:TOPK_K].cpu().tolist()
            ],
            "sorted_fragment_keys": [
                int(x) for x in sorted_fragment_keys_out.cpu().tolist()
            ],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the mixed payload example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
