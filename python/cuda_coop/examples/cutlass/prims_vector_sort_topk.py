# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS block sort/TopK example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
TOPK_K = 5
TOPK_VALID_ITEMS = TOTAL_ITEMS - 9


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    return cutlass, cute, torch, from_dlpack, coop


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build and return the Prims vector JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    radix_temp_storage = coop.TempStorage(size_in_bytes=8192, sharing="shared")
    topk_temp_storage = coop.TempStorage(size_in_bytes=16384, sharing="shared")

    @cute.kernel
    def _vector_sort_topk_kernel(
        keys_in: cute.Tensor,
        sorted_keys_out: cute.Tensor,
        top_keys_out: cute.Tensor,
        topk_k: cutlass.Int32,
        num_valid: cutlass.Int32,
        begin_bit: cutlass.Int32,
        end_bit: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        keys_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        coop.load(
            block,
            keys_in,
            keys_vec,
        )

        sorted_keys = coop.radix_sort_keys(
            block,
            keys_vec,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            temp_storage=radix_temp_storage,
        )
        top_keys = coop.topk_max_keys(
            block,
            keys_vec,
            topk_k,
            valid_items=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )

        coop.store(block, sorted_keys_out, sorted_keys)
        coop.store(block, top_keys_out, top_keys)

    @cute.jit
    def _run_vector_sort_topk(
        keys_in: cute.Tensor,
        sorted_keys_out: cute.Tensor,
        top_keys_out: cute.Tensor,
        topk_k: cutlass.Int32,
        num_valid: cutlass.Int32,
        begin_bit: cutlass.Int32,
        end_bit: cutlass.Int32,
    ):
        _vector_sort_topk_kernel(
            keys_in,
            sorted_keys_out,
            top_keys_out,
            topk_k,
            num_valid,
            begin_bit,
            end_bit,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        radix_temp_storage,
        topk_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_radix_keys(
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
    key_sig = [int((value >> begin_bit) & mask) for value in ordered]
    idx = list(range(len(key_sig)))
    if descending:
        idx = sorted(idx, key=lambda i: (-int(key_sig[i]), i))
    else:
        idx = sorted(idx, key=lambda i: (int(key_sig[i]), i))
    return torch.tensor([keys.tolist()[index] for index in idx], dtype=keys.dtype)


def _assert_topk_keys_unordered(
    actual_keys: Any,
    expected_keys: Any,
    *,
    torch: Any,
) -> None:
    torch.testing.assert_close(
        torch.sort(actual_keys.cpu()).values,
        torch.sort(expected_keys.cpu()).values,
        atol=0,
        rtol=0,
    )


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_sort_topk,
        torch,
        from_dlpack,
        cutlass,
        radix_temp_storage,
        topk_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    radix_temp_storage.reset_uses()
    topk_temp_storage.reset_uses()

    begin_bit = 0
    end_bit = 8
    keys_host = torch.tensor(
        [((idx * 37 + (idx % 13) * 5) % 251) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    keys_in = keys_host.cuda()
    sorted_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    top_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    sorted_keys_arg = from_dlpack(sorted_keys_out)
    top_keys_arg = from_dlpack(top_keys_out)

    def step() -> None:
        run_vector_sort_topk(
            keys_arg,
            sorted_keys_arg,
            top_keys_arg,
            cutlass.Int32(TOPK_K),
            cutlass.Int32(TOPK_VALID_ITEMS),
            cutlass.Int32(begin_bit),
            cutlass.Int32(end_bit),
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_sorted = _expected_radix_keys(
            keys_host,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=False,
            torch=torch,
        )
        expected_top = _expected_radix_keys(
            keys_host[:TOPK_VALID_ITEMS],
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=True,
            torch=torch,
        )
        torch.testing.assert_close(
            sorted_keys_out.cpu(),
            expected_sorted,
            atol=0,
            rtol=0,
        )
        _assert_topk_keys_unordered(
            top_keys_out[:TOPK_K],
            expected_top[:TOPK_K],
            torch=torch,
        )

        return {
            "topk_valid_items": TOPK_VALID_ITEMS,
            "sorted_keys": [int(x) for x in sorted_keys_out.cpu().tolist()],
            "top_keys": [int(x) for x in top_keys_out[:TOPK_K].cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the Prims vector sort/top-k example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
