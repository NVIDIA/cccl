# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block TopK score-window example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 3
TOPK_K = 9
BEGIN_BIT = 0
END_BIT = 8


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

    if descending:
        order = sorted(range(len(key_sig)), key=lambda idx: (-int(key_sig[idx]), idx))
    else:
        order = sorted(range(len(key_sig)), key=lambda idx: (int(key_sig[idx]), idx))
    index = values.new_tensor(order).long()
    return keys[index], values[index]


def _assert_topk_keys_unordered(actual_keys: Any, expected_keys: Any) -> None:
    import torch

    torch.testing.assert_close(
        torch.sort(actual_keys.cpu()).values,
        torch.sort(expected_keys.cpu()).values,
        atol=0,
        rtol=0,
    )


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


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _topk_score_window_kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        max_keys_out: cute.Tensor,
        min_pair_keys_out: cute.Tensor,
        min_pair_values_out: cute.Tensor,
    ):
        block = coop.this_block()
        keys = coop.ThreadData(ITEMS_PER_THREAD, Int32)
        values = coop.ThreadData(ITEMS_PER_THREAD, Int32)
        coop.load(
            block,
            keys_in,
            keys,
        )
        coop.load(
            block,
            values_in,
            values,
        )
        k = cutlass.const_expr(TOPK_K)

        top_keys = coop.topk_max_keys(
            block,
            keys,
            k,
            begin_bit=BEGIN_BIT,
            end_bit=END_BIT,
        )
        pair_keys, pair_values = coop.topk_min_pairs(
            block,
            keys,
            values,
            k,
            begin_bit=BEGIN_BIT,
            end_bit=END_BIT,
        )

        coop.store(
            block,
            max_keys_out,
            top_keys,
        )
        coop.store(
            block,
            min_pair_keys_out,
            pair_keys,
        )
        coop.store(
            block,
            min_pair_values_out,
            pair_values,
        )

    @cute.jit
    def _run_topk_score_window(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        max_keys_out: cute.Tensor,
        min_pair_keys_out: cute.Tensor,
        min_pair_values_out: cute.Tensor,
    ):
        _topk_score_window_kernel(
            keys_in,
            values_in,
            max_keys_out,
            min_pair_keys_out,
            min_pair_values_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_topk_score_window, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_topk_score_window, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    total_items = BLOCK_THREADS * ITEMS_PER_THREAD
    host_keys = torch.tensor(
        [((idx * 29 + 17) % 251) for idx in range(total_items)],
        dtype=torch.int32,
    )
    host_values = torch.arange(total_items, dtype=torch.int32) * 7 + 3
    keys_in = host_keys.cuda()
    values_in = host_values.cuda()
    # ThreadData TopK returns a ThreadData-shaped tile. The selected TopK
    # values occupy the first TOPK_K flattened positions; the rest of the tile
    # is outside the logical TopK result but still needs storage for block.store.
    max_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    min_pair_keys_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    min_pair_values_out = torch.zeros((total_items,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    values_arg = from_dlpack(values_in)
    max_keys_arg = from_dlpack(max_keys_out)
    min_pair_keys_arg = from_dlpack(min_pair_keys_out)
    min_pair_values_arg = from_dlpack(min_pair_values_out)

    def step() -> None:
        run_topk_score_window(
            keys_arg,
            values_arg,
            max_keys_arg,
            min_pair_keys_arg,
            min_pair_values_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_max_keys, _ = _expected_radix_order(
            host_keys,
            host_values,
            begin_bit=BEGIN_BIT,
            end_bit=END_BIT,
            descending=True,
        )
        expected_min_keys, expected_min_values = _expected_radix_order(
            host_keys,
            host_values,
            begin_bit=BEGIN_BIT,
            end_bit=END_BIT,
            descending=False,
        )

        result_window = slice(0, TOPK_K)

        _assert_topk_keys_unordered(
            max_keys_out[result_window],
            expected_max_keys[result_window],
        )
        _assert_topk_pairs_unordered(
            min_pair_keys_out[result_window],
            min_pair_values_out[result_window],
            expected_min_keys[result_window],
            expected_min_values[result_window],
        )

        return {
            "topk_k": TOPK_K,
            "max_keys": [int(x) for x in max_keys_out[result_window].cpu().tolist()],
            "min_pair_keys": [
                int(x) for x in min_pair_keys_out[result_window].cpu().tolist()
            ],
            "min_pair_values": [
                int(x) for x in min_pair_values_out[result_window].cpu().tolist()
            ],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the TopK score-window example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
