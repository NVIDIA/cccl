# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp merge-sort example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

WARP_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = WARP_THREADS * ITEMS_PER_THREAD


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _warp_merge_sort_kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        desc_keys_out: cute.Tensor,
        pair_keys_out: cute.Tensor,
        pair_values_out: cute.Tensor,
    ):
        group = coop.this_warp()
        keys = coop.ThreadData(ITEMS_PER_THREAD, Int32)
        values = coop.ThreadData(ITEMS_PER_THREAD, Int32)
        coop.load(
            group,
            keys_in,
            keys,
        )
        coop.load(
            group,
            values_in,
            values,
        )

        desc_keys = coop.merge_sort_keys(
            group,
            keys,
            compare_op=">",
        )
        pair_keys, pair_values = coop.merge_sort_pairs(
            group,
            keys,
            values,
        )

        coop.store(
            group,
            desc_keys_out,
            desc_keys,
        )
        coop.store(
            group,
            pair_keys_out,
            pair_keys,
        )
        coop.store(
            group,
            pair_values_out,
            pair_values,
        )

    @cute.jit
    def _run_warp_merge_sort(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        desc_keys_out: cute.Tensor,
        pair_keys_out: cute.Tensor,
        pair_values_out: cute.Tensor,
    ):
        _warp_merge_sort_kernel(
            keys_in,
            values_in,
            desc_keys_out,
            pair_keys_out,
            pair_values_out,
        ).launch(grid=(1, 1, 1), block=(WARP_THREADS, 1, 1))

    return _run_warp_merge_sort, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_warp_merge_sort, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    host_keys = torch.tensor(
        [((idx * 17 + 5) % TOTAL_ITEMS) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    host_values = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3 + 11
    keys_in = host_keys.cuda()
    values_in = host_values.cuda()
    desc_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    pair_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    values_arg = from_dlpack(values_in)
    desc_keys_arg = from_dlpack(desc_keys_out)
    pair_keys_arg = from_dlpack(pair_keys_out)
    pair_values_arg = from_dlpack(pair_values_out)

    def step() -> None:
        run_warp_merge_sort(
            keys_arg,
            values_arg,
            desc_keys_arg,
            pair_keys_arg,
            pair_values_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        asc_order = torch.tensor(
            sorted(range(TOTAL_ITEMS), key=lambda idx: int(host_keys[idx].item())),
            dtype=torch.long,
        )
        expected_pair_keys = host_keys[asc_order]
        expected_pair_values = host_values[asc_order]
        expected_desc_keys = torch.sort(host_keys, descending=True).values

        torch.testing.assert_close(
            desc_keys_out.cpu(), expected_desc_keys, atol=0, rtol=0
        )
        torch.testing.assert_close(
            pair_keys_out.cpu(), expected_pair_keys, atol=0, rtol=0
        )
        torch.testing.assert_close(
            pair_values_out.cpu(),
            expected_pair_values,
            atol=0,
            rtol=0,
        )

        return {
            "desc_keys_out": [int(x) for x in desc_keys_out.cpu().tolist()],
            "pair_keys_out": [int(x) for x in pair_keys_out.cpu().tolist()],
            "pair_values_out": [int(x) for x in pair_values_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the warp merge-sort example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


if __name__ == "__main__":
    print(run_example())
