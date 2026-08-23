# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS warp merge-sort example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
WARP_THREADS = 32
WARP_ITEMS = WARP_THREADS * ITEMS_PER_THREAD
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    return cutlass, cute, torch, from_dlpack, coop


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the Prims warp merge-sort JIT runner."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop

    @cute.kernel
    def _vector_warp_merge_sort_kernel(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        desc_keys_out: cute.Tensor,
        pair_keys_out: cute.Tensor,
        pair_values_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        warp = coop.this_warp()
        keys_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        values_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        coop.load(
            warp,
            keys_in,
            keys_vec,
        )
        coop.load(
            warp,
            values_in,
            values_vec,
        )

        desc_keys = coop.merge_sort_keys(
            warp,
            keys_vec,
            compare_op=">",
        )
        pair_keys, pair_values = coop.merge_sort_pairs(
            warp,
            keys_vec,
            values_vec,
        )

        coop.store(
            warp,
            desc_keys_out,
            desc_keys,
        )
        coop.store(
            warp,
            pair_keys_out,
            pair_keys,
        )
        coop.store(
            warp,
            pair_values_out,
            pair_values,
        )

    @cute.jit
    def _run_vector_warp_merge_sort(
        keys_in: cute.Tensor,
        values_in: cute.Tensor,
        desc_keys_out: cute.Tensor,
        pair_keys_out: cute.Tensor,
        pair_values_out: cute.Tensor,
    ):
        _vector_warp_merge_sort_kernel(
            keys_in,
            values_in,
            desc_keys_out,
            pair_keys_out,
            pair_values_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_vector_warp_merge_sort, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _make_unique_warp_keys(*, torch: Any) -> Any:
    keys = []
    for warp_base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        keys.extend(
            warp_base + ((local_idx * 17 + 5) % WARP_ITEMS)
            for local_idx in range(WARP_ITEMS)
        )
    return torch.tensor(keys, dtype=torch.int32)


def _sort_warp_tiles(values: Any, *, torch: Any, descending: bool = False) -> Any:
    sorted_tiles = []
    for warp_base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        tile = values[warp_base : warp_base + WARP_ITEMS]
        sorted_tiles.append(torch.sort(tile, descending=descending).values)
    return torch.cat(sorted_tiles).to(values.dtype)


def _sort_warp_pairs(keys: Any, values: Any, *, torch: Any) -> tuple[Any, Any]:
    sorted_keys = []
    sorted_values = []
    for warp_base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        order = torch.tensor(
            sorted(
                range(WARP_ITEMS),
                key=lambda idx, warp_base=warp_base: int(keys[warp_base + idx].item()),
            ),
            dtype=torch.long,
        )
        sorted_keys.append(keys[warp_base : warp_base + WARP_ITEMS][order])
        sorted_values.append(values[warp_base : warp_base + WARP_ITEMS][order])
    return torch.cat(sorted_keys), torch.cat(sorted_values)


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_vector_warp_merge_sort, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    keys_host = _make_unique_warp_keys(torch=torch)
    values_host = torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3 + 11
    keys_in = keys_host.cuda()
    values_in = values_host.cuda()
    desc_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    pair_keys_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    pair_values_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    values_arg = from_dlpack(values_in)
    desc_keys_arg = from_dlpack(desc_keys_out)
    pair_keys_arg = from_dlpack(pair_keys_out)
    pair_values_arg = from_dlpack(pair_values_out)

    def step() -> None:
        run_vector_warp_merge_sort(
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
        expected_desc_keys = _sort_warp_tiles(keys_host, torch=torch, descending=True)
        expected_pair_keys, expected_pair_values = _sort_warp_pairs(
            keys_host,
            values_host,
            torch=torch,
        )

        torch.testing.assert_close(
            desc_keys_out.cpu(),
            expected_desc_keys,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            pair_keys_out.cpu(),
            expected_pair_keys,
            atol=0,
            rtol=0,
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
    """Run the Prims vector warp merge-sort example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
