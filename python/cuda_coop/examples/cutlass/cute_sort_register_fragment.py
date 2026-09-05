# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CuTe rmem-fragment sort example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

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
    def _sort_register_fragment_kernel(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        base = tidx * items_per_thread

        key_fragment = cute.make_rmem_tensor((items_per_thread,), Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            key_fragment[item] = keys_in[base + item]
        block = coop.this_block()
        sorted_keys = coop.radix_sort_keys(
            block,
            key_fragment,
            begin_bit=0,
            end_bit=32,
            descending=False,
            temp_storage=temp_storage,
        )

        coop.store(
            block,
            sorted_key_out,
            sorted_keys,
        )

    @cute.jit
    def _run_sort_register_fragment(
        keys_in: cute.Tensor,
        sorted_key_out: cute.Tensor,
    ):
        _sort_register_fragment_kernel(
            keys_in,
            sorted_key_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_sort_register_fragment, torch, from_dlpack, cutlass, temp_storage


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_sort, torch, from_dlpack, cutlass, temp_storage = make_runner()
    cutlass.cuda.initialize_cuda_context()
    temp_storage.reset_uses()

    keys_host = torch.tensor(
        [((idx * 19 + 7) % 47) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    keys_in = keys_host.cuda()
    sorted_key_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    keys_arg = from_dlpack(keys_in)
    sorted_key_arg = from_dlpack(sorted_key_out)

    def step() -> None:
        run_sort(keys_arg, sorted_key_arg)

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_sorted = torch.sort(keys_host).values
        torch.testing.assert_close(
            sorted_key_out.cpu(),
            expected_sorted,
            atol=0,
            rtol=0,
        )
        return {
            "sorted_key": [int(x) for x in sorted_key_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the CuTe rmem-fragment sort example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
