# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS block-exchange example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD


def _require_runtime() -> tuple[Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    return cutlass, cute, torch, from_dlpack, coop


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Build and return the Prims block-exchange runner plus helpers."""

    cutlass, cute, torch, from_dlpack, coop = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    exchange_temp_storage = coop.TempStorage(
        size_in_bytes=8192,
        sharing="shared",
    )

    @cute.kernel
    def _vector_block_exchange_kernel(
        blocked_values_in: cute.Tensor,
        striped_values_in: cute.Tensor,
        reverse_ranks_in: cute.Tensor,
        valid_flags_in: cute.Tensor,
        striped_to_blocked_out: cute.Tensor,
        blocked_to_striped_out: cute.Tensor,
        scatter_to_striped_out: cute.Tensor,
        scatter_flagged_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        blocked_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        striped_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        reverse_ranks_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        valid_flags_vec = coop.ThreadData(items_per_thread, cutlass.Int32)
        coop.load(
            block,
            blocked_values_in,
            blocked_vec,
        )
        coop.load(
            block,
            striped_values_in,
            striped_vec,
            algorithm="striped",
        )
        coop.load(
            block,
            reverse_ranks_in,
            reverse_ranks_vec,
        )
        coop.load(
            block,
            valid_flags_in,
            valid_flags_vec,
        )

        blocked = coop.exchange(
            block,
            striped_vec,
            mode="striped_to_blocked",
        )
        striped = coop.exchange(
            block,
            blocked_vec,
            mode="blocked_to_striped",
        )
        scatter_striped = coop.exchange(
            block,
            blocked_vec,
            mode="scatter_to_striped",
            ranks=reverse_ranks_vec,
        )
        scatter_flagged = coop.exchange(
            block,
            blocked_vec,
            mode="scatter_to_striped_flagged",
            ranks=reverse_ranks_vec,
            valid_flags=valid_flags_vec,
        )

        coop.store(block, striped_to_blocked_out, blocked)
        coop.store(block, blocked_to_striped_out, striped)
        coop.store(
            block,
            scatter_to_striped_out,
            scatter_striped,
            algorithm="striped",
        )
        coop.store(
            block,
            scatter_flagged_out,
            scatter_flagged,
            algorithm="striped",
        )

    @cute.jit
    def _run_vector_block_exchange(
        blocked_values_in: cute.Tensor,
        striped_values_in: cute.Tensor,
        reverse_ranks_in: cute.Tensor,
        valid_flags_in: cute.Tensor,
        striped_to_blocked_out: cute.Tensor,
        blocked_to_striped_out: cute.Tensor,
        scatter_to_striped_out: cute.Tensor,
        scatter_flagged_out: cute.Tensor,
    ):
        _vector_block_exchange_kernel(
            blocked_values_in,
            striped_values_in,
            reverse_ranks_in,
            valid_flags_in,
            striped_to_blocked_out,
            blocked_to_striped_out,
            scatter_to_striped_out,
            scatter_flagged_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_block_exchange,
        torch,
        from_dlpack,
        cutlass,
        exchange_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_blocked_to_striped(
    values: Any,
    *,
    block_threads: int,
    items_per_thread: int,
    torch: Any,
) -> Any:
    expected = torch.empty_like(values)
    for tid in range(block_threads):
        for item in range(items_per_thread):
            expected[tid * items_per_thread + item] = values[item * block_threads + tid]
    return expected


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_block_exchange,
        torch,
        from_dlpack,
        cutlass,
        exchange_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    exchange_temp_storage.reset_uses()

    blocked_values_host = (torch.arange(TOTAL_ITEMS, dtype=torch.int32) * 3) - 17
    striped_values_host = blocked_values_host.clone()
    blocked_to_striped_expected = _expected_blocked_to_striped(
        blocked_values_host,
        block_threads=BLOCK_THREADS,
        items_per_thread=ITEMS_PER_THREAD,
        torch=torch,
    )
    reverse_ranks_host = torch.arange(TOTAL_ITEMS - 1, -1, -1, dtype=torch.int32)
    valid_flags_host = torch.ones((TOTAL_ITEMS,), dtype=torch.int32)

    blocked_values_in = blocked_values_host.cuda()
    striped_values_in = striped_values_host.cuda()
    reverse_ranks_in = reverse_ranks_host.cuda()
    valid_flags_in = valid_flags_host.cuda()
    striped_to_blocked_out = torch.zeros(
        (TOTAL_ITEMS,), dtype=torch.int32, device="cuda"
    )
    blocked_to_striped_out = torch.zeros(
        (TOTAL_ITEMS,), dtype=torch.int32, device="cuda"
    )
    scatter_to_striped_out = torch.zeros(
        (TOTAL_ITEMS,), dtype=torch.int32, device="cuda"
    )
    scatter_flagged_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

    blocked_values_arg = from_dlpack(blocked_values_in)
    striped_values_arg = from_dlpack(striped_values_in)
    reverse_ranks_arg = from_dlpack(reverse_ranks_in)
    valid_flags_arg = from_dlpack(valid_flags_in)
    striped_to_blocked_arg = from_dlpack(striped_to_blocked_out)
    blocked_to_striped_arg = from_dlpack(blocked_to_striped_out)
    scatter_to_striped_arg = from_dlpack(scatter_to_striped_out)
    scatter_flagged_arg = from_dlpack(scatter_flagged_out)

    def step() -> None:
        run_vector_block_exchange(
            blocked_values_arg,
            striped_values_arg,
            reverse_ranks_arg,
            valid_flags_arg,
            striped_to_blocked_arg,
            blocked_to_striped_arg,
            scatter_to_striped_arg,
            scatter_flagged_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        reverse_expected = torch.flip(blocked_values_host, dims=(0,))

        torch.testing.assert_close(
            striped_to_blocked_out.cpu(),
            blocked_values_host,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            blocked_to_striped_out.cpu(),
            blocked_to_striped_expected,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            scatter_to_striped_out.cpu(),
            reverse_expected,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            scatter_flagged_out.cpu(),
            reverse_expected,
            atol=0,
            rtol=0,
        )

        return {
            "striped_to_blocked": [
                int(x) for x in striped_to_blocked_out[:8].cpu().tolist()
            ],
            "blocked_to_striped": [
                int(x) for x in blocked_to_striped_out[:8].cpu().tolist()
            ],
            "scatter_to_striped": [
                int(x) for x in scatter_to_striped_out[:8].cpu().tolist()
            ],
            "scatter_flagged": [int(x) for x in scatter_flagged_out[:8].cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the Prims vector block-exchange example and validate it."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
