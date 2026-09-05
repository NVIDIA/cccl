# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block run-length decode window example for the cuda.coop.cutlass root."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
RUNS_PER_THREAD = 2
DECODED_ITEMS_PER_THREAD = 4
DECODED_WINDOW_OFFSET = 5


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop = require_runtime()
    from cutlass.base_dsl.typing import Uint32

    return cutlass, cute, torch, from_dlpack, coop, Uint32


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Uint32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Uint32"] = Uint32

    @cute.kernel
    def _run_length_decode_window_kernel(
        values_in: cute.Tensor,
        lengths_in: cute.Tensor,
        decoded_out: cute.Tensor,
        offsets_out: cute.Tensor,
        total_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        run_base = tidx * RUNS_PER_THREAD
        runs = coop.ThreadData.from_values(
            values_in[run_base + 0],
            values_in[run_base + 1],
            dtype=Uint32,
        )
        lengths = coop.ThreadData.from_values(
            lengths_in[run_base + 0],
            lengths_in[run_base + 1],
            dtype=Uint32,
        )
        decoded_items_per_thread = cutlass.const_expr(DECODED_ITEMS_PER_THREAD)
        relative_offsets = coop.ThreadData(
            decoded_items_per_thread,
            dtype=Uint32,
        )
        total_decoded_size = coop.ThreadData(1, dtype=Uint32)
        block = coop.this_block()
        decoded = coop.run_length_decode(
            block,
            runs,
            lengths,
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_window_offset=cutlass.const_expr(DECODED_WINDOW_OFFSET),
            relative_offsets=relative_offsets,
            total_decoded_size=total_decoded_size,
        )

        coop.store(
            block,
            decoded_out,
            decoded,
        )
        coop.store(
            block,
            offsets_out,
            relative_offsets,
        )
        coop.store(
            block,
            total_out,
            total_decoded_size,
        )

    @cute.jit
    def _run_run_length_decode_window(
        values_in: cute.Tensor,
        lengths_in: cute.Tensor,
        decoded_out: cute.Tensor,
        offsets_out: cute.Tensor,
        total_out: cute.Tensor,
    ):
        _run_length_decode_window_kernel(
            values_in,
            lengths_in,
            decoded_out,
            offsets_out,
            total_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_run_length_decode_window, torch, from_dlpack, cutlass


class PreparedExample:
    """Reusable launch, sync, and validation callbacks for the example."""

    __slots__ = ("step", "synchronize", "validate")

    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]

    def __init__(
        self,
        *,
        step: Callable[[], None],
        synchronize: Callable[[], None],
        validate: Callable[[], dict[str, Any]],
    ) -> None:
        self.step = step
        self.synchronize = synchronize
        self.validate = validate


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_run_length_decode_window, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    total_runs = BLOCK_THREADS * RUNS_PER_THREAD
    window_size = BLOCK_THREADS * DECODED_ITEMS_PER_THREAD
    host_values = (torch.arange(total_runs, dtype=torch.int64) + 100).to(torch.uint32)
    lengths_i64 = (torch.arange(total_runs, dtype=torch.int64) % 3) + 1
    required_decoded = DECODED_WINDOW_OFFSET + window_size
    decoded_total = int(lengths_i64.sum().item())
    if decoded_total < required_decoded:
        lengths_i64[-1] += required_decoded - decoded_total
        decoded_total = required_decoded
    host_lengths = lengths_i64.to(torch.uint32)

    values_in = host_values.cuda()
    lengths_in = host_lengths.cuda()
    decoded_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    offsets_out = torch.zeros((window_size,), dtype=torch.uint32, device="cuda")
    total_out = torch.zeros((BLOCK_THREADS,), dtype=torch.uint32, device="cuda")

    values_arg = from_dlpack(values_in)
    lengths_arg = from_dlpack(lengths_in)
    decoded_arg = from_dlpack(decoded_out)
    offsets_arg = from_dlpack(offsets_out)
    total_arg = from_dlpack(total_out)

    def step() -> None:
        run_run_length_decode_window(
            values_arg,
            lengths_arg,
            decoded_arg,
            offsets_arg,
            total_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        decoded_values = []
        relative_offsets = []
        for value, length in zip(host_values.tolist(), host_lengths.tolist()):
            for offset in range(length):
                decoded_values.append(value)
                relative_offsets.append(offset)
        window = slice(
            DECODED_WINDOW_OFFSET,
            DECODED_WINDOW_OFFSET + window_size,
        )
        expected_decoded = torch.tensor(decoded_values[window], dtype=torch.uint32)
        expected_offsets = torch.tensor(relative_offsets[window], dtype=torch.uint32)
        expected_total = torch.full(
            (BLOCK_THREADS,),
            decoded_total,
            dtype=torch.uint32,
        )

        torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
        torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
        torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)

        return {
            "decoded_window_offset": DECODED_WINDOW_OFFSET,
            "decoded": [int(x) for x in decoded_out[:8].cpu().tolist()],
            "relative_offsets": [int(x) for x in offsets_out[:8].cpu().tolist()],
            "total_decoded_size": int(total_out[0].cpu().item()),
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the run-length decode-window example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
