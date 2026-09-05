# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CUTLASS histogram/run-length example using ``ThreadData``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
HISTOGRAM_BINS = BLOCK_THREADS * ITEMS_PER_THREAD
HISTOGRAM_BINS_PER_THREAD = ITEMS_PER_THREAD
DECODED_ITEMS_PER_THREAD = 2
DECODED_WINDOW_OFFSET = 3


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, coop, Int32 = require_runtime(include_int32=True)
    from cutlass.base_dsl.typing import Uint32

    return cutlass, cute, torch, from_dlpack, coop, Int32, Uint32


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Build and return the Prims histogram/RLE runner plus helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32, Uint32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    globals()["Uint32"] = Uint32
    run_length_temp_storage = coop.TempStorage(
        size_in_bytes=8192,
        sharing="shared",
    )

    @cute.kernel
    def _vector_histogram_run_length_kernel(
        samples_in: cute.Tensor,
        run_values_in: cute.Tensor,
        run_lengths_in: cute.Tensor,
        histogram_out: cute.Tensor,
        decoded_out: cute.Tensor,
        offsets_out: cute.Tensor,
        total_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        samples_vec = coop.ThreadData(items_per_thread, Int32)
        run_values_vec = coop.ThreadData(items_per_thread, Int32)
        run_lengths_vec = coop.ThreadData(items_per_thread, Uint32)
        coop.load(
            block,
            samples_in,
            samples_vec,
        )
        coop.load(
            block,
            run_values_in,
            run_values_vec,
        )
        coop.load(
            block,
            run_lengths_in,
            run_lengths_vec,
        )

        histogram_counts = coop.histogram(
            block,
            samples_vec,
            bins=HISTOGRAM_BINS,
            bins_per_thread=HISTOGRAM_BINS_PER_THREAD,
            counter_dtype=Int32,
            algorithm="sort",
        )
        coop.store(
            block,
            histogram_out,
            histogram_counts,
            algorithm="striped",
        )

        decoded_items_per_thread = cutlass.const_expr(DECODED_ITEMS_PER_THREAD)
        relative_offsets = coop.ThreadData(
            decoded_items_per_thread,
            dtype=Uint32,
        )
        total_decoded_size = coop.ThreadData(1, dtype=Uint32)
        decoded = coop.run_length_decode(
            block,
            run_values_vec,
            run_lengths_vec,
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_window_offset=cutlass.const_expr(DECODED_WINDOW_OFFSET),
            relative_offsets=relative_offsets,
            total_decoded_size=total_decoded_size,
        )
        coop.store(block, decoded_out, decoded)
        coop.store(block, offsets_out, relative_offsets)
        coop.store(block, total_out, total_decoded_size)

    @cute.jit
    def _run_vector_histogram_run_length(
        samples_in: cute.Tensor,
        run_values_in: cute.Tensor,
        run_lengths_in: cute.Tensor,
        histogram_out: cute.Tensor,
        decoded_out: cute.Tensor,
        offsets_out: cute.Tensor,
        total_out: cute.Tensor,
    ):
        _vector_histogram_run_length_kernel(
            samples_in,
            run_values_in,
            run_lengths_in,
            histogram_out,
            decoded_out,
            offsets_out,
            total_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_vector_histogram_run_length,
        torch,
        from_dlpack,
        cutlass,
        run_length_temp_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _expected_run_length_window(
    run_values: Any,
    run_lengths: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, int]:
    decoded_values = []
    relative_offsets = []
    for value, length in zip(run_values.tolist(), run_lengths.tolist(), strict=True):
        for offset in range(int(length)):
            decoded_values.append(int(value))
            relative_offsets.append(offset)

    window_size = BLOCK_THREADS * DECODED_ITEMS_PER_THREAD
    window = slice(
        DECODED_WINDOW_OFFSET,
        DECODED_WINDOW_OFFSET + window_size,
    )
    return (
        torch.tensor(decoded_values[window], dtype=run_values.dtype),
        torch.tensor(relative_offsets[window], dtype=torch.uint32),
        int(sum(int(length) for length in run_lengths.tolist())),
    )


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_vector_histogram_run_length,
        torch,
        from_dlpack,
        cutlass,
        run_length_temp_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    run_length_temp_storage.reset_uses()

    samples_host = torch.tensor(
        [((idx * 7 + idx // 3) % HISTOGRAM_BINS) for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    run_values_host = (torch.arange(TOTAL_ITEMS, dtype=torch.int64) + 200).to(
        torch.int32
    )
    run_lengths_host = ((torch.arange(TOTAL_ITEMS, dtype=torch.int64) % 4) + 1).to(
        torch.uint32
    )

    samples_in = samples_host.cuda()
    run_values_in = run_values_host.cuda()
    run_lengths_in = run_lengths_host.cuda()
    histogram_out = torch.zeros((HISTOGRAM_BINS,), dtype=torch.int32, device="cuda")
    decoded_out = torch.zeros(
        (BLOCK_THREADS * DECODED_ITEMS_PER_THREAD,),
        dtype=torch.int32,
        device="cuda",
    )
    offsets_out = torch.zeros(
        (BLOCK_THREADS * DECODED_ITEMS_PER_THREAD,),
        dtype=torch.uint32,
        device="cuda",
    )
    total_out = torch.zeros((BLOCK_THREADS,), dtype=torch.uint32, device="cuda")

    samples_arg = from_dlpack(samples_in)
    run_values_arg = from_dlpack(run_values_in)
    run_lengths_arg = from_dlpack(run_lengths_in)
    histogram_arg = from_dlpack(histogram_out)
    decoded_arg = from_dlpack(decoded_out)
    offsets_arg = from_dlpack(offsets_out)
    total_arg = from_dlpack(total_out)

    def step() -> None:
        run_vector_histogram_run_length(
            samples_arg,
            run_values_arg,
            run_lengths_arg,
            histogram_arg,
            decoded_arg,
            offsets_arg,
            total_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_histogram = torch.bincount(
            samples_host,
            minlength=HISTOGRAM_BINS,
        ).to(torch.int32)
        expected_decoded, expected_offsets, decoded_total = _expected_run_length_window(
            run_values_host,
            run_lengths_host,
            torch=torch,
        )
        expected_total = torch.full(
            (BLOCK_THREADS,),
            decoded_total,
            dtype=torch.uint32,
        )

        torch.testing.assert_close(
            histogram_out.cpu(),
            expected_histogram,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(decoded_out.cpu(), expected_decoded, atol=0, rtol=0)
        torch.testing.assert_close(offsets_out.cpu(), expected_offsets, atol=0, rtol=0)
        torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)

        return {
            "histogram": [int(x) for x in histogram_out[:8].cpu().tolist()],
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
    """Run the Prims vector histogram/run-length example and validate it."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
