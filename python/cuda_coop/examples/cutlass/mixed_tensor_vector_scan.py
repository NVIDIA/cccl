# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first CuTe tensor scan example for cuda.coop.cutlass."""

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
def make_runner() -> tuple[Any, Any, Any, Any]:
    """Build and return the group-first scan JIT runner plus helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32

    # docs: start cutlass-load-scan-store
    @cute.kernel
    def _mixed_tensor_vector_scan_kernel(
        tensor_values_in: cute.Tensor,
        vector_values_in: cute.Tensor,
        tensor_prefix_out: cute.Tensor,
        vector_prefix_out: cute.Tensor,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        tensor_values = coop.ThreadData(
            items_per_thread=items_per_thread,
            dtype=Int32,
        )
        vector_values = coop.ThreadData(
            items_per_thread=items_per_thread,
            dtype=Int32,
        )
        coop.load(
            block,
            tensor_values_in,
            tensor_values,
        )
        coop.load(
            block,
            vector_values_in,
            vector_values,
        )

        tensor_prefix = coop.scan(
            block,
            tensor_values,
            mode="exclusive",
        )
        vector_prefix = coop.scan(
            block,
            vector_values,
            mode="exclusive",
        )

        coop.store(
            block,
            tensor_prefix_out,
            tensor_prefix,
        )
        coop.store(
            block,
            vector_prefix_out,
            vector_prefix,
        )

    # docs: end cutlass-load-scan-store

    @cute.jit
    def _run_mixed_tensor_vector_scan(
        tensor_values_in: cute.Tensor,
        vector_values_in: cute.Tensor,
        tensor_prefix_out: cute.Tensor,
        vector_prefix_out: cute.Tensor,
    ):
        _mixed_tensor_vector_scan_kernel(
            tensor_values_in,
            vector_values_in,
            tensor_prefix_out,
            vector_prefix_out,
            ITEMS_PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return _run_mixed_tensor_vector_scan, torch, from_dlpack, cutlass


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _exclusive_prefix(values: Any, *, torch: Any) -> Any:
    inclusive = torch.cumsum(values.to(torch.int64), dim=0)
    return (inclusive - values.to(torch.int64)).to(values.dtype)


def prepare_example() -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    run_mixed_tensor_vector_scan, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    tensor_values_host = torch.tensor(
        [((idx * 3 + 1) % 17) - 8 for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    vector_values_host = torch.tensor(
        [((idx * 5 + 7) % 23) - 11 for idx in range(TOTAL_ITEMS)],
        dtype=torch.int32,
    )
    tensor_values_in = tensor_values_host.cuda()
    vector_values_in = vector_values_host.cuda()
    tensor_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")
    vector_prefix_out = torch.zeros((TOTAL_ITEMS,), dtype=torch.int32, device="cuda")

    tensor_values_arg = from_dlpack(tensor_values_in)
    vector_values_arg = from_dlpack(vector_values_in)
    tensor_prefix_arg = from_dlpack(tensor_prefix_out)
    vector_prefix_arg = from_dlpack(vector_prefix_out)

    def step() -> None:
        run_mixed_tensor_vector_scan(
            tensor_values_arg,
            vector_values_arg,
            tensor_prefix_arg,
            vector_prefix_arg,
        )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        expected_tensor_prefix = _exclusive_prefix(tensor_values_host, torch=torch)
        expected_vector_prefix = _exclusive_prefix(vector_values_host, torch=torch)
        torch.testing.assert_close(
            tensor_prefix_out.cpu(),
            expected_tensor_prefix,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            vector_prefix_out.cpu(),
            expected_vector_prefix,
            atol=0,
            rtol=0,
        )
        return {
            "tensor_prefix": [int(x) for x in tensor_prefix_out.cpu().tolist()],
            "vector_prefix": [int(x) for x in vector_prefix_out.cpu().tolist()],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def run_example() -> dict[str, Any]:
    """Run the group-first scan example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
