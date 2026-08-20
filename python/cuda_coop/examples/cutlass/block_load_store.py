#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Copy one partial block tile with the CUTLASS backend."""

from __future__ import annotations

import functools
from typing import Any

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
INPUT_OFFSET = 3
OUTPUT_OFFSET = 5
LOAD_VALID_ITEMS = TILE_ITEMS - 7
STORE_VALID_ITEMS = TILE_ITEMS - 3
OOB_DEFAULT = -1
SENTINEL = -999


@functools.lru_cache(maxsize=2)
def make_runner(import_form: str = "root") -> tuple[Any, Any, Any, Any]:
    """Build the one-block CuTe JIT runner and return its runtime helpers."""

    import cutlass
    import torch
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    if import_form == "root":
        from cuda import coop
    elif import_form == "qualified":
        import cuda.coop.cutlass as coop
    else:
        raise ValueError("import_form must be 'root' or 'qualified'")

    globals()["cute"] = cute

    # example-begin block-load-store
    @cute.kernel
    def _partial_copy(source: cute.Tensor, destination: cute.Tensor):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS_PER_THREAD)
        coop.load(
            block,
            source,
            items,
            valid_items=LOAD_VALID_ITEMS,
            oob_default=OOB_DEFAULT,
            offset=INPUT_OFFSET,
        )
        coop.store(
            block,
            destination,
            items,
            valid_items=STORE_VALID_ITEMS,
            offset=OUTPUT_OFFSET,
        )

    # example-end block-load-store

    @cute.jit
    def _run(source: cute.Tensor, destination: cute.Tensor):
        _partial_copy(source, destination).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
        )

    return _run, torch, from_dlpack, cutlass


def run_example(import_form: str = "root") -> list[int]:
    """Run and validate the partial copy, returning the destination values."""

    run, torch, from_dlpack, cutlass = make_runner(import_form)
    cutlass.cuda.initialize_cuda_context()

    source_host = torch.arange(
        INPUT_OFFSET + LOAD_VALID_ITEMS,
        dtype=torch.int32,
    )
    source = source_host.cuda()
    destination = torch.full(
        (OUTPUT_OFFSET + TILE_ITEMS + 4,),
        SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )
    run(from_dlpack(source), from_dlpack(destination))
    torch.cuda.synchronize()

    expected = torch.full_like(destination, SENTINEL, device="cpu")
    expected[OUTPUT_OFFSET : OUTPUT_OFFSET + LOAD_VALID_ITEMS] = source_host[
        INPUT_OFFSET : INPUT_OFFSET + LOAD_VALID_ITEMS
    ]
    expected[OUTPUT_OFFSET + LOAD_VALID_ITEMS : OUTPUT_OFFSET + STORE_VALID_ITEMS] = (
        OOB_DEFAULT
    )
    torch.testing.assert_close(destination.cpu(), expected, atol=0, rtol=0)
    return destination.cpu().tolist()


def main() -> int:
    """Run the example and report its validated output length."""

    values = run_example()
    print(f"validated {len(values)} output values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
