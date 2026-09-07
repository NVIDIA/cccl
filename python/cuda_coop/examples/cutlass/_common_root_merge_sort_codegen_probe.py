# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for common keys-only MergeSort."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
WARP_ITEMS = 32 * ITEMS_PER_THREAD
BLOCK_VALID_ITEMS = 117
WARP_VALID_ITEMS = 53
BLOCK_DESCENDING_OOB_DEFAULT = -2_147_483_648
WARP_ASCENDING_OOB_DEFAULT = 2_147_483_647
OUTPUT_SEGMENTS = 10
DTYPE_OUTPUT_SEGMENTS = 6
_PORTABLE_DTYPE_NAMES = {
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
}


def _store(output, segment: int, rank, values) -> None:
    offset = segment * TOTAL_ITEMS + rank * ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any]:
    cutlass, cute, torch, from_dlpack, cutlass_coop, Int32 = require_runtime(
        include_int32=True
    )
    from cuda import coop as common_coop

    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(keys_in: cute.Tensor, output: cute.Tensor):
        tidx, tidy, tidz = cute.arch.thread_idx()
        rank = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        offset = rank * Int32(ITEMS_PER_THREAD)
        common_keys = common_coop.ThreadData(ITEMS_PER_THREAD, dtype=int)
        qualified_keys = cutlass_coop.ThreadData(ITEMS_PER_THREAD, dtype=Int32)
        common_keys[0] = keys_in[offset]
        common_keys[1] = keys_in[offset + Int32(1)]
        qualified_keys[0] = keys_in[offset]
        qualified_keys[1] = keys_in[offset + Int32(1)]

        common_block = common_coop.this_block()
        qualified_block = cutlass_coop.this_block()
        common_storage = common_coop.TempStorage()
        qualified_storage = cutlass_coop.TempStorage()
        common_block_full = common_coop.merge_sort_keys(
            common_block,
            common_keys,
            temp_storage=common_storage,
        )
        qualified_block_full = cutlass_coop.merge_sort_keys(
            qualified_block,
            qualified_keys,
            temp_storage=qualified_storage,
        )
        common_block_partial = common_coop.merge_sort_keys(
            common_block,
            common_keys,
            descending=True,
            valid_items=BLOCK_VALID_ITEMS,
            oob_default=BLOCK_DESCENDING_OOB_DEFAULT,
            temp_storage=common_storage,
        )
        qualified_block_partial = cutlass_coop.merge_sort_keys(
            qualified_block,
            qualified_keys,
            descending=True,
            valid_items=Int32(BLOCK_VALID_ITEMS),
            oob_default=Int32(BLOCK_DESCENDING_OOB_DEFAULT),
            temp_storage=qualified_storage,
        )

        common_warp = common_coop.this_warp()
        qualified_warp = cutlass_coop.this_warp()
        common_warp_full = common_coop.merge_sort_keys(
            common_warp,
            common_keys,
            descending=True,
        )
        qualified_warp_full = cutlass_coop.merge_sort_keys(
            qualified_warp,
            qualified_keys,
            descending=True,
        )
        common_warp_partial = common_coop.merge_sort_keys(
            common_warp,
            common_keys,
            valid_items=WARP_VALID_ITEMS,
            oob_default=WARP_ASCENDING_OOB_DEFAULT,
        )
        qualified_warp_partial = cutlass_coop.merge_sort_keys(
            qualified_warp,
            qualified_keys,
            valid_items=Int32(WARP_VALID_ITEMS),
            oob_default=Int32(WARP_ASCENDING_OOB_DEFAULT),
        )

        _store(output, 0, rank, common_keys)
        _store(output, 1, rank, qualified_keys)
        _store(output, 2, rank, common_block_full)
        _store(output, 3, rank, qualified_block_full)
        _store(output, 4, rank, common_block_partial)
        _store(output, 5, rank, qualified_block_partial)
        _store(output, 6, rank, common_warp_full)
        _store(output, 7, rank, qualified_warp_full)
        _store(output, 8, rank, common_warp_partial)
        _store(output, 9, rank, qualified_warp_partial)

    @cute.jit
    def _run(keys_in: cute.Tensor, output: cute.Tensor):
        _kernel(keys_in, output).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass


@functools.lru_cache(maxsize=3)
def make_dtype_runner(dtype_name: str) -> tuple[Any, Any, Any, Any, Any]:
    """Build one runtime-only portable-dtype differential runner."""

    try:
        compiler_dtype_name = _PORTABLE_DTYPE_NAMES[dtype_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PORTABLE_DTYPE_NAMES))
        raise ValueError(
            f"dtype_name must be one of {{{supported}}}; got {dtype_name!r}"
        ) from exc

    import numpy as np
    from cutlass.base_dsl import typing as cutlass_typing

    cutlass, cute, torch, from_dlpack, cutlass_coop, Int32 = require_runtime(
        include_int32=True
    )
    from cuda import coop as common_coop

    compiler_dtype = getattr(cutlass_typing, compiler_dtype_name)
    ordinary_dtype = getattr(np, dtype_name)
    torch_dtype = getattr(torch, dtype_name)
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(keys_in: cute.Tensor, output: cute.Tensor):
        tidx, tidy, tidz = cute.arch.thread_idx()
        rank = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        offset = rank * Int32(ITEMS_PER_THREAD)
        common_keys = common_coop.ThreadData(
            ITEMS_PER_THREAD,
            dtype=ordinary_dtype,
        )
        qualified_keys = cutlass_coop.ThreadData(
            ITEMS_PER_THREAD,
            dtype=compiler_dtype,
        )
        common_keys[0] = keys_in[offset]
        common_keys[1] = keys_in[offset + Int32(1)]
        qualified_keys[0] = keys_in[offset]
        qualified_keys[1] = keys_in[offset + Int32(1)]

        common_block_result = common_coop.merge_sort_keys(
            common_coop.this_block(),
            common_keys,
        )
        qualified_block_result = cutlass_coop.merge_sort_keys(
            cutlass_coop.this_block(),
            qualified_keys,
        )
        common_warp_result = common_coop.merge_sort_keys(
            common_coop.this_warp(),
            common_keys,
            descending=True,
        )
        qualified_warp_result = cutlass_coop.merge_sort_keys(
            cutlass_coop.this_warp(),
            qualified_keys,
            descending=True,
        )

        _store(output, 0, rank, common_keys)
        _store(output, 1, rank, qualified_keys)
        _store(output, 2, rank, common_block_result)
        _store(output, 3, rank, qualified_block_result)
        _store(output, 4, rank, common_warp_result)
        _store(output, 5, rank, qualified_warp_result)

    @cute.jit
    def _run(keys_in: cute.Tensor, output: cute.Tensor):
        _kernel(keys_in, output).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass, torch_dtype


def _expected_warp_sort(keys, *, descending: bool, torch):
    result = torch.empty_like(keys)
    for base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        result[base : base + WARP_ITEMS] = torch.sort(
            keys[base : base + WARP_ITEMS],
            descending=descending,
        ).values
    return result


def _assert_partial_warp_sort(actual, keys, *, torch) -> None:
    for base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        expected = torch.sort(keys[base : base + WARP_VALID_ITEMS]).values
        torch.testing.assert_close(
            actual[base : base + WARP_VALID_ITEMS],
            expected,
            atol=0,
            rtol=0,
        )


def _portable_dtype_keys(dtype_name: str, *, torch):
    bits = 64 if dtype_name.endswith("64") else 32
    unsigned = dtype_name.startswith("u")
    values: list[int] = []
    for index in range(TOTAL_ITEMS):
        magnitude = (index * 37 + (index % 7) * 11) % 53
        if unsigned:
            value = magnitude
            if index % 3 == 0:
                value += 1 << (bits - 1)
            elif bits == 64:
                value += (index % 5) << 36
        else:
            value = -magnitude if index % 2 else magnitude
            value += (1 << 35) if index % 3 == 0 else -(1 << 34)
        values.append(value)
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def run_dtype_example(dtype_name: str) -> dict[str, Any]:
    import numpy as np

    run, torch, from_dlpack, cutlass, torch_dtype = make_dtype_runner(dtype_name)
    cutlass.cuda.initialize_cuda_context()

    keys_host = _portable_dtype_keys(dtype_name, torch=torch)
    keys_numpy = keys_host.numpy()
    bits = 64 if dtype_name.endswith("64") else 32
    if dtype_name.startswith("u"):
        high_bit = 1 << (bits - 1)
        assert bool((keys_numpy < high_bit).any())
        assert bool((keys_numpy >= high_bit).any())
    else:
        assert int(np.abs(keys_numpy).max()) > 1 << 32

    keys = keys_host.cuda()
    output = torch.zeros(
        (DTYPE_OUTPUT_SEGMENTS * TOTAL_ITEMS,),
        dtype=torch_dtype,
        device="cuda",
    )
    run(from_dlpack(keys), from_dlpack(output))
    torch.cuda.synchronize()

    observed = output.cpu().reshape(DTYPE_OUTPUT_SEGMENTS, TOTAL_ITEMS)
    assert observed.dtype == keys_host.dtype
    observed_numpy = observed.numpy()
    np.testing.assert_array_equal(observed_numpy[0], keys_numpy)
    np.testing.assert_array_equal(observed_numpy[1], keys_numpy)
    np.testing.assert_array_equal(observed_numpy[2], observed_numpy[3])
    np.testing.assert_array_equal(observed_numpy[2], np.sort(keys_numpy))
    np.testing.assert_array_equal(observed_numpy[4], observed_numpy[5])
    expected_warp = np.empty_like(keys_numpy)
    for base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        expected_warp[base : base + WARP_ITEMS] = np.sort(
            keys_numpy[base : base + WARP_ITEMS]
        )[::-1]
    np.testing.assert_array_equal(observed_numpy[4], expected_warp)

    return {
        "dtype": dtype_name,
        "high_bit_or_wide_values": True,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "physical_warps": BLOCK_THREADS // 32,
    }


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    indices = torch.arange(TOTAL_ITEMS, dtype=torch.int32)
    keys_host = ((indices * 11 + indices // 3) % 19) - 9
    assert BLOCK_DESCENDING_OOB_DEFAULT < int(
        keys_host[:BLOCK_VALID_ITEMS].min().item()
    )
    assert all(
        WARP_ASCENDING_OOB_DEFAULT
        > int(keys_host[base : base + WARP_VALID_ITEMS].max().item())
        for base in range(0, TOTAL_ITEMS, WARP_ITEMS)
    )
    keys = keys_host.cuda()
    output = torch.full(
        (OUTPUT_SEGMENTS * TOTAL_ITEMS,),
        -777_777,
        dtype=torch.int32,
        device="cuda",
    )
    run(from_dlpack(keys), from_dlpack(output))
    torch.cuda.synchronize()

    observed = output.cpu().reshape(OUTPUT_SEGMENTS, TOTAL_ITEMS)
    torch.testing.assert_close(observed[0], keys_host, atol=0, rtol=0)
    torch.testing.assert_close(observed[1], keys_host, atol=0, rtol=0)
    torch.testing.assert_close(observed[2], observed[3], atol=0, rtol=0)
    torch.testing.assert_close(
        observed[2], torch.sort(keys_host).values, atol=0, rtol=0
    )
    torch.testing.assert_close(
        observed[4, :BLOCK_VALID_ITEMS],
        observed[5, :BLOCK_VALID_ITEMS],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        observed[4, :BLOCK_VALID_ITEMS],
        torch.sort(keys_host[:BLOCK_VALID_ITEMS], descending=True).values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(observed[6], observed[7], atol=0, rtol=0)
    torch.testing.assert_close(
        observed[6],
        _expected_warp_sort(keys_host, descending=True, torch=torch),
        atol=0,
        rtol=0,
    )
    _assert_partial_warp_sort(observed[8], keys_host, torch=torch)
    _assert_partial_warp_sort(observed[9], keys_host, torch=torch)
    for base in range(0, TOTAL_ITEMS, WARP_ITEMS):
        torch.testing.assert_close(
            observed[8, base : base + WARP_VALID_ITEMS],
            observed[9, base : base + WARP_VALID_ITEMS],
            atol=0,
            rtol=0,
        )

    return {
        "block_partial_items": BLOCK_VALID_ITEMS,
        "duplicate_keys": True,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "sentinel_ordering": True,
        "warp_partial_items": WARP_VALID_ITEMS,
    }


if __name__ == "__main__":
    print(run_example())
