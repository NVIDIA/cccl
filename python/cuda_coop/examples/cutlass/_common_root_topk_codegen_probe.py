# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for common keys-only TopK."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
OUTPUT_SEGMENTS = 6
TOPK_K = 11
VALID_ITEMS = TOTAL_ITEMS - 13
_PORTABLE_DTYPE_NAMES = {
    "int32": "Int32",
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
}


def _store(output, segment: int, rank, values) -> None:
    offset = segment * TOTAL_ITEMS + rank * ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


@functools.lru_cache(maxsize=4)
def make_dtype_runner(dtype_name: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one portable-integer common-versus-qualified TopK runner."""

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
    ordinary_dtype = int if dtype_name == "int32" else getattr(np, dtype_name)
    torch_dtype = getattr(torch, dtype_name)
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32
    topk_temp_storage = cutlass_coop.TempStorage(
        size_in_bytes=16_384,
        alignment=16,
        sharing="shared",
    )

    @cute.kernel
    def _kernel(
        keys_in: cute.Tensor,
        controls: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        offset = tidx * Int32(ITEMS_PER_THREAD)
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

        topk_k = controls[0]
        valid_items = controls[1]
        begin_bit = controls[2]
        end_bit = controls[3]
        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_max = common_coop.topk_max_keys(
            common_group,
            common_keys,
            topk_k,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )
        qualified_max = cutlass_coop.topk_max_keys(
            qualified_group,
            qualified_keys,
            topk_k,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )
        common_min = common_coop.topk_min_keys(
            common_group,
            common_keys,
            topk_k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )
        qualified_min = cutlass_coop.topk_min_keys(
            qualified_group,
            qualified_keys,
            topk_k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
            temp_storage=topk_temp_storage,
        )

        _store(output, 0, tidx, common_keys)
        _store(output, 1, tidx, qualified_keys)
        _store(output, 2, tidx, common_max)
        _store(output, 3, tidx, qualified_max)
        _store(output, 4, tidx, common_min)
        _store(output, 5, tidx, qualified_min)

    @cute.jit
    def _run(
        keys_in: cute.Tensor,
        controls: cute.Tensor,
        output: cute.Tensor,
    ):
        _kernel(keys_in, controls, output).launch(
            grid=(1, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
        )

    return _run, torch, from_dlpack, cutlass, torch_dtype, topk_temp_storage


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return the canonical signed-int32 runner used by final-link tooling."""

    return make_dtype_runner("int32")


def _portable_dtype_keys(dtype_name: str, *, torch):
    bit_width = 64 if dtype_name.endswith("64") else 32
    signed = not dtype_name.startswith("u")
    value_mask = (1 << bit_width) - 1
    multiplier = 0x9E37_79B1 if bit_width == 32 else 0x9E37_79B9_7F4A_7C15
    raw_values: list[int] = []
    values: list[int] = []
    for index in range(TOTAL_ITEMS):
        raw = (
            index * multiplier
            ^ (index % 13) << (bit_width // 2)
            ^ (index * 29 + index % 7 * 11)
        ) & value_mask
        if index and index % 17 == 0:
            raw = raw_values[-1]
        raw_values.append(raw)
        if signed and raw >= 1 << (bit_width - 1):
            raw -= 1 << bit_width
        values.append(raw)
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def _assert_unordered_selection(actual, expected, *, torch) -> None:
    torch.testing.assert_close(
        torch.sort(actual.cpu()).values,
        torch.sort(expected.cpu()).values,
        atol=0,
        rtol=0,
    )


def run_dtype_example(dtype_name: str) -> dict[str, Any]:
    """Run one portable integer dtype against an independent selection oracle."""

    run, torch, from_dlpack, cutlass, torch_dtype, topk_temp_storage = (
        make_dtype_runner(dtype_name)
    )
    cutlass.cuda.initialize_cuda_context()
    topk_temp_storage.reset_uses()

    bit_width = 64 if dtype_name.endswith("64") else 32
    keys_host = _portable_dtype_keys(dtype_name, torch=torch)
    keys = keys_host.cuda()
    controls = torch.tensor(
        [TOPK_K, VALID_ITEMS, 0, bit_width],
        dtype=torch.int32,
        device="cuda",
    )
    output = torch.zeros(
        (OUTPUT_SEGMENTS * TOTAL_ITEMS,),
        dtype=torch_dtype,
        device="cuda",
    )

    run(
        from_dlpack(keys),
        from_dlpack(controls),
        from_dlpack(output),
    )
    torch.cuda.synchronize()

    segments = output.view(OUTPUT_SEGMENTS, TOTAL_ITEMS).cpu()
    torch.testing.assert_close(segments[0], keys_host, atol=0, rtol=0)
    torch.testing.assert_close(segments[1], keys_host, atol=0, rtol=0)

    expected_max = torch.sort(keys_host, descending=True).values[:TOPK_K]
    expected_min = torch.sort(keys_host[:VALID_ITEMS]).values[:TOPK_K]
    _assert_unordered_selection(
        segments[2, :TOPK_K],
        expected_max,
        torch=torch,
    )
    _assert_unordered_selection(
        segments[3, :TOPK_K],
        expected_max,
        torch=torch,
    )
    _assert_unordered_selection(
        segments[4, :TOPK_K],
        expected_min,
        torch=torch,
    )
    _assert_unordered_selection(
        segments[5, :TOPK_K],
        expected_min,
        torch=torch,
    )
    _assert_unordered_selection(
        segments[2, :TOPK_K],
        segments[3, :TOPK_K],
        torch=torch,
    )
    _assert_unordered_selection(
        segments[4, :TOPK_K],
        segments[5, :TOPK_K],
        torch=torch,
    )

    python_values = [int(value) for value in keys_host.tolist()]
    high_bit = 1 << (bit_width - 1)
    if dtype_name.startswith("u"):
        high_bit_covered = any(value >= high_bit for value in python_values)
    else:
        high_bit_covered = any(value < 0 for value in python_values) and any(
            value >= 0 for value in python_values
        )
    assert high_bit_covered
    assert len(set(python_values)) < len(python_values)

    return {
        "block_threads": BLOCK_THREADS,
        "dtype": dtype_name,
        "duplicate_keys": True,
        "full_and_partial": True,
        "high_bit_values": True,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "runtime_controls": True,
    }


def run_example() -> dict[str, Any]:
    """Run the canonical signed-int32 final-link probe."""

    return run_dtype_example("int32")


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
