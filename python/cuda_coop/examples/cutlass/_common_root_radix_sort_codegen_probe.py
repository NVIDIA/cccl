# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for common keys-only Radix Sort."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
OUTPUT_SEGMENTS = 8
BEGIN_ONLY = 8
EXPLICIT_BEGIN = 4
EXPLICIT_END = 12
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

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_storage = common_coop.TempStorage()
        qualified_storage = cutlass_coop.TempStorage()
        common_full = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            temp_storage=common_storage,
        )
        qualified_full = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            temp_storage=qualified_storage,
        )
        common_begin_only = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=BEGIN_ONLY,
            descending=True,
        )
        qualified_begin_only = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=BEGIN_ONLY,
            descending=True,
        )
        common_explicit = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=EXPLICIT_BEGIN,
            end_bit=EXPLICIT_END,
            temp_storage=common_storage,
        )
        qualified_explicit = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=EXPLICIT_BEGIN,
            end_bit=EXPLICIT_END,
            temp_storage=qualified_storage,
        )

        _store(output, 0, rank, common_keys)
        _store(output, 1, rank, qualified_keys)
        _store(output, 2, rank, common_full)
        _store(output, 3, rank, qualified_full)
        _store(output, 4, rank, common_begin_only)
        _store(output, 5, rank, qualified_begin_only)
        _store(output, 6, rank, common_explicit)
        _store(output, 7, rank, qualified_explicit)

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
    bits = 64 if dtype_name.endswith("64") else 32
    explicit_begin = 36 if bits == 64 else EXPLICIT_BEGIN
    explicit_end = 52 if bits == 64 else EXPLICIT_END
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

        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_storage = common_coop.TempStorage()
        qualified_storage = cutlass_coop.TempStorage()
        common_full = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            temp_storage=common_storage,
        )
        qualified_full = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            temp_storage=qualified_storage,
        )
        common_begin_only = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=BEGIN_ONLY,
            descending=True,
        )
        qualified_begin_only = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=BEGIN_ONLY,
            descending=True,
        )
        common_explicit = common_coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=explicit_begin,
            end_bit=explicit_end,
            temp_storage=common_storage,
        )
        qualified_explicit = cutlass_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=explicit_begin,
            end_bit=explicit_end,
            temp_storage=qualified_storage,
        )

        _store(output, 0, rank, common_keys)
        _store(output, 1, rank, qualified_keys)
        _store(output, 2, rank, common_full)
        _store(output, 3, rank, qualified_full)
        _store(output, 4, rank, common_begin_only)
        _store(output, 5, rank, qualified_begin_only)
        _store(output, 6, rank, common_explicit)
        _store(output, 7, rank, qualified_explicit)

    @cute.jit
    def _run(keys_in: cute.Tensor, output: cute.Tensor):
        _kernel(keys_in, output).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return _run, torch, from_dlpack, cutlass, torch_dtype


def _expected_radix_order(
    keys,
    *,
    bits: int,
    signed: bool,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    torch,
):
    value_mask = (1 << bits) - 1
    sign_mask = 1 << (bits - 1) if signed else 0
    mask = (1 << (end_bit - begin_bit)) - 1
    python_keys = keys.tolist()

    def digit(index: int) -> int:
        raw = int(python_keys[index]) & value_mask
        ordered = raw ^ sign_mask
        return (ordered >> begin_bit) & mask

    order = sorted(
        range(len(keys)),
        key=lambda index: ((-digit(index) if descending else digit(index)), index),
    )
    # Reorder through NumPy: released Torch builds do not implement CPU
    # advanced indexing for uint32/uint64 tensors.
    return torch.from_numpy(keys.numpy()[order])


def _portable_dtype_keys(dtype_name: str, *, torch):
    bits = 64 if dtype_name.endswith("64") else 32
    signed = not dtype_name.startswith("u")
    value_mask = (1 << bits) - 1
    multiplier = 0x9E37_79B1 if bits == 32 else 0x9E37_79B9_7F4A_7C15
    raw_values: list[int] = []
    values: list[int] = []
    for index in range(TOTAL_ITEMS):
        raw = (
            index * multiplier
            ^ (index % 13) << (bits // 2)
            ^ (index * 29 + index % 7 * 11)
        ) & value_mask
        if index and index % 17 == 0:
            raw = raw_values[-1]
        raw_values.append(raw)
        if signed and raw >= 1 << (bits - 1):
            raw -= 1 << bits
        values.append(raw)
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def run_dtype_example(dtype_name: str) -> dict[str, Any]:
    import numpy as np

    run, torch, from_dlpack, cutlass, torch_dtype = make_dtype_runner(dtype_name)
    cutlass.cuda.initialize_cuda_context()

    bits = 64 if dtype_name.endswith("64") else 32
    signed = not dtype_name.startswith("u")
    explicit_begin = 36 if bits == 64 else EXPLICIT_BEGIN
    explicit_end = 52 if bits == 64 else EXPLICIT_END
    keys_host = _portable_dtype_keys(dtype_name, torch=torch)
    python_values = [int(value) for value in keys_host.tolist()]
    high_bit = 1 << (bits - 1)
    if signed:
        assert any(value < 0 for value in python_values)
        assert any(value >= 0 for value in python_values)
    else:
        assert any(value < high_bit for value in python_values)
        assert any(value >= high_bit for value in python_values)
    if bits == 64:
        assert max(abs(value) for value in python_values) > 1 << 32

    keys = keys_host.cuda()
    output = torch.zeros(
        (OUTPUT_SEGMENTS * TOTAL_ITEMS,),
        dtype=torch_dtype,
        device="cuda",
    )
    run(from_dlpack(keys), from_dlpack(output))
    torch.cuda.synchronize()

    observed = output.cpu().reshape(OUTPUT_SEGMENTS, TOTAL_ITEMS)
    assert observed.dtype == keys_host.dtype
    observed_numpy = observed.numpy()
    keys_numpy = keys_host.numpy()
    np.testing.assert_array_equal(observed_numpy[0], keys_numpy)
    np.testing.assert_array_equal(observed_numpy[1], keys_numpy)
    for common_index, qualified_index, begin_bit, end_bit, descending in (
        (2, 3, 0, bits, False),
        (4, 5, BEGIN_ONLY, bits, True),
        (6, 7, explicit_begin, explicit_end, False),
    ):
        np.testing.assert_array_equal(
            observed_numpy[common_index],
            observed_numpy[qualified_index],
        )
        expected = _expected_radix_order(
            keys_host,
            bits=bits,
            signed=signed,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            torch=torch,
        )
        np.testing.assert_array_equal(
            observed_numpy[common_index],
            expected.numpy(),
        )

    return {
        "begin_only_defaults_to_width": True,
        "bit_width": bits,
        "block_threads": BLOCK_THREADS,
        "dtype": dtype_name,
        "explicit_subrange": (explicit_begin, explicit_end),
        "high_bit_or_wide_values": True,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
    }


def run_example() -> dict[str, Any]:
    run, torch, from_dlpack, cutlass = make_runner()
    cutlass.cuda.initialize_cuda_context()

    indices = torch.arange(TOTAL_ITEMS, dtype=torch.int32)
    keys_host = ((indices * 37 + (indices % 7) * 11) % 53) - 26
    keys_host[::9] = keys_host[1::9][: len(keys_host[::9])]
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
    for common_index, qualified_index, begin_bit, end_bit, descending in (
        (2, 3, 0, 32, False),
        (4, 5, BEGIN_ONLY, 32, True),
        (6, 7, EXPLICIT_BEGIN, EXPLICIT_END, False),
    ):
        torch.testing.assert_close(
            observed[common_index],
            observed[qualified_index],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            observed[common_index],
            _expected_radix_order(
                keys_host,
                bits=32,
                signed=True,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=descending,
                torch=torch,
            ),
            atol=0,
            rtol=0,
        )

    return {
        "begin_only_defaults_to_width": True,
        "block_threads": BLOCK_THREADS,
        "duplicate_keys": True,
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "signed_bit_order": True,
    }
