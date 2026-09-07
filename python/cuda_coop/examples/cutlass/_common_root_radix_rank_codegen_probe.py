# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for common Radix Rank."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
ITEMS_PER_THREAD = 2
TOTAL_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD
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
    bit_width = 64 if dtype_name.endswith("64") else 32
    begin_bit = bit_width - 8
    end_bit = bit_width
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(
        keys_in: cute.Tensor,
        preserved_keys: cute.Tensor,
        ranks: cute.Tensor,
    ):
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
        common_ascending = common_coop.radix_rank(
            common_group,
            common_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        qualified_ascending = cutlass_coop.radix_rank(
            qualified_group,
            qualified_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )
        common_descending = common_coop.radix_rank(
            common_group,
            common_keys,
            begin_bit=begin_bit,
            radix_bits=end_bit - begin_bit,
            descending=True,
        )
        qualified_descending = cutlass_coop.radix_rank(
            qualified_group,
            qualified_keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=True,
        )

        _store(preserved_keys, 0, rank, common_keys)
        _store(preserved_keys, 1, rank, qualified_keys)
        _store(ranks, 0, rank, common_ascending)
        _store(ranks, 1, rank, qualified_ascending)
        _store(ranks, 2, rank, common_descending)
        _store(ranks, 3, rank, qualified_descending)

    @cute.jit
    def _run(
        keys_in: cute.Tensor,
        preserved_keys: cute.Tensor,
        ranks: cute.Tensor,
    ):
        _kernel(keys_in, preserved_keys, ranks).launch(
            grid=(1, 1, 1),
            block=BLOCK_DIM,
        )

    return _run, torch, from_dlpack, cutlass, torch_dtype


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any]:
    """Return the canonical signed-int32 runner used by final-link tooling."""

    return make_dtype_runner("int32")


def _expected_ranks(
    keys,
    *,
    bit_width: int,
    signed: bool,
    begin_bit: int,
    end_bit: int,
    descending: bool,
    torch,
):
    digit_mask = (1 << (end_bit - begin_bit)) - 1
    value_mask = (1 << bit_width) - 1
    sign_mask = 1 << (bit_width - 1) if signed else 0
    python_keys = keys.tolist()

    def digit(index: int) -> int:
        raw = int(python_keys[index]) & value_mask
        ordered = raw ^ sign_mask
        return (ordered >> begin_bit) & digit_mask

    order = sorted(
        range(len(keys)),
        key=lambda index: (
            -digit(index) if descending else digit(index),
            index,
        ),
    )
    ranks = torch.empty((len(keys),), dtype=torch.int32)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _portable_dtype_keys(dtype_name: str, *, torch):
    bit_width = 64 if dtype_name.endswith("64") else 32
    signed = not dtype_name.startswith("u")
    begin_bit = bit_width - 8
    value_mask = (1 << bit_width) - 1
    low_mask = (1 << begin_bit) - 1
    values: list[int] = []
    for index in range(TOTAL_ITEMS):
        digit = (index * 29 + index // 3) & 0xFF
        low_bits = (index * 0x0012_345 + (index % 11) * 0x102_01) & low_mask
        raw = ((digit << begin_bit) | low_bits) & value_mask
        if signed and raw >= 1 << (bit_width - 1):
            raw -= 1 << bit_width
        values.append(raw)
    return torch.tensor(values, dtype=getattr(torch, dtype_name))


def run_dtype_example(dtype_name: str) -> dict[str, Any]:
    import numpy as np

    run, torch, from_dlpack, cutlass, torch_dtype = make_dtype_runner(dtype_name)
    cutlass.cuda.initialize_cuda_context()

    bit_width = 64 if dtype_name.endswith("64") else 32
    signed = not dtype_name.startswith("u")
    begin_bit = bit_width - 8
    end_bit = bit_width
    high_bit = 1 << (bit_width - 1)
    keys_host = _portable_dtype_keys(dtype_name, torch=torch)
    python_values = [int(value) for value in keys_host.tolist()]
    if signed:
        assert any(value < 0 for value in python_values)
        assert any(value >= 0 for value in python_values)
    else:
        assert any(value < high_bit for value in python_values)
        assert any(value >= high_bit for value in python_values)
    if bit_width == 64:
        assert max(abs(value) for value in python_values) > 1 << 32

    keys = keys_host.cuda()
    preserved_keys = torch.zeros(
        (2 * TOTAL_ITEMS,),
        dtype=torch_dtype,
        device="cuda",
    )
    ranks = torch.full(
        (4 * TOTAL_ITEMS,),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    run(
        from_dlpack(keys),
        from_dlpack(preserved_keys),
        from_dlpack(ranks),
    )
    torch.cuda.synchronize()

    observed_keys = preserved_keys.cpu().reshape(2, TOTAL_ITEMS).numpy()
    expected_keys = keys_host.numpy()
    np.testing.assert_array_equal(observed_keys[0], expected_keys)
    np.testing.assert_array_equal(observed_keys[1], expected_keys)

    observed_ranks = ranks.cpu().reshape(4, TOTAL_ITEMS)
    for common_index, qualified_index, descending in (
        (0, 1, False),
        (2, 3, True),
    ):
        torch.testing.assert_close(
            observed_ranks[common_index],
            observed_ranks[qualified_index],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            observed_ranks[common_index],
            _expected_ranks(
                keys_host,
                bit_width=bit_width,
                signed=signed,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=descending,
                torch=torch,
            ),
            atol=0,
            rtol=0,
        )

    return {
        "bit_width": bit_width,
        "block_threads": BLOCK_THREADS,
        "dtype": dtype_name,
        "high_or_sign_bit_interval": (begin_bit, end_bit),
        "input_preserved": True,
        "items_per_thread": ITEMS_PER_THREAD,
        "radix_bits_matches_end_bit": True,
        "stable_exact_ranks": True,
    }


def run_example() -> dict[str, Any]:
    """Run the canonical signed-int32 final-link probe."""

    return run_dtype_example("int32")
