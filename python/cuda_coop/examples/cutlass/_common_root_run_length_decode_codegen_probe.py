# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private runtime and final-cubin probe for common Run-Length Decode."""

from __future__ import annotations

import functools
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_DIM = (8, 4, 2)
BLOCK_THREADS = 64
RUNS_PER_THREAD = 2
DECODED_ITEMS_PER_THREAD = 3
TOTAL_RUNS = BLOCK_THREADS * RUNS_PER_THREAD
TOTAL_OUTPUT_ITEMS = BLOCK_THREADS * DECODED_ITEMS_PER_THREAD
DECODED_OUTPUT_SEGMENTS = 6
WINDOW_OFFSET_COUNT = 3
PORTABLE_DTYPE_CASES = (
    ("uint8", "int32"),
    ("int32", "uint32"),
    ("uint32", "int64"),
    ("int64", "uint64"),
    ("uint64", "uint64"),
)
_PORTABLE_DTYPE_NAMES = {
    "uint8": "Uint8",
    "int32": "Int32",
    "uint32": "Uint32",
    "int64": "Int64",
    "uint64": "Uint64",
}


def _store_runs(output, segment: int, rank, values) -> None:
    offset = segment * TOTAL_RUNS + rank * RUNS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]


def _store_decoded(output, segment: int, rank, values) -> None:
    offset = segment * TOTAL_OUTPUT_ITEMS + rank * DECODED_ITEMS_PER_THREAD
    output[offset] = values[0]
    output[offset + 1] = values[1]
    output[offset + 2] = values[2]


def _store_total(output, rank, value) -> None:
    output[rank] = value[0]


def _compiler_dtype(dtype_name: str, *, cutlass_typing) -> type:
    try:
        return getattr(cutlass_typing, _PORTABLE_DTYPE_NAMES[dtype_name])
    except KeyError as exc:
        supported = ", ".join(sorted(_PORTABLE_DTYPE_NAMES))
        raise ValueError(
            f"dtype name must be one of {{{supported}}}; got {dtype_name!r}"
        ) from exc


@functools.lru_cache(maxsize=len(PORTABLE_DTYPE_CASES))
def make_dtype_runner(
    value_dtype_name: str,
    length_dtype_name: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one portable value/run-length differential runner."""

    if (value_dtype_name, length_dtype_name) not in PORTABLE_DTYPE_CASES:
        supported = ", ".join(
            f"{value}/{length}" for value, length in PORTABLE_DTYPE_CASES
        )
        raise ValueError(
            "value/length dtype pair must be one of "
            f"{{{supported}}}; got {value_dtype_name}/{length_dtype_name}"
        )

    import numpy as np
    from cutlass.base_dsl import typing as cutlass_typing

    cutlass, cute, torch, from_dlpack, cutlass_coop, Int32 = require_runtime(
        include_int32=True
    )
    from cuda import coop as common_coop

    value_type = _compiler_dtype(value_dtype_name, cutlass_typing=cutlass_typing)
    length_type = _compiler_dtype(length_dtype_name, cutlass_typing=cutlass_typing)
    ordinary_value_dtype = getattr(np, value_dtype_name)
    ordinary_length_dtype = getattr(np, length_dtype_name)
    value_torch_dtype = getattr(torch, value_dtype_name)
    length_torch_dtype = getattr(torch, length_dtype_name)
    globals()["cute"] = cute
    globals()["common_coop"] = common_coop
    globals()["cutlass_coop"] = cutlass_coop
    globals()["Int32"] = Int32

    @cute.kernel
    def _kernel(
        run_values_in: cute.Tensor,
        run_lengths_in: cute.Tensor,
        decoded_window_offsets: cute.Tensor,
        preserved_values: cute.Tensor,
        preserved_lengths: cute.Tensor,
        decoded_output: cute.Tensor,
        relative_output: cute.Tensor,
        total_output: cute.Tensor,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        rank = tidx + Int32(8) * (tidy + Int32(4) * tidz)
        run_offset = rank * Int32(RUNS_PER_THREAD)

        common_values = common_coop.ThreadData(
            RUNS_PER_THREAD,
            dtype=ordinary_value_dtype,
        )
        common_lengths = common_coop.ThreadData(
            RUNS_PER_THREAD,
            dtype=ordinary_length_dtype,
        )
        qualified_values = cutlass_coop.ThreadData(
            RUNS_PER_THREAD,
            dtype=value_type,
        )
        qualified_lengths = cutlass_coop.ThreadData(
            RUNS_PER_THREAD,
            dtype=length_type,
        )
        maximum_relative_offsets = cutlass_coop.ThreadData(
            DECODED_ITEMS_PER_THREAD,
            dtype=length_type,
        )
        maximum_total_decoded_size = cutlass_coop.ThreadData(
            1,
            dtype=length_type,
        )
        common_values[0] = run_values_in[run_offset]
        common_values[1] = run_values_in[run_offset + Int32(1)]
        common_lengths[0] = run_lengths_in[run_offset]
        common_lengths[1] = run_lengths_in[run_offset + Int32(1)]
        qualified_values[0] = run_values_in[run_offset]
        qualified_values[1] = run_values_in[run_offset + Int32(1)]
        qualified_lengths[0] = run_lengths_in[run_offset]
        qualified_lengths[1] = run_lengths_in[run_offset + Int32(1)]

        partial_offset = decoded_window_offsets[0]
        after_total_offset = decoded_window_offsets[Int32(1)]
        maximum_offset = decoded_window_offsets[Int32(2)]
        common_group = common_coop.this_block()
        qualified_group = cutlass_coop.this_block()
        common_partial = common_coop.run_length_decode(
            common_group,
            common_values,
            common_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=partial_offset,
        )
        qualified_partial = cutlass_coop.run_length_decode(
            qualified_group,
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=partial_offset,
        )
        common_after_total = common_coop.run_length_decode(
            common_group,
            common_values,
            common_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=after_total_offset,
        )
        qualified_after_total = cutlass_coop.run_length_decode(
            qualified_group,
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=after_total_offset,
        )
        common_maximum = common_coop.run_length_decode(
            common_group,
            common_values,
            common_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=maximum_offset,
        )
        qualified_maximum = cutlass_coop.run_length_decode(
            qualified_group,
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=DECODED_ITEMS_PER_THREAD,
            decoded_window_offset=maximum_offset,
            relative_offsets=maximum_relative_offsets,
            total_decoded_size=maximum_total_decoded_size,
        )

        _store_runs(preserved_values, 0, rank, common_values)
        _store_runs(preserved_values, 1, rank, qualified_values)
        _store_runs(preserved_lengths, 0, rank, common_lengths)
        _store_runs(preserved_lengths, 1, rank, qualified_lengths)
        _store_decoded(decoded_output, 0, rank, common_partial)
        _store_decoded(decoded_output, 1, rank, qualified_partial)
        _store_decoded(decoded_output, 2, rank, common_after_total)
        _store_decoded(decoded_output, 3, rank, qualified_after_total)
        _store_decoded(decoded_output, 4, rank, common_maximum)
        _store_decoded(decoded_output, 5, rank, qualified_maximum)
        _store_decoded(relative_output, 0, rank, maximum_relative_offsets)
        _store_total(total_output, rank, maximum_total_decoded_size)

    @cute.jit
    def _run(
        run_values_in: cute.Tensor,
        run_lengths_in: cute.Tensor,
        decoded_window_offsets: cute.Tensor,
        preserved_values: cute.Tensor,
        preserved_lengths: cute.Tensor,
        decoded_output: cute.Tensor,
        relative_output: cute.Tensor,
        total_output: cute.Tensor,
    ):
        _kernel(
            run_values_in,
            run_lengths_in,
            decoded_window_offsets,
            preserved_values,
            preserved_lengths,
            decoded_output,
            relative_output,
            total_output,
        ).launch(grid=(1, 1, 1), block=BLOCK_DIM)

    return (
        _run,
        torch,
        from_dlpack,
        cutlass,
        value_torch_dtype,
        length_torch_dtype,
    )


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Return the canonical UInt64/UInt64 final-link runner."""

    return make_dtype_runner("uint64", "uint64")


def _host_values(dtype_name: str) -> list[int]:
    bits = 8 if dtype_name == "uint8" else (64 if dtype_name.endswith("64") else 32)
    signed = not dtype_name.startswith("u")
    values: list[int] = []
    for index in range(TOTAL_RUNS):
        if bits == 8:
            value = (index * 17 + 3) % 251 + 1
        elif signed:
            magnitude = index * 101 + 7
            if bits == 64:
                magnitude += (1 << 40) + (index % 7) * (1 << 34)
            value = -magnitude if index % 2 else magnitude
        else:
            value = index * 193 + 11
            if index % 2:
                value += 1 << (bits - 1)
            if bits == 64:
                value += (index % 5) << 40
        assert value != 0
        values.append(value)
    return values


def _host_lengths(length_dtype_name: str) -> tuple[list[int], int, int]:
    if length_dtype_name in {"int64", "uint64"}:
        lengths = [0] * TOTAL_RUNS
        total_max = _maximum_representable_offset(length_dtype_name)
        lengths[0] = total_max - 7
        lengths[1] = 4
        lengths[2] = 3
        total = sum(lengths)
        assert total == total_max
        partial_offset = total - 5
        after_total_offset = total
    else:
        active_runs = 96
        lengths = [1 + (index * 7) % 4 for index in range(active_runs)]
        lengths.extend([0] * (TOTAL_RUNS - active_runs))
        total = sum(lengths)
        partial_offset = total - 17
        after_total_offset = total + 11
    first_padding_run = lengths.index(0)
    assert all(length > 0 for length in lengths[:first_padding_run])
    assert all(length == 0 for length in lengths[first_padding_run:])
    assert 0 < total - partial_offset < TOTAL_OUTPUT_ITEMS
    return lengths, partial_offset, after_total_offset


def _decoded_window(
    values: list[int],
    lengths: list[int],
    *,
    offset: int,
) -> list[int]:
    total = sum(lengths)
    run_index = 0
    run_begin = 0
    while run_index < len(lengths) and offset >= run_begin + lengths[run_index]:
        run_begin += lengths[run_index]
        run_index += 1

    result: list[int] = []
    for target in range(offset, offset + TOTAL_OUTPUT_ITEMS):
        while run_index < len(lengths) and target >= run_begin + lengths[run_index]:
            run_begin += lengths[run_index]
            run_index += 1
        result.append(values[run_index] if target < total else 0)
    return result


def _maximum_representable_offset(dtype_name: str) -> int:
    bits = 64 if dtype_name.endswith("64") else 32
    value_bits = bits - 1 if not dtype_name.startswith("u") else bits
    return (1 << value_bits) - 1


def run_dtype_example(
    value_dtype_name: str,
    length_dtype_name: str,
) -> dict[str, Any]:
    """Run one common/qualified parity case twice and validate exact results."""

    import numpy as np

    run, torch, from_dlpack, cutlass, value_torch_dtype, length_torch_dtype = (
        make_dtype_runner(value_dtype_name, length_dtype_name)
    )
    cutlass.cuda.initialize_cuda_context()

    values = _host_values(value_dtype_name)
    lengths, partial_offset, after_total_offset = _host_lengths(length_dtype_name)
    maximum_offset = _maximum_representable_offset(length_dtype_name)
    decoded_total = sum(lengths)
    partial_valid_items = decoded_total - partial_offset
    expected_partial = np.asarray(
        _decoded_window(values, lengths, offset=partial_offset),
        dtype=getattr(np, value_dtype_name),
    )
    expected_after_total = np.zeros(
        (TOTAL_OUTPUT_ITEMS,),
        dtype=getattr(np, value_dtype_name),
    )
    values_host = torch.tensor(values, dtype=value_torch_dtype)
    lengths_host = torch.tensor(lengths, dtype=length_torch_dtype)
    offsets_host = torch.tensor(
        [partial_offset, after_total_offset, maximum_offset],
        dtype=length_torch_dtype,
    )
    values_in = values_host.cuda()
    lengths_in = lengths_host.cuda()
    offsets_in = offsets_host.cuda()
    preserved_values = torch.zeros(
        (2 * TOTAL_RUNS,),
        dtype=value_torch_dtype,
        device="cuda",
    )
    preserved_lengths = torch.zeros(
        (2 * TOTAL_RUNS,),
        dtype=length_torch_dtype,
        device="cuda",
    )
    decoded_output = torch.ones(
        (DECODED_OUTPUT_SEGMENTS * TOTAL_OUTPUT_ITEMS,),
        dtype=value_torch_dtype,
        device="cuda",
    )
    relative_output = torch.zeros(
        (TOTAL_OUTPUT_ITEMS,),
        dtype=length_torch_dtype,
        device="cuda",
    )
    total_output = torch.zeros(
        (BLOCK_THREADS,),
        dtype=length_torch_dtype,
        device="cuda",
    )
    values_arg = from_dlpack(values_in)
    lengths_arg = from_dlpack(lengths_in)
    offsets_arg = from_dlpack(offsets_in)
    preserved_values_arg = from_dlpack(preserved_values)
    preserved_lengths_arg = from_dlpack(preserved_lengths)
    decoded_output_arg = from_dlpack(decoded_output)
    relative_output_arg = from_dlpack(relative_output)
    total_output_arg = from_dlpack(total_output)
    snapshots: list[np.ndarray] = []

    for _ in range(2):
        preserved_values.zero_()
        preserved_lengths.zero_()
        decoded_output.fill_(1)
        relative_output.zero_()
        total_output.zero_()
        run(
            values_arg,
            lengths_arg,
            offsets_arg,
            preserved_values_arg,
            preserved_lengths_arg,
            decoded_output_arg,
            relative_output_arg,
            total_output_arg,
        )
        torch.cuda.synchronize()

        observed_values = preserved_values.cpu().reshape(2, TOTAL_RUNS).numpy()
        observed_lengths = preserved_lengths.cpu().reshape(2, TOTAL_RUNS).numpy()
        expected_values = values_host.numpy()
        expected_lengths = lengths_host.numpy()
        np.testing.assert_array_equal(observed_values[0], expected_values)
        np.testing.assert_array_equal(observed_values[1], expected_values)
        np.testing.assert_array_equal(observed_lengths[0], expected_lengths)
        np.testing.assert_array_equal(observed_lengths[1], expected_lengths)

        observed_decoded = (
            decoded_output.cpu()
            .reshape(DECODED_OUTPUT_SEGMENTS, TOTAL_OUTPUT_ITEMS)
            .numpy()
        )
        np.testing.assert_array_equal(observed_decoded[0], expected_partial)
        np.testing.assert_array_equal(observed_decoded[1], expected_partial)
        np.testing.assert_array_equal(observed_decoded[2], expected_after_total)
        np.testing.assert_array_equal(observed_decoded[3], expected_after_total)
        np.testing.assert_array_equal(observed_decoded[4], expected_after_total)
        np.testing.assert_array_equal(observed_decoded[5], expected_after_total)
        relative_oob_sentinel = (
            -1
            if not length_dtype_name.startswith("u")
            else np.iinfo(getattr(np, length_dtype_name)).max
        )
        np.testing.assert_array_equal(
            relative_output.cpu().numpy(),
            np.full(
                (TOTAL_OUTPUT_ITEMS,),
                relative_oob_sentinel,
                dtype=getattr(np, length_dtype_name),
            ),
        )
        np.testing.assert_array_equal(
            total_output.cpu().numpy(),
            np.full(
                (BLOCK_THREADS,),
                decoded_total,
                dtype=getattr(np, length_dtype_name),
            ),
        )
        snapshots.append(observed_decoded.copy())

    np.testing.assert_array_equal(snapshots[0], snapshots[1])
    return {
        "after_total_zero_filled": True,
        "block_dim": BLOCK_DIM,
        "common_qualified_exact": True,
        "decoded_items_per_thread": DECODED_ITEMS_PER_THREAD,
        "decoded_total": decoded_total,
        "genuine_64bit_window": length_dtype_name.endswith("64"),
        "input_preserved": True,
        "length_dtype": length_dtype_name,
        "maximum_offset": maximum_offset,
        "maximum_offset_zero_filled": True,
        "multi_run": True,
        "partial_tail_zero_filled": True,
        "partial_valid_items": partial_valid_items,
        "relative_oob_sentinel": relative_oob_sentinel,
        "repeat_launches": 2,
        "runs_per_thread": RUNS_PER_THREAD,
        "value_dtype": value_dtype_name,
        "trailing_zero_length_padding": True,
    }


def run_example() -> dict[str, Any]:
    """Run the canonical wide-offset UInt64 final-link probe."""

    return run_dtype_example("uint64", "uint64")
