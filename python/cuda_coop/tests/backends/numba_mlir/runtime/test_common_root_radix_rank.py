# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Differential GPU evidence for portable block Radix Rank."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _radix_rank32_kernel(
    source,
    common_original,
    qualified_original,
    common_ascending,
    qualified_ascending,
    common_descending,
    qualified_descending,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        value = source[index]
        common_keys[item] = value
        qualified_keys[item] = value

    common_group = coop.this_block()
    qualified_group = numba_coop.this_block()
    common_asc = coop.radix_rank(
        common_group,
        common_keys,
        begin_bit=28,
        end_bit=32,
    )
    qualified_asc = numba_coop.radix_rank(
        qualified_group,
        qualified_keys,
        begin_bit=28,
        radix_bits=4,
    )
    common_desc = coop.radix_rank(
        common_group,
        common_keys,
        begin_bit=28,
        radix_bits=4,
        descending=True,
    )
    qualified_desc = numba_coop.radix_rank(
        qualified_group,
        qualified_keys,
        begin_bit=28,
        end_bit=32,
        descending=True,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_original[index] = common_keys[item]
        qualified_original[index] = qualified_keys[item]
        common_ascending[index] = common_asc[item]
        qualified_ascending[index] = qualified_asc[item]
        common_descending[index] = common_desc[item]
        qualified_descending[index] = qualified_desc[item]


@cuda.jit
def _radix_rank64_kernel(
    source,
    common_original,
    qualified_original,
    common_ascending,
    qualified_ascending,
    common_descending,
    qualified_descending,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        value = source[index]
        common_keys[item] = value
        qualified_keys[item] = value

    common_group = coop.this_block()
    qualified_group = numba_coop.this_block()
    common_asc = coop.radix_rank(
        common_group,
        common_keys,
        begin_bit=60,
        end_bit=64,
    )
    qualified_asc = numba_coop.radix_rank(
        qualified_group,
        qualified_keys,
        begin_bit=60,
        radix_bits=4,
    )
    common_desc = coop.radix_rank(
        common_group,
        common_keys,
        begin_bit=60,
        radix_bits=4,
        descending=True,
    )
    qualified_desc = numba_coop.radix_rank(
        qualified_group,
        qualified_keys,
        begin_bit=60,
        end_bit=64,
        descending=True,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_original[index] = common_keys[item]
        qualified_original[index] = qualified_keys[item]
        common_ascending[index] = common_asc[item]
        qualified_ascending[index] = qualified_asc[item]
        common_descending[index] = common_desc[item]
        qualified_descending[index] = qualified_desc[item]


def _keys_with_high_bits(dtype: type[np.generic]) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    bit_width = resolved_dtype.itemsize * 8
    mask = (1 << bit_width) - 1
    sign_bit = 1 << (bit_width - 1)
    multiplier = 0x9E37_79B9_7F4A_7C15 & mask
    raw = [
        ((index * multiplier) ^ (index << 17) ^ (index * index * 29)) & mask
        for index in range(_TOTAL_ITEMS)
    ]
    for index in range(11, _TOTAL_ITEMS, 17):
        raw[index] = raw[index - 1]
    raw[:8] = [
        0,
        sign_bit,
        mask,
        sign_bit - 1,
        1,
        sign_bit | 1,
        mask - 1,
        sign_bit + 2,
    ]

    unsigned_dtype = np.dtype(f"uint{bit_width}")
    unsigned = np.asarray(raw, dtype=unsigned_dtype)
    if np.issubdtype(resolved_dtype, np.signedinteger):
        return unsigned.view(resolved_dtype).copy()
    return unsigned.astype(resolved_dtype, copy=False)


def _stable_rank_oracle(
    values: np.ndarray,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
) -> np.ndarray:
    bit_width = values.dtype.itemsize * 8
    raw_mask = (1 << bit_width) - 1
    sign_bit = 1 << (bit_width - 1)
    digit_mask = (1 << (end_bit - begin_bit)) - 1
    signed = np.issubdtype(values.dtype, np.signedinteger)
    digits = []
    for value in values:
        ordered = int(value) & raw_mask
        if signed:
            ordered ^= sign_bit
        digits.append((ordered >> begin_bit) & digit_mask)

    ranks = np.empty(values.size, dtype=np.int32)
    for index, digit in enumerate(digits):
        rank = 0
        for peer_index, peer_digit in enumerate(digits):
            before = peer_digit > digit if descending else peer_digit < digit
            if before or (peer_digit == digit and peer_index < index):
                rank += 1
        ranks[index] = rank
    return ranks


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.evidence_for("group.radix_rank", backend="numba_mlir", evidence="runtime")
def test_common_and_qualified_radix_rank_match_stable_bit_ordered_oracles(dtype):
    values = _keys_with_high_bits(dtype)
    originals = [np.full_like(values, 7) for _ in range(2)]
    ranks = [np.full(values.size, -1, dtype=np.int32) for _ in range(4)]
    kernel = (
        _radix_rank64_kernel if values.dtype.itemsize == 8 else _radix_rank32_kernel
    )

    kernel[1, _THREADS](values, *originals, *ranks)

    common_original, qualified_original = originals
    common_ascending, qualified_ascending, common_descending, qualified_descending = (
        ranks
    )
    np.testing.assert_array_equal(common_original, values)
    np.testing.assert_array_equal(qualified_original, values)
    bit_width = values.dtype.itemsize * 8
    begin_bit = bit_width - 4
    for common_result, qualified_result, descending in (
        (common_ascending, qualified_ascending, False),
        (common_descending, qualified_descending, True),
    ):
        np.testing.assert_array_equal(common_result, qualified_result)
        np.testing.assert_array_equal(
            common_result,
            _stable_rank_oracle(
                values,
                begin_bit=begin_bit,
                end_bit=bit_width,
                descending=descending,
            ),
        )


@cuda.jit
def _radix_rank_width8_kernel(source, common_output, qualified_output):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        value = source[index]
        common_keys[item] = value
        qualified_keys[item] = value

    common_ranks = coop.radix_rank(
        coop.this_block(),
        common_keys,
        begin_bit=24,
        radix_bits=8,
    )
    qualified_ranks = numba_coop.radix_rank(
        numba_coop.this_block(),
        qualified_keys,
        begin_bit=24,
        end_bit=32,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_output[index] = common_ranks[item]
        qualified_output[index] = qualified_ranks[item]


@pytest.mark.evidence_for("group.radix_rank", backend="numba_mlir", evidence="runtime")
def test_common_and_qualified_radix_rank_execute_width_eight_boundary():
    values = _keys_with_high_bits(np.int32)
    common = np.full(values.size, -1, dtype=np.int32)
    qualified = np.full(values.size, -1, dtype=np.int32)

    _radix_rank_width8_kernel[1, _THREADS](values, common, qualified)

    expected = _stable_rank_oracle(
        values,
        begin_bit=24,
        end_bit=32,
        descending=False,
    )
    np.testing.assert_array_equal(common, qualified)
    np.testing.assert_array_equal(common, expected)
