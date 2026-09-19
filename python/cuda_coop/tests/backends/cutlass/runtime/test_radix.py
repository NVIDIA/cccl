# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Stable radix ordering, inverse ranks, and independent bin-prefix oracles."""

from contextlib import ExitStack

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
    values_for,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_INTEGER_KEYS = (np.int32, np.uint32, np.int64, np.uint64)


class _Readonly:
    def __init__(self, source):
        self.items_per_thread = source.items_per_thread
        self.dtype = source.dtype
        self.alignment = source.alignment
        self._items = tuple(source)

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


def _keys(dtype, size):
    dtype = np.dtype(dtype)
    result = values_for(dtype, size, shift=19)
    if dtype.kind == "f":
        unsigned = np.dtype(f"uint{dtype.itemsize * 8}")
        if dtype.itemsize == 4:
            patterns = [
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC00001,
                0x7FC00123,
                0xFFC00005,
                0xFFC00019,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x3FC00000,
                0xBFC00000,
            ]
        else:
            patterns = [
                0x0000000000000000,
                0x8000000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000000001,
                0x7FF8000000000123,
                0xFFF8000000000005,
                0xFFF8000000000019,
                0x7FEFFFFFFFFFFFFF,
                0xFFEFFFFFFFFFFFFF,
                0x3FF8000000000000,
                0xBFF8000000000000,
            ]
        bits = result.view(unsigned)
        bits[: len(patterns)] = patterns
        # Repeat signed zeros and NaN payloads to expose unstable ties.
        bits[len(patterns) : 2 * len(patterns)] = patterns[::-1]
    else:
        limits = np.iinfo(dtype)
        result[:6] = [limits.min, limits.max, 0, 1, limits.min, limits.max]
    permutation = np.random.default_rng(23).permutation(size)
    return result[permutation].copy()


def _digits(keys, begin, end):
    unsigned = np.dtype(f"uint{keys.dtype.itemsize * 8}")
    bits = keys.view(unsigned).copy()
    sign = unsigned.type(1 << (keys.dtype.itemsize * 8 - 1))
    if keys.dtype.kind == "i":
        bits ^= sign
    elif keys.dtype.kind == "f":
        bits = np.where(bits & sign, ~bits, bits ^ sign)
        # Signed zeros compare equally while retaining their original bits.
        bits[keys == 0] = sign
    return (bits >> begin) & unsigned.type((1 << (end - begin)) - 1)


def _permutation(keys, begin, end, descending):
    digits = _digits(keys, begin, end)
    return np.argsort(~digits if descending else digits, kind="stable")


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def _check_result(result, source, dtype, items, scalar, alignment):
    if scalar:
        assert isinstance(result, dtype)
    else:
        assert isinstance(result, cutlass_coop.ThreadData)
        assert result is not source
        assert result.dtype is dtype
        assert len(result) == items
        assert result.alignment == alignment


def _run_sort(
    api=coop,
    *,
    dtype=np.int32,
    value_dtype=np.float64,
    pairs=True,
    descending=False,
    begin_bit=0,
    end_bit=None,
    bounds="static",
    control_type=cutlass.Int64,
    scalar=False,
    striped=False,
    readonly=False,
    inferred=False,
    block=(8, 4, 2),
    blocks=2,
    items=3,
    compile_options=(),
    reuse=False,
    sharing=None,
    capacity=None,
    auto_sync=True,
    alignment=64,
    chain=False,
):
    key_type, value_type = cutlass_dtype(dtype), cutlass_dtype(value_dtype)
    items = 1 if scalar else items
    threads = int(np.prod(block))
    tile, size = threads * items, blocks * threads * items
    resolved_end = np.dtype(dtype).itemsize * 8 if end_bit is None else end_bit
    is_qualified = api is cutlass_coop

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        key_check: cute.Pointer,
        value_check: cute.Pointer,
        dynamic_begin: control_type,
        dynamic_end: control_type,
        repeats: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        block_index, _, _ = cute.arch.block_idx()
        offset = block_index * tile
        sources = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), key_type
        )
        payloads = cute.recast_tensor(
            cute.make_tensor(payload, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout(size)), key_type
        )
        associated = cute.recast_tensor(
            cute.make_tensor(associations, cute.make_layout(size)), value_type
        )
        keys_check = cute.recast_tensor(
            cute.make_tensor(key_check, cute.make_layout(size)), key_type
        )
        values_check = cute.recast_tensor(
            cute.make_tensor(value_check, cute.make_layout(size)), value_type
        )
        group = api.this_block()
        if cutlass.const_expr(scalar):
            keys = sources[offset + thread]
            values = payloads[offset + thread]
        else:
            keys = api.ThreadData(
                items, dtype=None if inferred else key_type, alignment=alignment
            )
            values = api.ThreadData(
                items, dtype=None if inferred else value_type, alignment=alignment
            )
            for item in cutlass.range_constexpr(items):
                keys[item] = sources[offset + thread * items + item]
                values[item] = payloads[offset + thread * items + item]
        if cutlass.const_expr(readonly):
            key_input, value_input = _Readonly(keys), _Readonly(values)
        else:
            key_input, value_input = keys, values
        if cutlass.const_expr(sharing is None):
            storage = None
        else:
            storage = api.TempStorage(
                capacity, sharing=sharing, auto_sync=auto_sync, alignment=alignment
            )
        if cutlass.const_expr(bounds in {"runtime", "runtime-begin"}):
            begin = dynamic_begin
        else:
            begin = begin_bit
        if cutlass.const_expr(bounds in {"runtime", "runtime-end"}):
            end = dynamic_end
        else:
            end = end_bit
        result, result_values = keys, values
        for iteration in range(repeats):
            if cutlass.const_expr(pairs):
                if cutlass.const_expr(is_qualified):
                    result, result_values = api.radix_sort_pairs(
                        group,
                        key_input,
                        value_input,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                        blocked_to_striped=striped,
                    )
                else:
                    result, result_values = api.radix_sort_pairs(
                        group,
                        key_input,
                        value_input,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                    )
            else:
                if cutlass.const_expr(is_qualified):
                    result = api.radix_sort_keys(
                        group,
                        key_input,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                        blocked_to_striped=striped,
                    )
                else:
                    result = api.radix_sort_keys(
                        group,
                        key_input,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                    )
            if cutlass.const_expr(sharing is not None and not auto_sync):
                storage.sync()
            if cutlass.const_expr(chain):
                if cutlass.const_expr(pairs):
                    result, result_values = api.radix_sort_pairs(
                        group,
                        result,
                        result_values,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                    )
                else:
                    result = api.radix_sort_keys(
                        group,
                        result,
                        begin_bit=begin,
                        end_bit=end,
                        descending=descending,
                        temp_storage=storage,
                    )
                if cutlass.const_expr(sharing is not None and not auto_sync):
                    storage.sync()
        _check_result(result, keys, key_type, items, scalar, alignment)
        if cutlass.const_expr(pairs):
            _check_result(result_values, values, value_type, items, scalar, alignment)
        if cutlass.const_expr(scalar):
            outputs[offset + thread] = result
            associated[offset + thread] = result_values
            keys_check[offset + thread] = keys
            values_check[offset + thread] = values
        else:
            for item in cutlass.range_constexpr(items):
                if cutlass.const_expr(striped):
                    index = offset + item * threads + thread
                else:
                    index = offset + thread * items + item
                outputs[index] = result[item]
                associated[index] = result_values[item]
                keys_check[offset + thread * items + item] = keys[item]
                values_check[offset + thread * items + item] = values[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        key_check: cute.Pointer,
        value_check: cute.Pointer,
        dynamic_begin: control_type,
        dynamic_end: control_type,
        repeats: cutlass.Int32,
    ):
        kernel(
            source,
            payload,
            output,
            associations,
            key_check,
            value_check,
            dynamic_begin,
            dynamic_end,
            repeats,
        ).launch(grid=blocks, block=block)

    source = _keys(dtype, size)
    payload = (np.arange(size) % 113).astype(value_dtype)
    if np.dtype(value_dtype).kind == "f":
        payload = np.arange(size, dtype=value_dtype) + np.dtype(value_dtype).type(0.25)
    observed, associated = np.empty_like(source), np.empty_like(payload)
    preserved_keys, preserved_values = np.empty_like(source), np.empty_like(payload)
    arrays = (source, payload, observed, associated, preserved_keys, preserved_values)
    with ExitStack() as stack:
        pointers = [stack.enter_context(device_array(array)) for array in arrays]
        args = (
            *pointers,
            control_type(begin_bit),
            control_type(resolved_end),
            cutlass.Int32(3 if reuse else 1),
        )
        compiled = (
            cute.compile[compile_options](launch, *args)
            if compile_options
            else cute.compile(launch, *args)
        )
        compiled(*args)
    _assert_bits(preserved_keys, source)
    _assert_bits(preserved_values, payload)
    for start in range(0, size, tile):
        order = _permutation(
            source[start : start + tile], begin_bit, resolved_end, descending
        )
        _assert_bits(
            observed[start : start + tile], source[start : start + tile][order]
        )
        if pairs:
            _assert_bits(
                associated[start : start + tile], payload[start : start + tile][order]
            )
    return observed, associated


def _run_rank(
    api=coop,
    *,
    dtype=np.int32,
    begin_bit=0,
    radix_bits=4,
    descending=False,
    prefix=False,
    prefix_inferred=False,
    scalar=False,
    readonly=False,
    inferred=False,
    block=(8, 4, 2),
    blocks=2,
    items=3,
    compile_options=(),
    reuse=False,
    chain=False,
):
    key_type = cutlass_dtype(dtype)
    items = 1 if scalar else items
    threads = int(np.prod(block))
    tile, size = threads * items, blocks * threads * items
    bins = 1 << radix_bits
    prefix_items = max(1, (bins + threads - 1) // threads)
    prefix_tile = threads * prefix_items

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        output: cute.Pointer,
        prefix_output: cute.Pointer,
        preserved: cute.Pointer,
        ordered: cute.Pointer,
        repeats: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        block_index, _, _ = cute.arch.block_idx()
        offset = block_index * tile
        sources = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), key_type
        )
        outputs = cute.make_tensor(output, cute.make_layout(size))
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), key_type
        )
        ordered_out = cute.make_tensor(ordered, cute.make_layout(size))
        prefixes = cute.make_tensor(
            prefix_output, cute.make_layout(blocks * prefix_tile)
        )
        group = api.this_block()
        if cutlass.const_expr(scalar):
            keys = sources[offset + thread]
        else:
            keys = api.ThreadData(
                items, dtype=None if inferred else key_type, alignment=64
            )
            for item in cutlass.range_constexpr(items):
                keys[item] = sources[offset + thread * items + item]
        if cutlass.const_expr(readonly):
            input_keys = _Readonly(keys)
        else:
            input_keys = keys
        if cutlass.const_expr(prefix):
            digit_prefix = api.ThreadData(
                prefix_items, dtype=None if prefix_inferred else cutlass.Int32
            )
        if cutlass.const_expr(scalar):
            result = cutlass.Int32(0)
        else:
            result = api.ThreadData(items, dtype=cutlass.Int32, alignment=64)
            for item in cutlass.range_constexpr(items):
                result[item] = cutlass.Int32(0)
        for iteration in range(repeats):
            if cutlass.const_expr(prefix):
                result = api.radix_rank(
                    group,
                    input_keys,
                    begin_bit=begin_bit,
                    radix_bits=radix_bits,
                    descending=descending,
                    exclusive_digit_prefix=digit_prefix,
                )
            else:
                result = api.radix_rank(
                    group,
                    input_keys,
                    begin_bit=begin_bit,
                    radix_bits=radix_bits,
                    descending=descending,
                )
            if cutlass.const_expr(prefix):
                for item in cutlass.range_constexpr(prefix_items):
                    prefixes[
                        block_index * prefix_tile + thread * prefix_items + item
                    ] = digit_prefix[item]
        _check_result(result, keys, cutlass.Int32, items, scalar, 64)
        if cutlass.const_expr(chain):
            sorted_ranks = api.radix_sort_keys(group, result)
        if cutlass.const_expr(scalar):
            outputs[offset + thread] = result
            checks[offset + thread] = keys
            if cutlass.const_expr(chain):
                ordered_out[offset + thread] = sorted_ranks
        else:
            for item in cutlass.range_constexpr(items):
                outputs[offset + thread * items + item] = result[item]
                checks[offset + thread * items + item] = keys[item]
                if cutlass.const_expr(chain):
                    ordered_out[offset + thread * items + item] = sorted_ranks[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        output: cute.Pointer,
        prefix_output: cute.Pointer,
        preserved: cute.Pointer,
        ordered: cute.Pointer,
        repeats: cutlass.Int32,
    ):
        kernel(source, output, prefix_output, preserved, ordered, repeats).launch(
            grid=blocks, block=block
        )

    source = _keys(dtype, size)
    observed = np.empty(size, dtype=np.int32)
    prefixes = np.full(blocks * prefix_tile, -777, dtype=np.int32)
    preserved, ordered = np.empty_like(source), np.empty(size, dtype=np.int32)
    with ExitStack() as stack:
        pointers = [
            stack.enter_context(device_array(array))
            for array in (source, observed, prefixes, preserved, ordered)
        ]
        args = (*pointers, cutlass.Int32(3 if reuse else 1))
        compiled = (
            cute.compile[compile_options](launch, *args)
            if compile_options
            else cute.compile(launch, *args)
        )
        compiled(*args)
    _assert_bits(preserved, source)
    for block_index, start in enumerate(range(0, size, tile)):
        digits = _digits(
            source[start : start + tile], begin_bit, begin_bit + radix_bits
        )
        expected = np.empty(tile, dtype=np.int32)
        for index, digit in enumerate(digits):
            preceding = digits > digit if descending else digits < digit
            expected[index] = np.count_nonzero(preceding) + np.count_nonzero(
                digits[:index] == digit
            )
        np.testing.assert_array_equal(observed[start : start + tile], expected)
        if chain:
            np.testing.assert_array_equal(
                ordered[start : start + tile], np.arange(tile, dtype=np.int32)
            )
        if prefix:
            counts = np.bincount(digits.astype(np.int64), minlength=bins)
            expected_prefix = (
                tile - np.cumsum(counts)
                if descending
                else np.r_[0, np.cumsum(counts[:-1])]
            )
            # Both directions retain ascending bin ownership; trailing slots are undefined.
            first = block_index * prefix_tile
            np.testing.assert_array_equal(
                prefixes[first : first + bins], expected_prefix
            )
    return observed, prefixes


@pytest.mark.parametrize("dtype", _INTEGER_KEYS)
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize("pairs", (False, True))
def test_common_integral_sort(dtype, descending, pairs):
    _run_sort(dtype=dtype, descending=descending, pairs=pairs)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize("striped", (False, True))
@pytest.mark.parametrize("selected", (False, True))
def test_qualified_float_bits_and_stable_ties(dtype, descending, striped, selected):
    width = np.dtype(dtype).itemsize * 8
    _run_sort(
        cutlass_coop,
        dtype=dtype,
        descending=descending,
        striped=striped,
        begin_bit=width - 8 if selected else 0,
        end_bit=width,
    )


@pytest.mark.parametrize("value_dtype", NUMPY_DTYPES)
def test_independent_pair_value_dtypes(value_dtype):
    _run_sort(value_dtype=value_dtype, begin_bit=2, end_bit=6, blocks=1)


@pytest.mark.parametrize(
    "bounds", ("static", "runtime", "runtime-begin", "runtime-end")
)
def test_mixed_bit_bounds(bounds):
    _run_sort(dtype=np.int64, begin_bit=59, end_bit=64, bounds=bounds, descending=True)


@pytest.mark.parametrize("dtype", (np.int32, np.uint64))
def test_nonzero_begin_with_default_end(dtype):
    _run_sort(dtype=dtype, begin_bit=3, end_bit=None, bounds="runtime-begin")


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("pairs", (False, True))
def test_readonly_inferred_sort_composes(api, pairs):
    _run_sort(
        api, dtype=np.int64, pairs=pairs, readonly=True, inferred=True, chain=True
    )


@pytest.mark.parametrize(
    "dtype,pairs,descending",
    (
        (np.int32, False, False),
        (np.uint64, True, True),
        (np.float32, False, True),
        (np.float64, True, False),
    ),
)
def test_qualified_scalar_sort(dtype, pairs, descending):
    _run_sort(
        cutlass_coop, dtype=dtype, pairs=pairs, descending=descending, scalar=True
    )


@pytest.mark.parametrize("dtype", _INTEGER_KEYS)
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize("sign_window", (False, True))
def test_rank_stability_and_sign_window(dtype, descending, sign_window):
    begin = np.dtype(dtype).itemsize * 8 - 4 if sign_window else 0
    _run_rank(dtype=dtype, descending=descending, begin_bit=begin, chain=True)


@pytest.mark.parametrize(
    "bits,block",
    (
        (1, (8, 4, 2)),
        (4, (8, 4, 2)),
        (7, (8, 4, 2)),
        (8, (8, 4, 1)),
    ),
)
@pytest.mark.parametrize("descending", (False, True))
def test_prefix_ascending_bin_ownership(bits, block, descending):
    _run_rank(
        cutlass_coop,
        dtype=np.int64,
        radix_bits=bits,
        block=block,
        descending=descending,
        prefix=True,
        prefix_inferred=True,
    )


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_readonly_inferred_rank_composes(api):
    _run_rank(api, dtype=np.uint64, readonly=True, inferred=True, chain=True)


@pytest.mark.parametrize("descending", (False, True))
def test_qualified_scalar_rank(descending):
    _run_rank(
        cutlass_coop,
        dtype=np.int32,
        scalar=True,
        descending=descending,
        prefix=True,
        chain=True,
    )


def test_non_power_of_two_multidimensional_block():
    _run_sort(block=(8, 3, 2), begin_bit=1, end_bit=9)
    _run_rank(block=(8, 3, 2), radix_bits=4)
