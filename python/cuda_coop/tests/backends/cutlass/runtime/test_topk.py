# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent TopK membership, pairing, and preservation oracles."""

from collections import Counter
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


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def _bit_counts(values):
    return Counter(row.tobytes() for row in values.reshape(-1, 1))


def _check_result(result, source, dtype, items, alignment):
    assert isinstance(result, cutlass_coop.ThreadData)
    assert result is not source
    assert result.dtype is dtype
    assert len(result) == items
    assert result.alignment == alignment


def _run(
    api=coop,
    *,
    mode="min",
    pairs=True,
    dtype=np.int32,
    value_dtype=np.int64,
    k=17,
    valid_items=None,
    controls="runtime",
    control_type=cutlass.Int64,
    threads=64,
    items=2,
    blocks=2,
    readonly=False,
    inferred=False,
    reuse=False,
    sharing=None,
    capacity=None,
    alignment=64,
    auto_sync=True,
    chain=False,
    compile_options=(),
    source=None,
    cases=None,
):
    key_type, value_type = cutlass_dtype(dtype), cutlass_dtype(value_dtype)
    tile, size = threads * items, blocks * threads * items
    topk = getattr(api, f"topk_{mode}_{'pairs' if pairs else 'keys'}")
    opposite = "max" if mode == "min" else "min"
    next_topk = getattr(api, f"topk_{opposite}_{'pairs' if pairs else 'keys'}")
    if cases is None:
        cases = [(k, tile if valid_items is None else valid_items)]
    else:
        assert controls == "runtime" and valid_items is not None

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        key_check: cute.Pointer,
        value_check: cute.Pointer,
        dynamic_k: control_type,
        dynamic_count: control_type,
        selected_count: cutlass.Int64,
        output_count: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
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
        if cutlass.const_expr(controls in {"runtime", "runtime-k"}):
            keep = dynamic_k
        else:
            keep = k
        if cutlass.const_expr(valid_items is None):
            count = None
        elif cutlass.const_expr(controls in {"runtime", "runtime-count"}):
            count = dynamic_count
        else:
            count = valid_items
        result, result_values = keys, values
        for iteration in range(repeats):
            if cutlass.const_expr(pairs):
                result, result_values = topk(
                    group,
                    key_input,
                    value_input,
                    k=keep,
                    valid_items=count,
                    temp_storage=storage,
                )
            else:
                result = topk(
                    group, key_input, k=keep, valid_items=count, temp_storage=storage
                )
            if cutlass.const_expr(sharing is not None and not auto_sync):
                storage.sync()
            if cutlass.const_expr(chain):
                if cutlass.const_expr(pairs):
                    result, result_values = next_topk(
                        group,
                        result,
                        result_values,
                        k=1,
                        valid_items=selected_count,
                        temp_storage=storage,
                    )
                else:
                    result = next_topk(
                        group,
                        result,
                        k=1,
                        valid_items=selected_count,
                        temp_storage=storage,
                    )
                if cutlass.const_expr(sharing is not None and not auto_sync):
                    storage.sync()
        _check_result(result, keys, key_type, items, alignment)
        if cutlass.const_expr(pairs):
            _check_result(result_values, values, value_type, items, alignment)
        for item in cutlass.range_constexpr(items):
            index = thread * items + item
            if index < output_count:
                outputs[offset + index] = result[item]
                if cutlass.const_expr(pairs):
                    associated[offset + index] = result_values[item]
            keys_check[offset + index] = keys[item]
            values_check[offset + index] = values[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        key_check: cute.Pointer,
        value_check: cute.Pointer,
        dynamic_k: control_type,
        dynamic_count: control_type,
        selected_count: cutlass.Int64,
        output_count: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        kernel(
            source,
            payload,
            output,
            associations,
            key_check,
            value_check,
            dynamic_k,
            dynamic_count,
            selected_count,
            output_count,
            repeats,
        ).launch(grid=blocks, block=threads)

    original = (
        values_for(dtype, size)
        if source is None
        else np.ascontiguousarray(source, dtype=dtype)
    )
    assert len(original) == size
    payload = np.tile(np.arange(tile), blocks).astype(value_dtype)
    np.testing.assert_array_equal(payload[:tile], np.arange(tile))
    compiled = None
    for keep, count in cases:
        selected = min(keep, count)
        written = min(1, selected) if chain else selected
        source, input_values = original.copy(), payload.copy()
        observed, associated = np.empty_like(source), np.empty_like(input_values)
        preserved_keys, preserved_values = np.empty_like(source), np.empty_like(payload)
        arrays = (
            source,
            input_values,
            observed,
            associated,
            preserved_keys,
            preserved_values,
        )
        with ExitStack() as stack:
            pointers = [stack.enter_context(device_array(array)) for array in arrays]
            args = (
                *pointers,
                control_type(keep),
                control_type(count),
                cutlass.Int64(selected),
                cutlass.Int64(written),
                cutlass.Int32(3 if reuse else 1),
            )
            if compiled is None:
                compiled = (
                    cute.compile[compile_options](launch, *args)
                    if compile_options
                    else cute.compile(launch, *args)
                )
            compiled(*args)
        _assert_bits(source, original)
        _assert_bits(input_values, payload)
        _assert_bits(preserved_keys, original)
        _assert_bits(preserved_values, payload)
        for start in range(0, size, tile):
            candidates = original[start : start + count]
            ordered = np.sort(candidates)
            expected = (
                ordered[:selected] if mode == "min" else ordered[count - selected :]
            )
            if chain and selected:
                expected = expected[-1:] if mode == "min" else expected[:1]
            actual = observed[start : start + written]
            np.testing.assert_array_equal(np.sort(actual), expected)
            assert not (_bit_counts(actual) - _bit_counts(candidates))
            if pairs:
                selected_values = associated[start : start + written]
                indices = selected_values.astype(np.int64)
                assert len(set(indices)) == written
                assert np.all((0 <= indices) & (indices < count))
                _assert_bits(selected_values, payload[start + indices])
                _assert_bits(actual, original[start + indices])
    return observed, associated


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("mode", ("min", "max"))
def test_numeric_counts(dtype, mode):
    _run(
        dtype=dtype,
        mode=mode,
        pairs=False,
        valid_items=128,
        cases=[(0, 128), (0, 0), (17, 0), (17, 7), (128, 128), (7, 91)],
    )


@pytest.mark.parametrize("value_dtype", NUMPY_DTYPES)
def test_independent_pair_value_types(value_dtype):
    _run(dtype=np.float64, value_dtype=value_dtype, valid_items=59, threads=32)


@pytest.mark.parametrize("api", (coop, cutlass_coop))
@pytest.mark.parametrize("mode", ("min", "max"))
@pytest.mark.parametrize("pairs", (False, True))
def test_common_and_qualified_entrypoints(api, mode, pairs):
    _run(api, mode=mode, pairs=pairs, valid_items=103)


@pytest.mark.parametrize("threads,items", ((1, 3), (16, 1), (32, 4), (128, 2)))
@pytest.mark.parametrize("mode", ("min", "max"))
def test_static_controls_block_sizes_and_chaining(threads, items, mode):
    _run(
        mode=mode,
        threads=threads,
        items=items,
        k=min(3, threads * items),
        controls="static",
        chain=True,
    )


@pytest.mark.parametrize("controls", ("static", "runtime-k", "runtime-count"))
@pytest.mark.parametrize("pairs", (False, True))
def test_mixed_count_bindings(controls, pairs):
    _run(controls=controls, pairs=pairs, k=7, valid_items=91)


@pytest.mark.parametrize("api", (coop, cutlass_coop))
@pytest.mark.parametrize("pairs", (False, True))
def test_readonly_inferred_composition(api, pairs):
    _run(
        api,
        pairs=pairs,
        dtype=np.uint32,
        value_dtype=np.float32,
        valid_items=103,
        readonly=True,
        inferred=True,
        chain=True,
    )


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("mode", ("min", "max"))
def test_selected_signed_zero_pairs_preserve_original_bits(dtype, mode):
    nonzero = 1 if mode == "min" else -1
    source = np.tile(np.array([-0.0, 0.0, nonzero, 2 * nonzero], dtype=dtype), 64)
    _run(cutlass_coop, mode=mode, dtype=dtype, source=source)


@pytest.mark.parametrize("mode", ("min", "max"))
@pytest.mark.parametrize("pairs", (False, True))
def test_duplicate_keys_do_not_require_stable_or_sorted_results(mode, pairs):
    source = np.tile(np.array([5, 5, 2, 9, 2, 5, 9, 2], dtype=np.int32), 32)
    _run(mode=mode, pairs=pairs, source=source, k=43, valid_items=103)
