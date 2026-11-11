# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

import pytest

from .harness import (
    TILEIR_LP_SUCCESS,
    TileIRForeignModule,
    TileIRLPErrorPtr,
    TileIRResolvedCall,
    _create_provider,
    _load_provider_or_skip,
    _string_view_value,
)
from .reduce_scan import _resolved_scan_call
from .topk_sort import _resolved_topk_call


def test_native_provider_scan_topk_mixed_scratch_reuse_bundle_compiles(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    generic_scan_options = {"mode": "inclusive", "scan_op": "max"}
    try:
        scan_sum, scan_sum_keepalive = _resolved_scan_call(
            1,
            "i32",
        )
        scan_generic, scan_generic_keepalive = _resolved_scan_call(
            1,
            "i32",
            **generic_scan_options,
        )
        topk_keys, topk_keys_keepalive = _resolved_topk_call(
            "keys",
            1,
            "i32",
            None,
            True,
            True,
        )
        topk_pairs, topk_pairs_keepalive = _resolved_topk_call(
            "pairs",
            1,
            "i64",
            "f32",
            False,
            False,
        )
        calls = (TileIRResolvedCall * 4)(
            scan_sum,
            scan_generic,
            topk_keys,
            topk_pairs,
        )
        module_ptr = ctypes.POINTER(TileIRForeignModule)()
        error = TileIRLPErrorPtr()

        assert (
            instance.vtable.instantiate(
                instance.handle,
                calls,
                len(calls),
                ctypes.byref(module_ptr),
                ctypes.byref(error),
            )
            == TILEIR_LP_SUCCESS
        )
        assert not error
        assert module_ptr
        try:
            module = module_ptr.contents
            expected_entries = [
                _string_view_value(call.impl.contents.symbol)
                for call in (scan_sum, scan_generic, topk_keys, topk_pairs)
            ]
            assert module.num_code_buffers == 1
            assert module.code_buffers[0].data
            assert module.code_buffers[0].size > 0
            assert [
                _string_view_value(module.code_buffers[0].symbols[index])
                for index in range(module.code_buffers[0].num_symbols)
            ] == expected_entries
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == expected_entries
        finally:
            instance.vtable.free_module(module_ptr)
        assert (
            scan_sum_keepalive
            and scan_generic_keepalive
            and topk_keys_keepalive
            and topk_pairs_keepalive
        )
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("architecture", ["sm_80", "sm_120a"])
def test_native_provider_batches_deduplicates_and_caches_call_symbols(
    monkeypatch, tmp_path, architecture
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    compile_counter = tmp_path / "nvrtc-compiles.txt"
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE",
        str(compile_counter),
    )
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    instance = _create_provider(lib)

    def instantiate(calls):
        call_array = (TileIRResolvedCall * len(calls))(*calls)
        module_ptr = ctypes.POINTER(TileIRForeignModule)()
        error = TileIRLPErrorPtr()
        assert (
            instance.vtable.instantiate(
                instance.handle,
                call_array,
                len(call_array),
                ctypes.byref(module_ptr),
                ctypes.byref(error),
            )
            == TILEIR_LP_SUCCESS
        )
        assert not error
        assert module_ptr
        return module_ptr

    call_one, keepalive_one = _resolved_scan_call(
        1, "f32", architecture=architecture
    )
    call_four, keepalive_four = _resolved_scan_call(
        4, "f32", architecture=architecture
    )
    call_one_uniquified, keepalive_one_uniquified = _resolved_scan_call(
        1,
        "f32",
        architecture=architecture,
        concrete_suffix="_0",
    )
    assert keepalive_one and keepalive_four and keepalive_one_uniquified
    expected_symbols = [
        _string_view_value(call.impl.contents.symbol)
        for call in (call_one, call_four)
    ]
    uniquified_symbol = _string_view_value(
        call_one_uniquified.impl.contents.symbol
    )
    assert expected_symbols[0].endswith("__CompilerConcrete1")
    assert uniquified_symbol.endswith("_0")

    try:
        # One semantic specialization may appear more than once in a compiler
        # batch under distinct selected names. Both names must be exported and
        # mapped, while the symbol-free resource identity remains shared.
        mixed_symbol_module = instantiate([call_one, call_one_uniquified])
        try:
            module = mixed_symbol_module.contents
            mixed_symbols = [expected_symbols[0], uniquified_symbol]
            assert module.num_code_buffers == 1
            assert [
                _string_view_value(module.code_buffers[0].symbols[index])
                for index in range(module.code_buffers[0].num_symbols)
            ] == mixed_symbols
            assert [
                (
                    module.call_symbols[index].call_index,
                    _string_view_value(module.call_symbols[index].symbol),
                )
                for index in range(module.num_call_symbols)
            ] == list(enumerate(mixed_symbols))
        finally:
            instance.vtable.free_module(mixed_symbol_module)
        assert compile_counter.read_text().splitlines() == ["1"]

        cached_mixed_symbol_module = instantiate(
            [call_one, call_one_uniquified]
        )
        instance.vtable.free_module(cached_mixed_symbol_module)
        assert compile_counter.read_text().splitlines() == ["1"]

        trace_lines = trace_file.read_text().splitlines()
        resource_compiles = [
            line
            for line in trace_lines
            if "resource_query_nvrtc_compile" in line
        ]
        assert len(resource_compiles) == 1
        assert f"sm={architecture} primitives=1" in resource_compiles[0]
        resource_hits = [
            line for line in trace_lines if "resource_query_cache_hit" in line
        ]
        assert len(resource_hits) == 1
        assert f"sm={architecture} primitives=2" in resource_hits[0]

        for _ in range(2):
            module_ptr = instantiate([call_one, call_four])
            try:
                module = module_ptr.contents
                assert module.num_code_buffers == 1
                code_buffer = module.code_buffers[0]
                assert [
                    _string_view_value(code_buffer.symbols[index])
                    for index in range(code_buffer.num_symbols)
                ] == expected_symbols
                assert module.num_call_symbols == 2
                assert [
                    (
                        module.call_symbols[index].call_index,
                        _string_view_value(module.call_symbols[index].symbol),
                    )
                    for index in range(module.num_call_symbols)
                ] == list(enumerate(expected_symbols))
            finally:
                instance.vtable.free_module(module_ptr)

        # The second identical batch is served from the provider's bundle cache.
        assert compile_counter.read_text().splitlines() == ["1", "1"]

        repeated_module = instantiate([call_one, call_one])
        try:
            module = repeated_module.contents
            assert module.code_buffers[0].num_symbols == 1
            assert module.num_call_symbols == 2
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == [expected_symbols[0]] * 2
        finally:
            instance.vtable.free_module(repeated_module)
        assert compile_counter.read_text().splitlines() == ["1", "1", "1"]

        cached_repeated_module = instantiate([call_one, call_one])
        instance.vtable.free_module(cached_repeated_module)
        assert compile_counter.read_text().splitlines() == ["1", "1", "1"]
    finally:
        instance.vtable.destroy(instance.handle)
