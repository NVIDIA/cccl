# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import re

import pytest

from .adjacent_shuffle_exchange import _resolved_exchange_call
from .discontinuity import _discontinuity_callsite, _resolved_discontinuity_call
from .harness import (
    TILEIR_LP_REQUEST_FORMAT_VERSION,
    TILEIR_LP_SUCCESS,
    TileIRForeignModule,
    TileIRLPCallsite,
    TileIRLPErrorPtr,
    TileIRLPProvideImplsRequest,
    TileIRResolvedCall,
    _compilation_context,
    _create_provider,
    _load_provider_or_skip,
    _provide_impls,
    _string_view_value,
)
from .histogram_rld import _histogram_callsite, _resolved_histogram_call
from .radix import _resolved_radix_rank_call
from .reduce_scan import _resolved_scan_call, _scan_callsite
from .topk_sort import (
    _merge_sort_callsite,
    _merge_sort_impl_symbol,
    _resolved_merge_sort_call,
)


def test_native_provider_discovery_filters_configured_scratch_limit(
    monkeypatch, tmp_path
):
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(2, "f32")
        callsites = (TileIRLPCallsite * 1)(callsite)
        context = _compilation_context()
        context.num_warps_per_cta = -1
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            context,
            None,
            0,
        )
        monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_MAX_STATIC_SMEM_BYTES", "1")
        assert _provide_impls(instance, request) == b""
        assert "discovery_resource_over_budget" in trace_file.read_text()

        # A budget between the candidate layouts retains the smaller ones.
        monkeypatch.setenv(
            "CUDA_COOP_TILE_PROVIDER_MAX_STATIC_SMEM_BYTES", "1024"
        )
        limited = _provide_impls(instance, request)
        assert 0 < limited.count(b"cuda_coop_tile_scan_impl__lpv4_") < 4

        monkeypatch.delenv("CUDA_COOP_TILE_PROVIDER_MAX_STATIC_SMEM_BYTES")
        assert (
            _provide_impls(instance, request).count(
                b"cuda_coop_tile_scan_impl__lpv4_"
            )
            == 4
        )
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_batches_auto_resource_queries_and_reuses_cache(
    monkeypatch,
    tmp_path,
):
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(2, "f32")
        callsites = (TileIRLPCallsite * 1)(callsite)
        context = _compilation_context()
        context.num_warps_per_cta = -1
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            context,
            None,
            0,
        )

        for _ in range(2):
            catalog = _provide_impls(instance, request)
            assert catalog.startswith(b"\x7fTileIR\x00")
            assert catalog.count(b"cuda_coop_tile_scan_impl__lpv4_") == 4

        trace_lines = trace_file.read_text().splitlines()
        compile_lines = [
            line
            for line in trace_lines
            if "resource_query_nvrtc_compile" in line
        ]
        assert len(compile_lines) == 1
        assert "sm=sm_80 primitives=4" in compile_lines[0]
        cache_lines = [
            line for line in trace_lines if "resource_query_cache_hit" in line
        ]
        assert len(cache_lines) == 1
        assert "sm=sm_80 primitives=4" in cache_lines[0]
        assert not any("bundle_cache_" in line for line in trace_lines)
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_histogram_mixed_bundle_has_exact_resources(
    monkeypatch,
    capfd,
):
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        histogram_callsite, histogram_callsite_keepalive = _histogram_callsite(
            "histogram",
            256,
            128,
            block_threads=0,
            items_per_thread=0,
        )
        fixed_callsite, fixed_callsite_keepalive = _histogram_callsite(
            "histogram_256",
            128,
            256,
            block_threads=0,
            items_per_thread=0,
        )
        scan_callsite, scan_callsite_keepalive = _scan_callsite(
            1,
            "i32",
        )
        discovery_callsites = (
            histogram_callsite,
            fixed_callsite,
            scan_callsite,
        )
        for discovery_callsite in discovery_callsites:
            callsites = (TileIRLPCallsite * 1)(discovery_callsite)
            request = TileIRLPProvideImplsRequest(
                TILEIR_LP_REQUEST_FORMAT_VERSION,
                callsites,
                len(callsites),
                _compilation_context(),
                None,
                0,
            )
            catalog = _provide_impls(instance, request)
            assert catalog.startswith(b"\x7fTileIR\x00")

        histogram_call, histogram_keepalive = _resolved_histogram_call(
            "histogram",
            256,
            128,
            block_threads=0,
            items_per_thread=0,
            resolved_block_threads=128,
        )
        fixed_call, fixed_keepalive = _resolved_histogram_call(
            "histogram_256",
            128,
            256,
            block_threads=0,
            items_per_thread=0,
            resolved_block_threads=128,
        )
        scan_call, scan_keepalive = _resolved_scan_call(
            1,
            "i32",
        )
        calls = (TileIRResolvedCall * 3)(
            histogram_call,
            fixed_call,
            scan_call,
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
                for call in (histogram_call, fixed_call, scan_call)
            ]
            assert module.num_code_buffers == 1
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

        trace = capfd.readouterr().err
        resource_lines = [
            line for line in trace.splitlines() if "resource_estimate" in line
        ]
        assert resource_lines
        assert all(" sm=sm_80 " in line for line in resource_lines)
        histogram_resources = re.findall(
            r"resource_estimate .*payload=(histogram(?:_256)?) .*"
            r"temp_storage=(\d+) extra_static_smem=(\d+) "
            r"static_smem=(\d+) scratch_alignment=(\d+)",
            trace,
        )
        assert len(histogram_resources) == 2
        for payload, temp, extra, total, alignment in histogram_resources:
            expected_bins = 256 if payload == "histogram_256" else 128
            temp_bytes = int(temp)
            extra_bytes = int(extra)
            total_bytes = int(total)
            alignment_bytes = int(alignment)
            assert temp_bytes > 0
            assert extra_bytes >= expected_bins * ctypes.sizeof(ctypes.c_int)
            assert total_bytes == temp_bytes + extra_bytes
            assert alignment_bytes > 0
            assert alignment_bytes & (alignment_bytes - 1) == 0
        assert (
            histogram_callsite_keepalive
            and fixed_callsite_keepalive
            and scan_callsite_keepalive
            and histogram_keepalive
            and fixed_keepalive
            and scan_keepalive
        )
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_exchange_inout_instantiates_with_exact_resources(
    monkeypatch,
    capfd,
):
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        exchange_calls = [
            _resolved_exchange_call("blocked_to_striped"),
            _resolved_exchange_call("striped_to_blocked", dtype="u32"),
        ]
        scan_call, scan_keepalive = _resolved_scan_call(
            2,
            "i32",
        )
        calls = (TileIRResolvedCall * 3)(
            exchange_calls[0][0],
            exchange_calls[1][0],
            scan_call,
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
                for call in (
                    exchange_calls[0][0],
                    exchange_calls[1][0],
                    scan_call,
                )
            ]
            assert module.num_code_buffers == 1
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == expected_entries
        finally:
            instance.vtable.free_module(module_ptr)

        trace = capfd.readouterr().err
        resources = re.findall(
            r"resource_estimate .*payload=exchange .*"
            r"temp_storage=(\d+) extra_static_smem=(\d+) "
            r"static_smem=(\d+) scratch_alignment=(\d+)",
            trace,
        )
        assert len(resources) == 2
        for temp, extra, total, alignment in resources:
            assert int(temp) > 0
            assert int(extra) == 0
            assert int(total) == int(temp)
            assert int(alignment) > 0
            assert int(alignment) & (int(alignment) - 1) == 0
        assert exchange_calls[0][1] and exchange_calls[1][1] and scan_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_provider_radix_rank_runtime_bits_instantiate_with_exact_resources(
    monkeypatch,
    capfd,
):
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        rank_calls = (
            _resolved_radix_rank_call(),
            _resolved_radix_rank_call(
                dtype="i32",
                radix_bits=8,
                descending=True,
                memoize_outer_scan=False,
                inner_scan_algorithm="raking",
                smem_config="eight_byte",
            ),
        )
        scan_call, scan_keepalive = _resolved_scan_call(
            2,
            "i32",
        )
        calls = (TileIRResolvedCall * 3)(
            rank_calls[0][0],
            rank_calls[1][0],
            scan_call,
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
                for call in (rank_calls[0][0], rank_calls[1][0], scan_call)
            ]
            assert module.num_code_buffers == 1
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == expected_entries
        finally:
            instance.vtable.free_module(module_ptr)

        trace = capfd.readouterr().err
        resources = re.findall(
            r"resource_estimate .*payload=radix_rank .*"
            r"temp_storage=(\d+) extra_static_smem=(\d+) "
            r"static_smem=(\d+) scratch_alignment=(\d+)",
            trace,
        )
        assert len(resources) == 2
        for temp, extra, total, alignment in resources:
            assert int(temp) > 0
            assert int(extra) == 0
            assert int(total) == int(temp)
            assert int(alignment) > 0
            assert int(alignment) & (int(alignment) - 1) == 0
        assert rank_calls[0][1] and rank_calls[1][1] and scan_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_discontinuity_mixed_bundle_has_exact_resources(
    monkeypatch,
    capfd,
):
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        heads_callsite, heads_callsite_keepalive = _discontinuity_callsite(
            "flags",
            "heads",
        )
        combined_callsite, combined_callsite_keepalive = (
            _discontinuity_callsite(
                "heads_and_tails",
                "heads_and_tails",
            )
        )
        scan_callsite, scan_callsite_keepalive = _scan_callsite(
            2,
            "i32",
        )
        discovery_callsites = (heads_callsite, combined_callsite, scan_callsite)
        for discovery_callsite in discovery_callsites:
            callsites = (TileIRLPCallsite * 1)(discovery_callsite)
            request = TileIRLPProvideImplsRequest(
                TILEIR_LP_REQUEST_FORMAT_VERSION,
                callsites,
                len(callsites),
                _compilation_context(),
                None,
                0,
            )
            catalog = _provide_impls(instance, request)
            assert catalog.startswith(b"\x7fTileIR\x00")

        heads_call, heads_keepalive = _resolved_discontinuity_call(
            "flags",
            "heads",
        )
        combined_call, combined_keepalive = _resolved_discontinuity_call(
            "heads_and_tails",
            "heads_and_tails",
        )
        scan_call, scan_keepalive = _resolved_scan_call(
            2,
            "i32",
        )
        calls = (TileIRResolvedCall * 3)(
            heads_call,
            combined_call,
            scan_call,
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
                for call in (heads_call, combined_call, scan_call)
            ]
            assert module.num_code_buffers == 1
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == expected_entries
        finally:
            instance.vtable.free_module(module_ptr)

        trace = capfd.readouterr().err
        resources = re.findall(
            r"resource_estimate .*payload=discontinuity .*"
            r"temp_storage=(\d+) extra_static_smem=(\d+) "
            r"static_smem=(\d+) scratch_alignment=(\d+)",
            trace,
        )
        assert len(resources) == 2
        for temp, extra, total, alignment in resources:
            temp_bytes = int(temp)
            extra_bytes = int(extra)
            total_bytes = int(total)
            alignment_bytes = int(alignment)
            assert temp_bytes > 0
            assert extra_bytes == 0
            assert total_bytes == temp_bytes
            assert alignment_bytes > 0
            assert alignment_bytes & (alignment_bytes - 1) == 0
        assert (
            heads_callsite_keepalive
            and combined_callsite_keepalive
            and scan_callsite_keepalive
            and heads_keepalive
            and combined_keepalive
            and scan_keepalive
        )
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    (
        "family",
        "key_dtype",
        "value_dtype",
        "descending",
        "partial_tile",
        "items_per_thread",
    ),
    [
        pytest.param("keys", "i32", None, False, False, 1, id="keys-full"),
        pytest.param(
            "pairs",
            "i64",
            "f32",
            True,
            True,
            2,
            id="heterogeneous-pairs-partial",
        ),
    ],
)
def test_native_provider_merge_sort_has_exact_scratch_artifact(
    monkeypatch,
    capfd,
    family,
    key_dtype,
    value_dtype,
    descending,
    partial_tile,
    items_per_thread,
):
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE", "1")
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, callsite_keepalive = _merge_sort_callsite(
            family,
            items_per_thread,
            key_dtype,
            value_dtype,
            descending,
            partial_tile,
        )
        callsites = (TileIRLPCallsite * 1)(callsite)
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            _compilation_context(),
            None,
            0,
        )
        catalog = _provide_impls(instance, request)
        assert catalog.startswith(b"\x7fTileIR\x00")
        assert (
            _merge_sort_impl_symbol(
                family,
                items_per_thread,
                key_dtype,
                value_dtype,
                descending,
                partial_tile,
            ).encode()
            in catalog
        )
        call, call_keepalive = _resolved_merge_sort_call(
            family,
            items_per_thread,
            key_dtype,
            value_dtype,
            descending,
            partial_tile,
        )
        calls = (TileIRResolvedCall * 1)(call)
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
            assert module.num_code_buffers == 1
            code_buffer = module.code_buffers[0]
            assert code_buffer.data and code_buffer.size > 0
            expected_entry = _string_view_value(call.impl.contents.symbol)
            assert code_buffer.num_symbols == 1
            assert _string_view_value(code_buffer.symbols[0]) == expected_entry
            assert module.num_call_symbols == 1
            assert [
                _string_view_value(module.call_symbols[index].symbol)
                for index in range(module.num_call_symbols)
            ] == [expected_entry]
        finally:
            instance.vtable.free_module(module_ptr)

        diagnostic = capfd.readouterr().err
        assert re.search(
            rf"resource_estimate .*payload=merge_sort_{family} .*"
            r"temp_storage=[1-9][0-9]* .*static_smem=[1-9][0-9]* "
            r".*scratch_alignment=[1-9][0-9]*",
            diagnostic,
        )
        assert callsite_keepalive and call_keepalive
    finally:
        instance.vtable.destroy(instance.handle)
