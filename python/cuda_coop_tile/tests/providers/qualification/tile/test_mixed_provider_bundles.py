# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import re

import pytest

from ...tile.native.adjacent_shuffle_exchange import _resolved_exchange_call
from ...tile.native.discontinuity import (
    _discontinuity_callsite,
    _resolved_discontinuity_call,
)
from ...tile.native.harness import (
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
from ...tile.native.histogram_rld import (
    _histogram_callsite,
    _resolved_histogram_call,
)
from ...tile.native.radix import _resolved_radix_rank_call
from ...tile.native.reduce_scan import (
    _resolved_scan_call,
    _scan_callsite,
)
from ...tile.native.topk_sort import _resolved_topk_call

pytestmark = [pytest.mark.qualification, pytest.mark.compile]


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
        for callsite in (histogram_callsite, fixed_callsite, scan_callsite):
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
        for callsite in (heads_callsite, combined_callsite, scan_callsite):
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
