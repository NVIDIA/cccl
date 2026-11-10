# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import os
import re
import shutil
import subprocess

import pytest

from .canonical_cases import CANONICAL_FAMILY_CASES
from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    TILEIR_LP_CANDIDATE_PROVIDER,
    TILEIR_LP_RANK_SCORED,
    TILEIR_LP_REQUEST_FORMAT_VERSION,
    TILEIR_LP_SUCCESS,
    TILEIR_OUTPUT_LTO_IR,
    TileIRForeignModule,
    TileIRLPCallsite,
    TileIRLPErrorPtr,
    TileIRLPFilterAndRankRequest,
    TileIRLPFilterCandidate,
    TileIRLPProvideImplsRequest,
    TileIRLPRankResult,
    TileIRResolvedCall,
    _compilation_context,
    _create_provider,
    _load_provider_or_skip,
    _provide_impls,
    _provider_topology_score,
    _string_view,
    _string_view_value,
)
from .reduce_scan import (
    _batched_reduce_callsite,
    _resolved_scan_call,
    _scan_callsite,
    _scan_impl_symbol,
)


def _translate_catalog_or_skip(catalog: bytes, path) -> str:
    translator = os.environ.get(
        "CUDA_COOP_TILE_INTERFACE_TRANSLATOR"
    ) or shutil.which("cuda-tile-translate")
    if not translator:
        pytest.skip("catalog regression requires cuda-tile-translate")
    path.write_bytes(catalog)
    result = subprocess.run(
        [
            translator,
            "--cudatilebc-to-mlir",
            "--preview=simt",
            str(path),
            "-o",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_native_provider_empty_instantiate_is_successful_noop(monkeypatch):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        module = ctypes.POINTER(TileIRForeignModule)()
        error = TileIRLPErrorPtr()
        result = instance.vtable.instantiate(
            instance.handle,
            None,
            0,
            ctypes.byref(module),
            ctypes.byref(error),
        )
        assert result == TILEIR_LP_SUCCESS
        assert not error
        if module:
            try:
                assert module.contents.num_code_buffers == 0
                assert module.contents.num_call_symbols == 0
            finally:
                instance.vtable.free_module(module)
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("extent", [64, 256])
def test_native_provider_discovery_scores_without_worker_count(
    monkeypatch, tmp_path, extent
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32", block_threads=extent)
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
        text = _translate_catalog_or_skip(
            _provide_impls(instance, request), tmp_path / "ranked.tileirbc"
        )
        scores = {}
        for threads in (32, 64, 128, 256):
            if threads > extent:
                continue
            symbol = _scan_impl_symbol(
                extent // threads, "f32", block_threads=threads
            )
            row = next(
                line for line in text.splitlines() if f"@{symbol}" in line
            )
            match = re.search(r"provider_score\s*=\s*(\d+)", row)
            assert match, row
            scores[threads] = int(match[1])
            assert scores[threads] == 100 + _provider_topology_score(threads)
        assert sorted(scores, key=scores.get, reverse=True) == (
            [128, 64, 32, 256] if extent == 256 else [64, 32]
        )
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_scan_catalog_rows_constrain_their_concrete_specialization(
    monkeypatch, tmp_path
):
    # Regression: a 512-element scan fragment was previously monomorphized for
    # the 128-element declaration specialization and failed TileAA verification.
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        for items_per_thread in (1, 4):
            callsite, keepalive = _scan_callsite(items_per_thread, "i32")
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
            text = _translate_catalog_or_skip(
                catalog,
                tmp_path / f"scan-{items_per_thread}.tileirbc",
            )

            extent = 128 * items_per_thread
            impl_header = text.split(" implements ", maxsplit=1)[0]
            assert impl_header.count("?") == 8
            assert (
                f"implements @cuda_coop_tile_scan<[{extent}, i32, "
                '"exclusive", "raking", "sum", 0, 0, '
                '"i32"]>'
            ) in text
            assert f"preview$fragment<{items_per_thread}xi32" in text
            stride = 0 if items_per_thread == 1 else 1
            assert (
                f"fragmap = <(1, 128, {items_per_thread}):"
                f"(0, {items_per_thread}, {stride})>"
            ) in text

            resolved_call, resolved_keepalive = _resolved_scan_call(
                items_per_thread, "i32"
            )
            exact_impl = _scan_impl_symbol(items_per_thread, "i32").encode()
            resolved_call.impl.contents.symbol = _string_view(exact_impl)
            candidates = (TileIRLPFilterCandidate * 1)(
                TileIRLPFilterCandidate(
                    17,
                    0,
                    TILEIR_LP_CANDIDATE_PROVIDER,
                    101,
                    resolved_call,
                )
            )
            filter_request = TileIRLPFilterAndRankRequest(
                TILEIR_LP_REQUEST_FORMAT_VERSION,
                _compilation_context(),
                candidates,
                len(candidates),
            )
            ranks = (TileIRLPRankResult * 1)()
            error = TileIRLPErrorPtr()
            assert (
                instance.vtable.filter_and_rank(
                    instance.handle,
                    ctypes.byref(filter_request),
                    ranks,
                    len(ranks),
                    ctypes.byref(error),
                )
                == TILEIR_LP_SUCCESS
            )
            assert not error
            assert ranks[0].disposition == TILEIR_LP_RANK_SCORED
            assert keepalive and resolved_keepalive and exact_impl
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "case", CANONICAL_FAMILY_CASES, ids=lambda case: case.name
)
@pytest.mark.parametrize("filter_before_instantiate", [False, True])
def test_native_provider_canonical_family_late_callbacks(
    monkeypatch, case, filter_before_instantiate
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, callsite_keepalive = case.build_callsite()
        callsites = (TileIRLPCallsite * 1)(callsite)
        context = _compilation_context()
        if not filter_before_instantiate:
            context.num_warps_per_cta = -1
        provide_request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            context,
            None,
            0,
        )
        catalog = _provide_impls(instance, provide_request)
        assert catalog.startswith(b"\x7fTileIR\x00")

        resolved_call, resolved_keepalive = case.build_resolved_call()
        selected_impl = _string_view_value(resolved_call.impl.contents.symbol)
        selected_decl = _string_view_value(resolved_call.decl.contents.symbol)
        assert selected_impl.endswith(DEFAULT_LATE_CALLBACK_SUFFIX)
        assert selected_decl.endswith(DEFAULT_LATE_CALLBACK_SUFFIX)
        impl_symbol = selected_impl.removesuffix(DEFAULT_LATE_CALLBACK_SUFFIX)
        decl_symbol = selected_decl.removesuffix(DEFAULT_LATE_CALLBACK_SUFFIX)
        assert impl_symbol.startswith(f"{decl_symbol}_impl__lpv4_")
        assert impl_symbol.encode() in catalog

        candidates = (TileIRLPFilterCandidate * 1)(
            TileIRLPFilterCandidate(
                17,
                0,
                TILEIR_LP_CANDIDATE_PROVIDER,
                101,
                resolved_call,
            )
        )
        filter_request = TileIRLPFilterAndRankRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            _compilation_context(),
            candidates,
            len(candidates),
        )
        ranks = (TileIRLPRankResult * 1)()
        error = TileIRLPErrorPtr()
        if filter_before_instantiate:
            assert (
                instance.vtable.filter_and_rank(
                    instance.handle,
                    ctypes.byref(filter_request),
                    ranks,
                    len(ranks),
                    ctypes.byref(error),
                )
                == TILEIR_LP_SUCCESS
            )
            assert not error
            assert ranks[0].disposition == TILEIR_LP_RANK_SCORED
            assert ranks[0].score == candidates[0].baseline_score

        calls = (TileIRResolvedCall * 1)(resolved_call)
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
            assert code_buffer.output_type == TILEIR_OUTPUT_LTO_IR
            assert code_buffer.data and code_buffer.size > 0
            assert code_buffer.num_symbols == 1
            assert _string_view_value(code_buffer.symbols[0]) == selected_impl
            assert module.num_call_symbols == 1
            assert module.call_symbols[0].call_index == 0
            assert (
                _string_view_value(module.call_symbols[0].symbol)
                == selected_impl
            )
        finally:
            instance.vtable.free_module(module_ptr)

        assert callsite_keepalive and resolved_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "family,extent,batches,dtype,algorithm,warps,rows_per_invocation",
    [
        ("sum", 1024, 8, "f64", "warp_reductions", 4, 1),
        ("reduce", 64, 8, "f64", "warp_reductions", 8, 2),
        ("sum", 128, 8, "f64", "warp_reductions", 16, 2),
        ("sum", 16, 8, "f64", "warp_reductions", 16, 2),
        ("sum", 128, 8, "f32", "warp_reductions", 16, 1),
        ("sum", 128, 8, "i32", "warp_reductions", 16, 1),
        ("sum", 8, 8, "f64", "warp_reductions", 4, 1),
        ("reduce", 24, 3, "f64", "warp_reductions", 4, 1),
        ("sum", 128, 1, "f64", "warp_reductions", 16, 1),
        ("reduce", 8, 1, "f64", "warp_reductions", 8, 1),
        ("reduce", 1024, 8, "f64", "raking", 16, 1),
    ],
)
def test_batched_reduction_catalog_maps_rows_and_partial_inputs(
    monkeypatch,
    tmp_path,
    family,
    extent,
    batches,
    dtype,
    algorithm,
    warps,
    rows_per_invocation,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _batched_reduce_callsite(
            family, extent, batches, dtype, algorithm
        )
        callsites = (TileIRLPCallsite * 1)(callsite)
        context = _compilation_context()
        context.num_warps_per_cta = warps
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            context,
            None,
            0,
        )
        text = _translate_catalog_or_skip(
            _provide_impls(instance, request),
            tmp_path / "reduce-batched.tileirbc",
        )
        assert f"implements @cuda_coop_tile_{family}_batched" in text
        participants = 32 if algorithm == "warp_reductions" else warps * 32
        row_extent = extent // batches
        valid = min(row_extent, participants)
        items = row_extent // valid
        if rows_per_invocation == 2:
            assert (
                f"fragmap = <({batches // 2}, ({valid}, {32 // valid}), "
                f"({items}, 2)):"
                f"({row_extent * 2}, ({items}, 0), "
                f"({int(items > 1)}, {row_extent}))>"
            ) in text
            assert (
                f"fragmap = <({batches // 2}, (2, 16), 1):(2, (1, 0), 0)>"
            ) in text
        else:
            assert (
                f"fragmap = <({batches}, ({valid}, {participants // valid}), "
                f"{items}):"
                f"({row_extent}, ({items}, 0), {int(items > 1)})>"
            ) in text
            assert (
                f"fragmap = <({batches}, ({valid}, "
                f"{participants // valid}), 1):"
                "(1, (0, 0), 0)>"
            ) in text
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)
