# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

import pytest

from .harness import (
    SCAN_DECL,
    TILEIR_LP_CANDIDATE_PROVIDER,
    TILEIR_LP_ERROR,
    TILEIR_LP_ERROR_INVALID_ARGUMENT,
    TILEIR_LP_ERROR_KIND_GENERIC,
    TILEIR_LP_ERROR_UNRESOLVED_CALLS,
    TILEIR_LP_RANK_FILTERED,
    TILEIR_LP_RANK_SCORED,
    TILEIR_LP_REQUEST_FORMAT_VERSION,
    TILEIR_LP_SUCCESS,
    TileIRForeignModule,
    TileIRLPCallsite,
    TileIRLPCompilationContext,
    TileIRLPErrorPtr,
    TileIRLPFilterAndRankRequest,
    TileIRLPFilterCandidate,
    TileIRLPProvideImplsRequest,
    TileIRLPRankResult,
    TileIRResolvedCall,
    TileIRStringView,
    _compilation_context,
    _copy_and_free_lp_error,
    _create_provider,
    _load_provider_or_skip,
    _provide_impls,
    _provide_impls_result,
    _string_view,
    _string_view_value,
)
from .reduce_scan import _resolved_scan_call, _scan_callsite, _scan_impl_symbol


def _scan_discovery_request(architecture):
    callsite, keepalive = _scan_callsite(1, "f32")
    callsites = (TileIRLPCallsite * 1)(callsite)
    context = (
        _compilation_context(architecture)
        if isinstance(architecture, str)
        else architecture
    )
    request = TileIRLPProvideImplsRequest(
        TILEIR_LP_REQUEST_FORMAT_VERSION,
        callsites,
        len(callsites),
        context,
        None,
        0,
    )
    return request, (callsites, context, keepalive)


def _instantiate_result(instance, calls):
    call_array = (TileIRResolvedCall * len(calls))(*calls)
    module = ctypes.POINTER(TileIRForeignModule)()
    error = TileIRLPErrorPtr()
    result = instance.vtable.instantiate(
        instance.handle,
        call_array,
        len(call_array),
        ctypes.byref(module),
        ctypes.byref(error),
    )
    return result, module, _copy_and_free_lp_error(instance, error)


@pytest.mark.parametrize(
    ("spellings", "canonical"),
    [
        pytest.param(("90", "sm_090", "compute_90"), "sm_90", id="generic"),
        pytest.param(
            ("90a", "sm_090a", "compute_90a"),
            "sm_90a",
            id="accelerated-family",
        ),
    ],
)
def test_native_provider_canonicalizes_target_spellings_and_resource_cache(
    monkeypatch,
    tmp_path,
    spellings,
    canonical,
):
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        keepalives = []
        for spelling in spellings:
            request, keepalive = _scan_discovery_request(spelling)
            keepalives.append(keepalive)
            catalog = _provide_impls(instance, request)
            assert catalog.startswith(b"\x7fTileIR\x00")
            assert canonical.encode() in catalog

        trace_lines = trace_file.read_text().splitlines()
        compile_lines = [
            line
            for line in trace_lines
            if "resource_query_nvrtc_compile" in line
        ]
        cache_lines = [
            line for line in trace_lines if "resource_query_cache_hit" in line
        ]
        assert len(compile_lines) == 1
        assert len(cache_lines) == len(spellings) - 1
        assert all(
            f"sm={canonical}" in line for line in compile_lines + cache_lines
        )
        assert keepalives
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_missing_and_malformed_target_spellings(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        malformed_contexts = [
            _compilation_context(architecture)
            for architecture in (
                "",
                "sm_",
                "SM_90",
                "sm_90A",
                " sm_90",
                "sm_90 ",
                "sm_0",
                "sm_90x",
                "sm_999999999999999999999999999999",
            )
        ]
        for context in malformed_contexts:
            request, keepalive = _scan_discovery_request(context)
            result, catalog, error = _provide_impls_result(instance, request)
            assert result == TILEIR_LP_ERROR
            assert catalog == b""
            assert error is not None
            assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
            diagnostic = f"{error.message}\n{error.details}"
            assert SCAN_DECL in diagnostic
            assert "[LP-TARGET-001]" in diagnostic
            assert keepalive

        invalid_view_context = TileIRLPCompilationContext(
            TileIRStringView(None, 5), 4, -1, None, 0
        )
        request, keepalive = _scan_discovery_request(invalid_view_context)
        result, catalog, error = _provide_impls_result(instance, request)
        assert result == TILEIR_LP_ERROR_INVALID_ARGUMENT
        assert catalog == b""
        assert error is None
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_target_environment_precedence_and_context_override(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        monkeypatch.setenv("CUDA_ARCH", "compute_080")
        request, secondary_keepalive = _scan_discovery_request("")
        assert b"sm_80" in _provide_impls(instance, request)

        monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_SM_ARCH", "sm_090a")
        request, primary_keepalive = _scan_discovery_request("")
        assert b"sm_90a" in _provide_impls(instance, request)

        monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_SM_ARCH", "SM_90")
        request, invalid_primary_keepalive = _scan_discovery_request("")
        result, catalog, error = _provide_impls_result(instance, request)
        assert result == TILEIR_LP_ERROR
        assert catalog == b""
        assert error is not None
        assert "[LP-TARGET-001]" in error.details
        assert "CUDA_COOP_TILE_PROVIDER_SM_ARCH" in error.details

        request, context_keepalive = _scan_discovery_request("compute_080")
        assert b"sm_80" in _provide_impls(instance, request)
        assert (
            secondary_keepalive
            and primary_keepalive
            and invalid_primary_keepalive
            and context_keepalive
        )
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_malformed_target_precedes_unsupported_specialization(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(
            1,
            "f32",
            mode="future_mode",
        )
        callsites = (TileIRLPCallsite * 1)(callsite)
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            _compilation_context("SM_90"),
            None,
            0,
        )

        result, catalog, error = _provide_impls_result(instance, request)

        assert result == TILEIR_LP_ERROR
        assert catalog == b""
        assert error is not None
        assert "[LP-TARGET-001]" in error.details
        assert "[LP-UNSUPPORTED-001]" not in error.details
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "target_spelling", ("100f", "sm_0100f", "compute_100f")
)
def test_native_provider_filters_exact_f_family_constraint_without_nvrtc(
    monkeypatch,
    target_spelling,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        exact_call, exact_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture=target_spelling,
        )
        mismatched_call, mismatched_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture=target_spelling,
        )
        generic_architecture = b"sm_100"
        context = mismatched_call.impl.contents.execution_context_constraints
        context.architecture = _string_view(generic_architecture)
        candidates = (TileIRLPFilterCandidate * 2)(
            TileIRLPFilterCandidate(
                7, 0, TILEIR_LP_CANDIDATE_PROVIDER, 100, exact_call
            ),
            TileIRLPFilterCandidate(
                7, 1, TILEIR_LP_CANDIDATE_PROVIDER, 100, mismatched_call
            ),
        )
        request = TileIRLPFilterAndRankRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            _compilation_context("sm_100f"),
            candidates,
            len(candidates),
        )
        ranks = (TileIRLPRankResult * len(candidates))()
        error = TileIRLPErrorPtr()
        assert (
            instance.vtable.filter_and_rank(
                instance.handle,
                ctypes.byref(request),
                ranks,
                len(ranks),
                ctypes.byref(error),
            )
            == TILEIR_LP_SUCCESS
        )
        assert not error
        assert ranks[0].disposition == TILEIR_LP_RANK_SCORED
        assert ranks[1].disposition == TILEIR_LP_RANK_FILTERED
        assert exact_keepalive and mismatched_keepalive and generic_architecture
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_mixed_architecture_batch_fails_before_nvrtc(
    monkeypatch,
    tmp_path,
):
    compile_counter = tmp_path / "nvrtc-compiles.txt"
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE",
        str(compile_counter),
    )
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        generic_call, generic_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture="sm_90",
        )
        accelerated_call, accelerated_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture="sm_90a",
        )
        result, module, error = _instantiate_result(
            instance, [generic_call, accelerated_call]
        )
        assert result == TILEIR_LP_ERROR
        assert not module
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "mixed target architectures" in error.message
        assert not compile_counter.exists()
        trace = trace_file.read_text() if trace_file.exists() else ""
        for work in (
            "cache_miss",
            "cache_hit",
            "resource_query",
            "nvrtc_",
            "bundle_cache",
        ):
            assert work not in trace
        assert generic_keepalive and accelerated_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_provider_accepts_generic_discovery_hash_for_late_family_refinement(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        discovery, discovery_keepalive = _scan_discovery_request("sm_90")
        stale_impl = _scan_impl_symbol(1, "f32", architecture="sm_90")
        assert stale_impl.encode() in _provide_impls(instance, discovery)

        call, call_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture="sm_90a",
            impl_symbol=stale_impl,
        )
        candidates = (TileIRLPFilterCandidate * 1)(
            TileIRLPFilterCandidate(
                7, 0, TILEIR_LP_CANDIDATE_PROVIDER, 100, call
            )
        )
        request = TileIRLPFilterAndRankRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            _compilation_context("sm_90a"),
            candidates,
            len(candidates),
        )
        ranks = (TileIRLPRankResult * 1)()
        error = TileIRLPErrorPtr()

        assert (
            instance.vtable.filter_and_rank(
                instance.handle,
                ctypes.byref(request),
                ranks,
                len(ranks),
                ctypes.byref(error),
            )
            == TILEIR_LP_SUCCESS
        )
        assert not error
        assert ranks[0].disposition == TILEIR_LP_RANK_SCORED

        result, module, captured = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_SUCCESS
        assert captured is None
        assert module
        instance.vtable.free_module(module)
        assert discovery_keepalive and call_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    ("discovery_architecture", "resolved_architecture"),
    [
        ("sm_90a", "sm_90"),
        ("sm_90a", "sm_90f"),
        ("sm_90", "sm_100a"),
    ],
)
def test_native_provider_rejects_non_refining_target_identity_changes(
    monkeypatch,
    discovery_architecture,
    resolved_architecture,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        selected_impl = _scan_impl_symbol(
            1, "f32", architecture=discovery_architecture
        )
        call, keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture=resolved_architecture,
            impl_symbol=selected_impl,
        )

        result, module, captured = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert not module
        assert captured is not None
        assert "selected implementation identity" in (
            captured.unresolved_calls[0].reason
        )
        assert selected_impl in captured.unresolved_calls[0].reason
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_architecture_family_separates_bundles_and_nvrtc_target(
    monkeypatch,
    tmp_path,
):
    compile_counter = tmp_path / "nvrtc-compiles.txt"
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE",
        str(compile_counter),
    )
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        generic_call, generic_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture="sm_90",
        )
        accelerated_call, accelerated_keepalive = _resolved_scan_call(
            1,
            "f32",
            architecture="sm_90a",
        )
        for call in (
            generic_call,
            accelerated_call,
            generic_call,
            accelerated_call,
        ):
            result, module, error = _instantiate_result(instance, [call])
            assert result == TILEIR_LP_SUCCESS
            assert error is None
            assert module
            try:
                expected_symbol = _string_view_value(call.impl.contents.symbol)
                assert module.contents.num_code_buffers == 1
                assert module.contents.code_buffers[0].num_symbols == 1
                assert (
                    _string_view_value(
                        module.contents.code_buffers[0].symbols[0]
                    )
                    == expected_symbol
                )
                assert module.contents.num_call_symbols == 1
                assert (
                    _string_view_value(module.contents.call_symbols[0].symbol)
                    == expected_symbol
                )
            finally:
                instance.vtable.free_module(module)

        assert compile_counter.read_text().splitlines() == ["1", "1"]
        trace = trace_file.read_text()
        assert "cache_miss" in trace and "sm=sm_90 " in trace
        assert "cache_miss" in trace and "sm=sm_90a " in trace
        assert "bundle_cache_hit" in trace
        assert (
            "nvrtc_resource_compile unit=coop_resource_query.cu "
            "option=--gpu-architecture=compute_90a"
        ) in trace
        assert (
            "nvrtc_lto_compile unit=coop_bundle.cu "
            "option=--gpu-architecture=compute_90a"
        ) in trace
        assert generic_keepalive and accelerated_keepalive
    finally:
        instance.vtable.destroy(instance.handle)
