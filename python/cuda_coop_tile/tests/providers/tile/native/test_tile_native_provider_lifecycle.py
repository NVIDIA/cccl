# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    TILEIR_LP_CANDIDATE_PROVIDER,
    TILEIR_LP_ERROR_INVALID_ARGUMENT,
    TILEIR_LP_ERROR_UNSUPPORTED_VERSION,
    TILEIR_LP_RANK_FILTERED,
    TILEIR_LP_RANK_SCORED,
    TILEIR_LP_REQUEST_FORMAT_VERSION,
    TILEIR_LP_SUCCESS,
    TileIRForeignModule,
    TileIRLinkageProviderAPIVersion,
    TileIRLinkageProviderInstance,
    TileIRLPCallsite,
    TileIRLPCompilationContext,
    TileIRLPErrorPtr,
    TileIRLPFilterAndRankRequest,
    TileIRLPFilterCandidate,
    TileIRLPProvideImplsRequest,
    TileIRLPRankResult,
    _assert_soft_miss,
    _compilation_context,
    _create_provider,
    _load_provider_or_skip,
    _provide_impls_result,
    _provider_topology_score,
    _string_view,
)
from .reduce_scan import _resolved_scan_call, _scan_callsite, _scan_impl_symbol


def test_native_provider_negotiates_lpv4_and_clears_failure_outputs(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        request = TileIRLPProvideImplsRequest(
            99, None, 0, _compilation_context(), None, 0
        )
        data = ctypes.c_void_p(1)
        size = ctypes.c_size_t(17)
        error = ctypes.cast(ctypes.c_void_p(1), TileIRLPErrorPtr)
        assert (
            instance.vtable.provide_impls(
                instance.handle,
                ctypes.byref(request),
                ctypes.byref(data),
                ctypes.byref(size),
                ctypes.byref(error),
            )
            == TILEIR_LP_ERROR_INVALID_ARGUMENT
        )
        assert not data
        assert size.value == 0
        assert not error

        module = ctypes.pointer(TileIRForeignModule())
        error = ctypes.cast(ctypes.c_void_p(1), TileIRLPErrorPtr)
        assert (
            instance.vtable.instantiate(
                None, None, 0, ctypes.byref(module), ctypes.byref(error)
            )
            == TILEIR_LP_ERROR_INVALID_ARGUMENT
        )
        assert not module
        assert not error

        poisoned = TileIRLinkageProviderInstance()
        ctypes.memset(ctypes.byref(poisoned), 0xA5, ctypes.sizeof(poisoned))
        assert (
            lib.tileir_create_linkage_provider(
                TileIRLinkageProviderAPIVersion(0, 0, 2),
                ctypes.byref(poisoned),
            )
            == TILEIR_LP_ERROR_UNSUPPORTED_VERSION
        )
        assert bytes(poisoned) == bytes(ctypes.sizeof(poisoned))
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_request_format_one_requires_exactly_one_callsite(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        for callsites, count, adapter_reports_error in (
            ((TileIRLPCallsite * 1)(callsite), 0, False),
            ((TileIRLPCallsite * 2)(callsite, callsite), 2, True),
        ):
            request = TileIRLPProvideImplsRequest(
                TILEIR_LP_REQUEST_FORMAT_VERSION,
                callsites,
                count,
                _compilation_context(),
                None,
                0,
            )

            result, catalog, error = _provide_impls_result(instance, request)

            assert result == TILEIR_LP_ERROR_INVALID_ARGUMENT
            assert catalog == b""
            if adapter_reports_error:
                assert error is not None
                assert error.code == TILEIR_LP_ERROR_INVALID_ARGUMENT
                assert "exactly one callsite" in error.message
            else:
                # LPv4 rejects the zero-call request in its ABI adapter before
                # entering the provider.  The source contract separately proves
                # that the provider body also keeps its no-work early return.
                assert error is None
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_optional_ranking_preserves_catalog_scores(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        topologies = ((32, 8), (256, 1), (64, 4), (128, 2))
        calls = []
        keepalives = []
        symbol_storage = []
        for block_threads, items_per_thread in topologies:
            call, keepalive = _resolved_scan_call(2, "f32")
            call.resolved_execution_context.num_warps_per_cta = (
                block_threads // 32
            )
            impl_symbol = (
                _scan_impl_symbol(
                    items_per_thread,
                    "f32",
                    block_threads=block_threads,
                )
                + DEFAULT_LATE_CALLBACK_SUFFIX
            ).encode()
            symbol_storage.append(impl_symbol)
            call.impl.contents.symbol = _string_view(impl_symbol)
            context = call.impl.contents.execution_context_constraints
            context.num_warps_per_cta = block_threads // 32
            calls.append(call)
            keepalives.append(keepalive)

        candidates = (TileIRLPFilterCandidate * len(calls))(
            *(
                TileIRLPFilterCandidate(
                    7,
                    index,
                    TILEIR_LP_CANDIDATE_PROVIDER,
                    100 + _provider_topology_score(topologies[index][0]),
                    call,
                )
                for index, call in enumerate(calls)
            )
        )
        request = TileIRLPFilterAndRankRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            TileIRLPCompilationContext(_string_view("sm_80"), -1, -1, None, 0),
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
        assert [rank.disposition for rank in ranks] == [
            TILEIR_LP_RANK_SCORED
        ] * len(ranks)
        assert [rank.score for rank in ranks] == [
            100 + _provider_topology_score(block_threads)
            for block_threads, _ in topologies
        ]
        assert [
            block_threads
            for (block_threads, _), _rank in sorted(
                zip(topologies, ranks),
                key=lambda pair: pair[1].score,
                reverse=True,
            )
        ] == [128, 64, 32, 256]
        assert keepalives and symbol_storage
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_topologies_above_1024_at_discovery_and_ranking(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, callsite_keepalive = _scan_callsite(1, "f32")
        callsite_keepalive[0][0].value = 2048
        callsite.const_arguments[0].int_val = 2048
        callsites = (TileIRLPCallsite * 1)(callsite)
        discovery_request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            TileIRLPCompilationContext(_string_view("sm_80"), 64, -1, None, 0),
            None,
            0,
        )
        _assert_soft_miss(instance, discovery_request)

        call, call_keepalive = _resolved_scan_call(1, "f32")
        call_keepalive[0][0].value = 2048
        call.const_arguments[0].int_val = 2048
        call.resolved_execution_context.num_warps_per_cta = 64
        candidates = (TileIRLPFilterCandidate * 1)(
            TileIRLPFilterCandidate(
                7, 0, TILEIR_LP_CANDIDATE_PROVIDER, 100, call
            )
        )
        rank_request = TileIRLPFilterAndRankRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            TileIRLPCompilationContext(_string_view("sm_80"), 64, -1, None, 0),
            candidates,
            len(candidates),
        )
        ranks = (TileIRLPRankResult * 1)()
        error = TileIRLPErrorPtr()
        assert (
            instance.vtable.filter_and_rank(
                instance.handle,
                ctypes.byref(rank_request),
                ranks,
                len(ranks),
                ctypes.byref(error),
            )
            == TILEIR_LP_SUCCESS
        )
        assert not error
        assert (ranks[0].disposition, ranks[0].score) == (
            TILEIR_LP_RANK_FILTERED,
            0,
        )
        assert callsite_keepalive and call_keepalive
    finally:
        instance.vtable.destroy(instance.handle)
