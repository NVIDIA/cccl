# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import shutil

import pytest

from .adjacent_shuffle_exchange import (
    _exchange_callsite,
    _resolved_exchange_call,
)
from .canonical_cases import CANONICAL_FAMILY_CASES
from .discontinuity import _discontinuity_callsite
from .harness import (
    CCCL_ROOT,
    DEFAULT_LATE_CALLBACK_SUFFIX,
    TILEIR_EXEC_SCOPE_CTA,
    TILEIR_LP_CANDIDATE_PROVIDER,
    TILEIR_LP_ERROR,
    TILEIR_LP_ERROR_KIND_GENERIC,
    TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS,
    TILEIR_LP_ERROR_UNRESOLVED_CALLS,
    TILEIR_LP_RANK_FILTERED,
    TILEIR_LP_RANK_SCORED,
    TILEIR_LP_REQUEST_FORMAT_VERSION,
    TILEIR_LP_SUCCESS,
    TileIRForeignModule,
    TileIRLPCallsite,
    TileIRLPErrorPtr,
    TileIRLPFilterAndRankRequest,
    TileIRLPFilterCandidate,
    TileIRLPProvideImplsRequest,
    TileIRLPRankResult,
    TileIRResolvedCall,
    TileIRStringView,
    _artifact_identity,
    _assert_malformed,
    _assert_soft_miss,
    _compilation_context,
    _copy_and_free_lp_error,
    _create_provider,
    _fnv1a,
    _load_provider_or_skip,
    _provide_impls,
    _string_view,
    _string_view_value,
)
from .histogram_rld import _resolved_histogram_call
from .radix import _radix_sort_callsite, _resolved_radix_sort_call
from .reduce_scan import (
    _batched_reduce_callsite,
    _resolved_reduce_call,
    _resolved_scan_call,
    _scan_callsite,
)
from .topk_sort import (
    _merge_sort_callsite,
    _merge_sort_semantic_key,
    _resolved_merge_sort_call,
)


def _provide_request(callsite: TileIRLPCallsite):
    callsites = (TileIRLPCallsite * 1)(callsite)
    request = TileIRLPProvideImplsRequest(
        TILEIR_LP_REQUEST_FORMAT_VERSION,
        callsites,
        len(callsites),
        _compilation_context(),
        None,
        0,
    )
    return request, callsites


def _filter_result(instance, resolved_call: TileIRResolvedCall):
    candidates = (TileIRLPFilterCandidate * 1)(
        TileIRLPFilterCandidate(
            7,
            0,
            TILEIR_LP_CANDIDATE_PROVIDER,
            41,
            resolved_call,
        )
    )
    request = TileIRLPFilterAndRankRequest(
        TILEIR_LP_REQUEST_FORMAT_VERSION,
        _compilation_context(),
        candidates,
        len(candidates),
    )
    results = (TileIRLPRankResult * 1)()
    error = TileIRLPErrorPtr()
    result = instance.vtable.filter_and_rank(
        instance.handle,
        ctypes.byref(request),
        results,
        len(results),
        ctypes.byref(error),
    )
    return result, results[0], _copy_and_free_lp_error(instance, error)


def _instantiate_result(instance, resolved_calls):
    calls = (TileIRResolvedCall * len(resolved_calls))(*resolved_calls)
    module = ctypes.POINTER(TileIRForeignModule)()
    error = TileIRLPErrorPtr()
    result = instance.vtable.instantiate(
        instance.handle,
        calls,
        len(calls),
        ctypes.byref(module),
        ctypes.byref(error),
    )
    captured_error = _copy_and_free_lp_error(instance, error)
    if module:
        instance.vtable.free_module(module)
    return result, captured_error


@pytest.mark.parametrize(
    "malformed", [False, True], ids=["valid", "wrong-schema"]
)
def test_native_provider_binds_calls_without_divisible_candidates(
    monkeypatch, tmp_path, malformed
):
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        # No offered width divides 33. Binding must still inspect the ABI.
        keepalive[0][0].value = 33
        callsite.const_arguments[0].int_val = 33
        wrong_name = b"WrongN"
        if malformed:
            callsite.decl.contents.const_params[0].name = _string_view(
                wrong_name
            )
        request, request_keepalive = _provide_request(callsite)
        request.compilation_context.num_warps_per_cta = -1

        if malformed:
            _assert_malformed(instance, request, "[LP-SCHEMA-001]")
        else:
            _assert_soft_miss(instance, request)

        trace = trace_file.read_text() if trace_file.exists() else ""
        assert "resource_query" not in trace
        assert "nvrtc_" not in trace
        assert keepalive and request_keepalive and wrong_name
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_merge_sort_rejects_non_power_of_two_workers(
    monkeypatch, tmp_path
):
    trace_file = tmp_path / "provider-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, discovery_keepalive = _merge_sort_callsite(
            "keys", 1, "i32", None, False, False
        )
        discovery_keepalive[0][0].value = 96
        callsite.const_arguments[0].int_val = 96
        request, request_keepalive = _provide_request(callsite)
        request.compilation_context.num_warps_per_cta = 3
        _assert_soft_miss(instance, request)

        call, keepalive = _resolved_merge_sort_call(
            "keys", 1, "i32", None, False, False
        )
        keepalive[0][0].value = 96
        call.const_arguments[0].int_val = 96
        call.resolved_execution_context.num_warps_per_cta = 3
        call.impl.contents.execution_context_constraints.num_warps_per_cta = 3
        result, rank, error = _filter_result(instance, call)
        assert result == TILEIR_LP_SUCCESS
        assert error is None
        assert rank.disposition == TILEIR_LP_RANK_FILTERED
        assert rank.score == 0

        result, error = _instantiate_result(instance, [call])
        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert (
            "BlockThreads must be a power of two"
            in error.unresolved_calls[0].reason
        )
        trace = trace_file.read_text() if trace_file.exists() else ""
        assert "resource_query" not in trace
        assert "nvrtc_" not in trace
        assert discovery_keepalive and request_keepalive and keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "resolved_scope", [2, TILEIR_EXEC_SCOPE_CTA], ids=["warp", "cta"]
)
def test_native_provider_merge_sort_requires_warp_scope(
    monkeypatch, resolved_scope
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, discovery_keepalive = _merge_sort_callsite(
            "keys", 1, "i32", None, False, False
        )
        discovery_keepalive[0][0].value = 32
        callsite.const_arguments[0].int_val = 32
        request, request_keepalive = _provide_request(callsite)
        catalog = _provide_impls(instance, request)
        semantic_key = _merge_sort_semantic_key(
            "keys", 1, "i32", None, False, False
        ).replace("block_threads=128;", "block_threads=32;")
        symbol = (
            "cuda_coop_tile_merge_sort_keys_impl__lpv4_"
            f"{_fnv1a(_artifact_identity(semantic_key, 'sm_80')):x}"
        )
        assert symbol.encode() in catalog

        call, keepalive = _resolved_merge_sort_call(
            "keys", 1, "i32", None, False, False, impl_symbol=symbol
        )
        keepalive[0][0].value = 32
        call.const_arguments[0].int_val = 32
        # Warp scope constrains participants, not the surrounding physical CTA.
        call.impl.contents.execution_context_constraints.scope = resolved_scope
        call.impl.contents.execution_context_constraints.num_warps_per_cta = -1
        call.resolved_execution_context.scope = resolved_scope
        call.resolved_execution_context.num_warps_per_cta = (
            4 if resolved_scope == 2 else 1
        )
        result, rank, error = _filter_result(instance, call)
        assert result == TILEIR_LP_SUCCESS, error
        assert error is None
        if resolved_scope == TILEIR_EXEC_SCOPE_CTA:
            assert rank.disposition == TILEIR_LP_RANK_FILTERED
            result, error = _instantiate_result(instance, [call])
            assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
            assert error is not None
            assert "execution scope" in error.unresolved_calls[0].reason
            return
        assert rank.disposition == TILEIR_LP_RANK_SCORED

        calls = (TileIRResolvedCall * 1)(call)
        module = ctypes.POINTER(TileIRForeignModule)()
        error = TileIRLPErrorPtr()
        result = instance.vtable.instantiate(
            instance.handle,
            calls,
            len(calls),
            ctypes.byref(module),
            ctypes.byref(error),
        )
        captured_error = _copy_and_free_lp_error(instance, error)
        try:
            assert result == TILEIR_LP_SUCCESS, captured_error
            assert captured_error is None
            assert module
            assert module.contents.num_code_buffers == 1
            assert module.contents.num_call_symbols == 1
            assert _string_view_value(
                module.contents.call_symbols[0].symbol
            ) == _string_view_value(call.impl.contents.symbol)
        finally:
            if module:
                instance.vtable.free_module(module)
        assert discovery_keepalive and request_keepalive and keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "width,extent,warps",
    [(64, 256, 4), (16, 256, -1), (2048, 2048, -1)],
    ids=["contradictory-workers", "partial-warp", "too-wide"],
)
def test_native_provider_exchange_diagnoses_invalid_explicit_width(
    monkeypatch, width, extent, warps
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _exchange_callsite(
            "blocked_to_striped", num_items=extent, block_threads=width
        )
        request, request_keepalive = _provide_request(callsite)
        request.compilation_context.num_warps_per_cta = warps
        error = _assert_malformed(instance, request, "[LP-TOPOLOGY-001]")
        assert "BlockThreads" in error.details
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_owned_declaration_reports_missing_headers(
    monkeypatch, tmp_path
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    bundle = tmp_path / "incomplete-headers"
    (bundle / "cub").mkdir(parents=True)
    (bundle / "cub/version.cuh").touch()
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(bundle))
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        request, request_keepalive = _provide_request(callsite)
        error = _assert_malformed(instance, request, "[LP-HEADERS-001]")
        assert "cub/block/block_scan.cuh" in error.details
        assert str(bundle) in error.details
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_reports_missing_batched_reduction_header(
    monkeypatch, tmp_path
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    incomplete = tmp_path / "incomplete-cccl"
    shutil.copytree(
        CCCL_ROOT / "cub/cub",
        incomplete / "cub/cub",
        ignore=shutil.ignore_patterns("warp_reduce_batched.cuh"),
    )
    (incomplete / "libcudacxx").mkdir()
    (incomplete / "libcudacxx/include").symlink_to(
        CCCL_ROOT / "libcudacxx/include", target_is_directory=True
    )
    (incomplete / "thrust").symlink_to(
        CCCL_ROOT / "thrust", target_is_directory=True
    )
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(incomplete))
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _batched_reduce_callsite("sum", 8, 2, "f64")
        request, request_keepalive = _provide_request(callsite)
        error = _assert_malformed(
            instance, request, "cub/warp/warp_reduce_batched.cuh"
        )
        assert "cannot open source file" in error.details
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("family", ["keys", "pairs"])
def test_native_provider_rejects_radix_sort_capacity_before_compilation(
    monkeypatch, tmp_path, family
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    trace_file = tmp_path / "radix-capacity-trace.txt"
    monkeypatch.setenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", str(trace_file))
    value_dtype = "i32" if family == "pairs" else None
    instance = _create_provider(lib)
    try:
        callsite, discovery_keepalive = _radix_sort_callsite(
            family, 512, "i32", value_dtype
        )
        request, request_keepalive = _provide_request(callsite)
        error = _assert_malformed(instance, request, "[LP-TOPOLOGY-001]")
        assert "at most 65535" in error.details

        call, keepalive = _resolved_radix_sort_call(
            family, 512, "i32", value_dtype
        )
        result, _, error = _filter_result(instance, call)
        assert result == TILEIR_LP_ERROR
        assert error is not None and "at most 65535" in error.details
        result, error = _instantiate_result(instance, [call])
        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert (
            error is not None
            and "at most 65535" in error.unresolved_calls[0].reason
        )
        trace = trace_file.read_text() if trace_file.exists() else ""
        assert "nvrtc_" not in trace
        assert discovery_keepalive and request_keepalive and keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_exchange_requires_explicit_permutation_topology(
    monkeypatch,
):
    # A 256-element exchange has different permutations at 128/256 threads.
    # The compiler must not select that semantic parameter for the caller.
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _exchange_callsite(
            "blocked_to_striped", block_threads=0
        )
        request, request_keepalive = _provide_request(callsite)
        error = _assert_malformed(instance, request, "[LP-TOPOLOGY-001]")
        assert (
            "exchange requires positive BlockThreads dividing N"
            in error.details
        )
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_exchange_rejects_automatic_topology_in_late_callbacks(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_exchange_call(
            "blocked_to_striped", block_threads=0, resolved_block_threads=128
        )
        result, _, error = _filter_result(instance, call)
        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert (
            "exchange requires positive BlockThreads dividing N"
            in error.details
        )

        result, error = _instantiate_result(instance, [call])
        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert (
            "exchange requires positive BlockThreads dividing N"
            in error.unresolved_calls[0].reason
        )
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "case", CANONICAL_FAMILY_CASES, ids=lambda case: case.name
)
def test_native_provider_classifies_malformed_schema_for_every_family(
    monkeypatch, case
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = case.build_callsite()
        symbol = _string_view_value(callsite.decl.contents.symbol)
        malformed_name = b"WrongN"
        callsite.decl.contents.const_params[0].name = _string_view(
            malformed_name
        )
        request, request_keepalive = _provide_request(callsite)

        error = _assert_malformed(instance, request, "[LP-SCHEMA-001]")

        assert symbol in f"{error.message}\n{error.details}"
        assert keepalive and request_keepalive and malformed_name
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "case", CANONICAL_FAMILY_CASES, ids=lambda case: case.name
)
def test_provider_classifies_future_dtype_as_unsupported_for_every_family(
    monkeypatch, case
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = case.build_callsite()
        future_dtype = b"future_dtype"
        dtype_token = callsite.const_arguments[callsite.num_const_arguments - 1]
        dtype_token.string_val = _string_view(future_dtype)
        request, request_keepalive = _provide_request(callsite)

        _assert_soft_miss(instance, request)

        assert keepalive and request_keepalive and future_dtype
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    "case", CANONICAL_FAMILY_CASES, ids=lambda case: case.name
)
def test_native_provider_schema_error_precedes_unsupported_dtype(
    monkeypatch, case
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = case.build_callsite()
        malformed_name = b"WrongN"
        future_dtype = b"future_dtype"
        callsite.decl.contents.const_params[0].name = _string_view(
            malformed_name
        )
        callsite.const_arguments[
            callsite.num_const_arguments - 1
        ].string_val = _string_view(future_dtype)
        request, request_keepalive = _provide_request(callsite)

        error = _assert_malformed(instance, request, "[LP-SCHEMA-001]")

        assert "[LP-UNSUPPORTED-001]" not in error.details
        assert (
            keepalive and request_keepalive and malformed_name and future_dtype
        )
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_unknown_symbol_is_a_successful_empty_catalog(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        foreign_symbol = b"foreign_unregistered_sum"
        callsite.decl.contents.symbol = _string_view(foreign_symbol)
        request, request_keepalive = _provide_request(callsite)

        _assert_soft_miss(instance, request)

        assert keepalive and request_keepalive and foreign_symbol
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_discovery_does_not_claim_concretized_symbol(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        concrete_symbol = b"cuda_coop_tile_scan__N128__Tf32"
        callsite.decl.contents.symbol = _string_view(concrete_symbol)
        request, request_keepalive = _provide_request(callsite)

        _assert_soft_miss(instance, request)

        assert keepalive and request_keepalive and concrete_symbol
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_dtype_token_contradiction(monkeypatch):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        conflicting_dtype = b"i32"
        callsite.const_arguments[
            callsite.num_const_arguments - 1
        ].string_val = _string_view(conflicting_dtype)
        request, request_keepalive = _provide_request(callsite)

        error = _assert_malformed(instance, request, "[LP-DTYPE-001]")

        assert "contradicts" in error.details
        assert keepalive and request_keepalive and conflicting_dtype
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("extent", [0, -1])
def test_native_provider_rejects_nonpositive_logical_extent(
    monkeypatch, extent
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _scan_callsite(1, "f32")
        callsite.const_arguments[0].int_val = extent
        request, request_keepalive = _provide_request(callsite)

        error = _assert_malformed(instance, request, "[LP-TOPOLOGY-001]")

        assert "N must be positive" in error.details
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_symbol_mode_contradiction(monkeypatch):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, keepalive = _discontinuity_callsite(
            "flags",
            "heads_and_tails",
        )
        request, request_keepalive = _provide_request(callsite)

        error = _assert_malformed(instance, request, "[LP-SYMBOL-001]")

        assert "Mode" in error.details
        assert keepalive and request_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_ranking_distinguishes_foreign_and_incompatible_calls(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        foreign_call, foreign_keepalive = _resolved_scan_call(
            1, "f32", impl_symbol="foreign_provider_scan_impl"
        )
        result, rank, error = _filter_result(instance, foreign_call)
        assert result == TILEIR_LP_SUCCESS
        assert error is None
        assert rank.disposition == TILEIR_LP_RANK_FILTERED
        assert rank.score == 0

        noncurrent_symbol = (
            "cuda_coop_tile_scan_impl__foreign_protocol_deadbeef"
        )
        noncurrent_call, noncurrent_keepalive = _resolved_scan_call(
            1, "f32", impl_symbol=noncurrent_symbol
        )
        result, rank, error = _filter_result(instance, noncurrent_call)
        assert result == TILEIR_LP_SUCCESS
        assert error is None
        assert rank.disposition == TILEIR_LP_RANK_FILTERED
        assert rank.score == 0

        result, error = _instantiate_result(instance, [noncurrent_call])
        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-SYMBOL-001]" in error.unresolved_calls[0].reason
        assert "not registered" in error.unresolved_calls[0].reason

        incompatible_call, incompatible_keepalive = _resolved_scan_call(
            1, "f32"
        )
        incompatible_target = b"sm_90"
        context = incompatible_call.impl.contents.execution_context_constraints
        context.architecture = _string_view(incompatible_target)
        result, rank, error = _filter_result(instance, incompatible_call)
        assert result == TILEIR_LP_SUCCESS
        assert error is None
        assert rank.disposition == TILEIR_LP_RANK_FILTERED
        assert rank.score == 0
        assert (
            foreign_keepalive
            and noncurrent_keepalive
            and incompatible_keepalive
            and incompatible_target
        )
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("constraint", (b"SM_80", b"sm_0"))
def test_native_provider_rejects_malformed_nonempty_impl_target_constraint(
    monkeypatch, constraint
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        call.impl.contents.execution_context_constraints.architecture = (
            _string_view(constraint)
        )

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "[LP-TARGET-001]" in f"{error.message}\n{error.details}"

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-TARGET-001]" in error.unresolved_calls[0].reason
        assert keepalive and constraint
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_nonempty_null_impl_target_view(monkeypatch):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        call.impl.contents.execution_context_constraints.architecture = (
            TileIRStringView(None, 5)
        )

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "[LP-TARGET-001]" in f"{error.message}\n{error.details}"

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-TARGET-001]" in error.unresolved_calls[0].reason
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_ranking_filters_incompatible_implementation_context(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        callsite, discovery_keepalive = _scan_callsite(2, "f32")
        callsites = (TileIRLPCallsite * 1)(callsite)
        discovery_context = _compilation_context()
        discovery_context.num_warps_per_cta = -1
        request = TileIRLPProvideImplsRequest(
            TILEIR_LP_REQUEST_FORMAT_VERSION,
            callsites,
            len(callsites),
            discovery_context,
            None,
            0,
        )
        assert _provide_impls(instance, request)

        call, keepalive = _resolved_scan_call(2, "f32")
        call.impl.contents.execution_context_constraints.num_warps_per_cta = 2

        result, rank, error = _filter_result(instance, call)

        assert result == TILEIR_LP_SUCCESS
        assert error is None
        assert rank.disposition == TILEIR_LP_RANK_FILTERED
        assert rank.score == 0
        assert discovery_keepalive and discovery_context and keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_provider_instantiation_rejects_incompatible_implementation_context(
    monkeypatch, tmp_path
):
    # Regression: ranking rejected these constraints, but instantiation compiled
    # the same selected calls without checking their implementation context.
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
        valid, valid_keepalive = _resolved_scan_call(2, "f32")
        wrong_scope, scope_keepalive = _resolved_scan_call(2, "f32")
        wrong_scope.impl.contents.execution_context_constraints.scope = (
            2  # Warp.
        )
        wrong_warps, warps_keepalive = _resolved_scan_call(2, "f32")
        context = wrong_warps.impl.contents.execution_context_constraints
        context.num_warps_per_cta = 2

        for call in (wrong_scope, wrong_warps):
            result, rank, error = _filter_result(instance, call)
            assert result == TILEIR_LP_SUCCESS
            assert error is None
            assert rank.disposition == TILEIR_LP_RANK_FILTERED
            assert rank.score == 0

        result, error = _instantiate_result(
            instance, [valid, wrong_scope, wrong_warps]
        )

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS
        assert tuple(call.call_index for call in error.unresolved_calls) == (
            1,
            2,
        )
        assert (
            "[LP-TOPOLOGY-001] implementation execution scope "
            "does not match resolved scope" in error.unresolved_calls[0].reason
        )
        assert (
            "[LP-TOPOLOGY-001] implementation warp count "
            "does not match resolved warp count"
            in error.unresolved_calls[1].reason
        )
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
        assert valid_keepalive and scope_keepalive and warps_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_ranking_rejects_corrupted_auto_topology_identity(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        corrupted_impl = b"cuda_coop_tile_scan_impl__lpv4_deadbeef"
        call, keepalive = _resolved_scan_call(
            2,
            "f32",
            impl_symbol=corrupted_impl.decode(),
        )

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert (
            "[LP-PROVIDER-INTERNAL-001]" in f"{error.message}\n{error.details}"
        )
        assert "cuda_coop_tile_scan" in f"{error.message}\n{error.details}"
        assert keepalive and corrupted_impl
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_empty_concrete_implementation_suffix(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        selected_impl = _string_view_value(call.impl.contents.symbol)
        empty_specialization = (
            selected_impl.removesuffix(DEFAULT_LATE_CALLBACK_SUFFIX).encode()
            + b"__"
        )
        call.impl.contents.symbol = _string_view(empty_specialization)

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        diagnostic = f"{error.message}\n{error.details}"
        assert "[LP-PROVIDER-INTERNAL-001]" in diagnostic

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-PROVIDER-INTERNAL-001]" in error.unresolved_calls[0].reason
        assert keepalive and empty_specialization
    finally:
        instance.vtable.destroy(instance.handle)


def test_provider_rejects_unrelated_and_unsafe_selected_symbols_before_nvrtc(
    monkeypatch,
    tmp_path,
):
    compile_counter = tmp_path / "nvrtc-compiles.txt"
    monkeypatch.setenv(
        "CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE",
        str(compile_counter),
    )
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        unrelated_alias, unrelated_keepalive = _resolved_scan_call(1, "f32")
        selected_impl = _string_view_value(unrelated_alias.impl.contents.symbol)
        concrete_base = selected_impl.removesuffix(DEFAULT_LATE_CALLBACK_SUFFIX)
        unrelated_alias.impl.contents.symbol = _string_view(
            f"{concrete_base}alias"
        )

        unsafe_alias, unsafe_keepalive = _resolved_scan_call(1, "f32")
        unsafe_alias.impl.contents.symbol = _string_view(
            f"{concrete_base}__bad-name"
        )

        for invalid_call in (unrelated_alias, unsafe_alias):
            result, _, error = _filter_result(instance, invalid_call)
            assert result == TILEIR_LP_ERROR
            assert error is not None
            assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
            assert (
                "[LP-PROVIDER-INTERNAL-001]"
                in f"{error.message}\n{error.details}"
            )

        result, error = _instantiate_result(
            instance,
            [unrelated_alias, unsafe_alias],
        )

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert tuple(call.call_index for call in error.unresolved_calls) == (
            0,
            1,
        )
        assert "[LP-PROVIDER-INTERNAL-001]" in error.unresolved_calls[0].reason
        assert "[LP-PROVIDER-INTERNAL-001]" in error.unresolved_calls[1].reason
        assert not compile_counter.exists()
        assert unrelated_keepalive and unsafe_keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_canonical_sibling_as_concrete_declaration(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_histogram_call("histogram", bins=256)
        sibling_declaration = b"cuda_coop_tile_histogram_256"
        call.decl.contents.symbol = _string_view(sibling_declaration)
        call.impl.contents.decl_symbol = _string_view(sibling_declaration)

        result, _, error = _filter_result(instance, call)
        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "[LP-SYMBOL-001]" in f"{error.message}\n{error.details}"

        result, error = _instantiate_result(instance, [call])
        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-SYMBOL-001]" in error.unresolved_calls[0].reason
        assert keepalive and sibling_declaration
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        pytest.param("malformed", "[LP-SCHEMA-001]", id="malformed"),
        pytest.param(
            "identity",
            "[LP-PROVIDER-INTERNAL-001]",
            id="provider-identity-corruption",
        ),
    ],
)
def test_native_provider_ranking_rejects_malformed_or_corrupted_owned_calls(
    monkeypatch, failure, diagnostic
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        corrupted_impl = b"cuda_coop_tile_scan_impl__lpv4_deadbeef"
        call, keepalive = _resolved_scan_call(
            1,
            "f32",
            impl_symbol=corrupted_impl.decode()
            if failure == "identity"
            else None,
        )
        if failure == "malformed":
            call.num_const_arguments -= 1

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert diagnostic in f"{error.message}\n{error.details}"
        assert "cuda_coop_tile_scan" in f"{error.message}\n{error.details}"
        assert keepalive and corrupted_impl
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_ranking_rejects_omitted_resolved_access_schema(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        call.decl.contents.param_access = None

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        diagnostic = f"{error.message}\n{error.details}"
        assert "cuda_coop_tile_scan" in diagnostic
        assert "[LP-SCHEMA-001]" in diagnostic
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_rejects_residual_resolved_constant_schema(monkeypatch):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        const_params = keepalive[4]
        call.decl.contents.const_params = const_params
        call.decl.contents.num_const_params = len(const_params)

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "[LP-SCHEMA-001]" in f"{error.message}\n{error.details}"

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        assert "[LP-SCHEMA-001]" in error.unresolved_calls[0].reason
        assert keepalive and const_params
    finally:
        instance.vtable.destroy(instance.handle)


@pytest.mark.parametrize("corruption", ("family", "declaration-relation"))
def test_native_provider_rejects_corrupted_concrete_symbol_relations(
    monkeypatch, corruption
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        if corruption == "family":
            corrupted = b"cuda_coop_tile_reduce__CompilerConcrete1"
            call.decl.contents.symbol = _string_view(corrupted)
            call.impl.contents.decl_symbol = _string_view(corrupted)
        else:
            corrupted = b"cuda_coop_tile_scan__DifferentConcrete2"
            call.impl.contents.decl_symbol = _string_view(corrupted)

        result, _, error = _filter_result(instance, call)

        assert result == TILEIR_LP_ERROR
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
        assert "[LP-SYMBOL-001]" in f"{error.message}\n{error.details}"

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert error.kind == TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS
        assert len(error.unresolved_calls) == 1
        assert "[LP-SYMBOL-001]" in error.unresolved_calls[0].reason
        assert keepalive and corrupted
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_instantiation_aggregates_every_invalid_selected_call(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        unsupported, unsupported_keepalive = _resolved_reduce_call(
            "sum", 1, "i32"
        )
        future_dtype = b"future_dtype"
        unsupported.const_arguments[
            unsupported.num_const_arguments - 1
        ].string_val = _string_view(future_dtype)

        malformed, malformed_keepalive = _resolved_scan_call(1, "f32")
        malformed.num_const_arguments -= 1

        result, error = _instantiate_result(instance, [unsupported, malformed])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert error.code == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error.kind == TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS
        assert tuple(call.call_index for call in error.unresolved_calls) == (
            0,
            1,
        )
        assert "[LP-UNSUPPORTED-001]" in error.unresolved_calls[0].reason
        assert "[LP-SCHEMA-001]" in error.unresolved_calls[1].reason
        assert "one or more selected calls" in error.message
        assert unsupported_keepalive and malformed_keepalive and future_dtype
    finally:
        instance.vtable.destroy(instance.handle)


def test_native_provider_null_selected_implementation_has_actionable_reason(
    monkeypatch,
):
    lib, _ = _load_provider_or_skip(monkeypatch)
    instance = _create_provider(lib)
    try:
        call, keepalive = _resolved_scan_call(1, "f32")
        call.impl = None

        result, error = _instantiate_result(instance, [call])

        assert result == TILEIR_LP_ERROR_UNRESOLVED_CALLS
        assert error is not None
        assert len(error.unresolved_calls) == 1
        reason = error.unresolved_calls[0].reason
        assert "[LP-SYMBOL-001]" in reason
        assert "<null>" in reason
        assert "cuda_coop_tile_scan" in reason
        assert keepalive
    finally:
        instance.vtable.destroy(instance.handle)
