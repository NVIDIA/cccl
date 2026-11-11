# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

CCCL_ROOT = SOURCE_ROOT.parents[1]

TILE_ROOT = SOURCE_ROOT / "cuda" / "coop" / "tile"

NATIVE_ROOT = CCCL_ROOT / "tile/cpp/tilelib"

PROVIDER_SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".inc"}
)

PACKAGE_CMAKE = SOURCE_ROOT / "CMakeLists.txt"


def _provider_source_paths():
    paths = tuple(
        sorted(
            path
            for path in NATIVE_ROOT.rglob("*")
            if path.is_file() and path.suffix in PROVIDER_SOURCE_SUFFIXES
        )
    )
    assert paths, f"no native provider sources found under {NATIVE_ROOT}"
    return paths


def _provider_source_text():
    return "\n".join(
        f"// source: {path.relative_to(NATIVE_ROOT)}\n{path.read_text()}"
        for path in _provider_source_paths()
    )


TILEIR_LP_SUCCESS = 0

TILEIR_LP_ERROR = 1

TILEIR_LP_ERROR_INVALID_ARGUMENT = 2

TILEIR_LP_ERROR_UNSUPPORTED_VERSION = 3

TILEIR_LP_ERROR_UNRESOLVED_CALLS = 6

TILEIR_LP_ERROR_KIND_GENERIC = 0

TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS = 1

TILEIR_LINKAGE_PROVIDER_API_VERSION = (0, 0, 1)

TILEIR_LP_REQUEST_FORMAT_VERSION = 1

TILEIR_OUTPUT_LTO_IR = 1

TILEIR_OUTPUT_FATBIN = 3

TILEIR_EXEC_SCOPE_CTA = 4

TILEIR_DIM_CONCRETE = 1

TILEIR_TYPE_TILE = 0

TILEIR_DTYPE_I8 = 2

TILEIR_DTYPE_I32 = 4

TILEIR_DTYPE_I64 = 5

TILEIR_DTYPE_F16 = 6

TILEIR_DTYPE_BF16 = 7

TILEIR_DTYPE_F32 = 8

TILEIR_DTYPE_F64 = 10

TILEIR_CONSTANT_ARGUMENT_INT = 0

TILEIR_CONSTANT_ARGUMENT_STRING = 2

TILEIR_CONSTANT_ARGUMENT_DTYPE = 3

TILEIR_CONSTANT_PARAMETER_INT = 0

TILEIR_CONSTANT_PARAMETER_DTYPE = 2

TILEIR_CONSTANT_PARAMETER_STRING = 3

TILEIR_SYMBOLIC_EXPRESSION_INT = 0

TILEIR_SYMBOLIC_EXPRESSION_STRING = 1

TILEIR_SYMBOLIC_EXPRESSION_DTYPE = 2

TILEIR_LP_ACCESS_IN = 0

TILEIR_LP_ACCESS_OUT = 1

TILEIR_LP_ACCESS_INOUT = 2

TILEIR_LP_CANDIDATE_FRONTEND = 0

TILEIR_LP_CANDIDATE_PROVIDER = 1

TILEIR_LP_RANK_FILTERED = 1

TILEIR_LP_RANK_SCORED = 2

DTYPE_KINDS = {
    "i8": TILEIR_DTYPE_I8,
    "u8": TILEIR_DTYPE_I8,
    "i32": TILEIR_DTYPE_I32,
    "u32": TILEIR_DTYPE_I32,
    "i64": TILEIR_DTYPE_I64,
    "u64": TILEIR_DTYPE_I64,
    "f16": TILEIR_DTYPE_F16,
    "bf16": TILEIR_DTYPE_BF16,
    "f32": TILEIR_DTYPE_F32,
    "f64": TILEIR_DTYPE_F64,
}

REDUCE_DECLS = {
    "sum": "cuda_coop_tile_sum",
    "reduce": "cuda_coop_tile_reduce",
}
SCAN_DECL = "cuda_coop_tile_scan"
HISTOGRAM_DECLS = {
    "histogram": "cuda_coop_tile_histogram",
    "histogram_256": "cuda_coop_tile_histogram_256",
}
RUN_LENGTH_DECODE_DECLS = {
    "items": "cuda_coop_tile_run_length_decode",
    "items_and_offsets": "cuda_coop_tile_run_length_decode_with_offsets",
}
DISCONTINUITY_DECLS = {
    "flags": "cuda_coop_tile_discontinuity_flags",
    "heads_and_tails": "cuda_coop_tile_discontinuity_heads_and_tails",
}
ADJACENT_DIFFERENCE_DECL = "cuda_coop_tile_adjacent_difference"
SHUFFLE_DECL = "cuda_coop_tile_shuffle"
EXCHANGE_DECL = "cuda_coop_tile_exchange"
RADIX_RANK_DECL = "cuda_coop_tile_radix_rank"
MERGE_SORT_DECLS = {
    "keys": "cuda_coop_tile_merge_sort_keys",
    "pairs": "cuda_coop_tile_merge_sort_pairs",
}
RADIX_SORT_DECLS = {
    "keys": "cuda_coop_tile_radix_sort_keys",
    "pairs": "cuda_coop_tile_radix_sort_pairs",
}
TOPK_DECLS = {
    "keys": "cuda_coop_tile_topk_keys",
    "pairs": "cuda_coop_tile_topk_pairs",
}

CANONICAL_DECLARATIONS = (
    "cuda_coop_tile_scan",
    "cuda_coop_tile_sum",
    "cuda_coop_tile_reduce",
    "cuda_coop_tile_sum_batched",
    "cuda_coop_tile_reduce_batched",
    "cuda_coop_tile_exchange",
    "cuda_coop_tile_shuffle",
    "cuda_coop_tile_adjacent_difference",
    "cuda_coop_tile_discontinuity_flags",
    "cuda_coop_tile_discontinuity_heads_and_tails",
    "cuda_coop_tile_histogram",
    "cuda_coop_tile_histogram_256",
    "cuda_coop_tile_run_length_decode",
    "cuda_coop_tile_run_length_decode_with_offsets",
    "cuda_coop_tile_merge_sort_keys",
    "cuda_coop_tile_merge_sort_pairs",
    "cuda_coop_tile_radix_sort_keys",
    "cuda_coop_tile_radix_sort_pairs",
    "cuda_coop_tile_radix_rank",
    "cuda_coop_tile_topk_keys",
    "cuda_coop_tile_topk_pairs",
)
CANONICAL_IMPLEMENTATION_BASES = tuple(
    f"{declaration}_impl" for declaration in CANONICAL_DECLARATIONS
)

DEFAULT_LATE_CALLBACK_SUFFIX = "__CompilerConcrete1"


class TileIRLinkageProviderAPIVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("patch", ctypes.c_uint32),
    ]


class TileIRStringView(ctypes.Structure):
    _fields_ = [("data", ctypes.c_char_p), ("len", ctypes.c_size_t)]


class TileIRDimValue(ctypes.Union):
    _fields_ = [("name", TileIRStringView), ("value", ctypes.c_int64)]


class TileIRDim(ctypes.Structure):
    _anonymous_ = ("payload",)
    _fields_ = [("kind", ctypes.c_int), ("payload", TileIRDimValue)]


class TileIRDataType(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int)]


class TileIRForeignType(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("shape", ctypes.POINTER(TileIRDim)),
        ("num_shape", ctypes.c_size_t),
        ("dtype", TileIRDataType),
    ]


class TileIRConstantArgument(ctypes.Structure):
    pass


class TileIRConstantArgumentTuple(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.POINTER(TileIRConstantArgument)),
        ("num_values", ctypes.c_size_t),
    ]


class TileIRConstantArgumentPayload(ctypes.Union):
    _fields_ = [
        ("int_val", ctypes.c_int64),
        ("float_val", ctypes.c_double),
        ("string_val", TileIRStringView),
        ("dtype_val", TileIRDataType),
        ("tuple_val", TileIRConstantArgumentTuple),
    ]


TileIRConstantArgument._anonymous_ = ("payload",)

TileIRConstantArgument._fields_ = [
    ("kind", ctypes.c_int),
    ("payload", TileIRConstantArgumentPayload),
]


class TileIRConstantParameter(ctypes.Structure):
    _fields_ = [("name", TileIRStringView), ("kind", ctypes.c_int)]


class TileIRSymbolicExpression(ctypes.Structure):
    pass


class TileIRSymbolicExpressionTuple(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.POINTER(TileIRSymbolicExpression)),
        ("num_values", ctypes.c_size_t),
    ]


class TileIRSymbolicExpressionBinary(ctypes.Structure):
    _fields_ = [
        ("lhs", ctypes.POINTER(TileIRSymbolicExpression)),
        ("rhs", ctypes.POINTER(TileIRSymbolicExpression)),
    ]


class TileIRSymbolicExpressionBuiltin(ctypes.Structure):
    _fields_ = [("name", TileIRStringView), ("result_kind", ctypes.c_int)]


class TileIRSymbolicExpressionPayload(ctypes.Union):
    _fields_ = [
        ("int_val", ctypes.c_int64),
        ("string_val", TileIRStringView),
        ("dtype_val", TileIRDataType),
        ("variable_name", TileIRStringView),
        ("builtin_val", TileIRSymbolicExpressionBuiltin),
        ("tuple_val", TileIRSymbolicExpressionTuple),
        ("binary_val", TileIRSymbolicExpressionBinary),
        ("operand", ctypes.POINTER(TileIRSymbolicExpression)),
    ]


TileIRSymbolicExpression._anonymous_ = ("payload",)

TileIRSymbolicExpression._fields_ = [
    ("kind", ctypes.c_int),
    ("payload", TileIRSymbolicExpressionPayload),
]


class TileIRExecutionContext(ctypes.Structure):
    _fields_ = [
        ("scope", ctypes.c_int),
        ("num_warps_per_cta", ctypes.c_int64),
        ("num_ctas_per_cga", ctypes.c_int64),
        ("architecture", TileIRStringView),
    ]


class TileIRForeignDecl(ctypes.Structure):
    _fields_ = [
        ("symbol", TileIRStringView),
        ("const_params", ctypes.POINTER(TileIRConstantParameter)),
        ("num_const_params", ctypes.c_size_t),
        ("params", ctypes.POINTER(TileIRForeignType)),
        ("num_params", ctypes.c_size_t),
        ("results", ctypes.POINTER(TileIRForeignType)),
        ("num_results", ctypes.c_size_t),
        ("param_access", ctypes.POINTER(ctypes.c_int)),
        ("num_param_access", ctypes.c_size_t),
    ]


class TileIRLPCallsite(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("ir_preorder", ctypes.c_uint64),
        ("decl", ctypes.POINTER(TileIRForeignDecl)),
        ("const_arguments", ctypes.POINTER(TileIRSymbolicExpression)),
        ("num_const_arguments", ctypes.c_size_t),
        ("arguments", ctypes.POINTER(TileIRForeignType)),
        ("num_arguments", ctypes.c_size_t),
        ("results", ctypes.POINTER(TileIRForeignType)),
        ("num_results", ctypes.c_size_t),
    ]


class TileIRLPContextValue(ctypes.Structure):
    _fields_ = [
        ("name", TileIRStringView),
        ("kind", ctypes.c_int),
        ("i64", ctypes.c_int64),
    ]


class TileIRLPCompilationContext(ctypes.Structure):
    _fields_ = [
        ("target_architecture", TileIRStringView),
        ("num_warps_per_cta", ctypes.c_int64),
        ("num_ctas_per_cga", ctypes.c_int64),
        ("context_values", ctypes.POINTER(TileIRLPContextValue)),
        ("num_context_values", ctypes.c_size_t),
    ]


class TileIRLPHint(ctypes.Structure):
    _fields_ = [("key", TileIRStringView), ("value", TileIRStringView)]


class TileIRLPProvideImplsRequest(ctypes.Structure):
    _fields_ = [
        ("format_version", ctypes.c_uint32),
        ("callsites", ctypes.POINTER(TileIRLPCallsite)),
        ("num_callsites", ctypes.c_size_t),
        ("compilation_context", TileIRLPCompilationContext),
        ("hints", ctypes.POINTER(TileIRLPHint)),
        ("num_hints", ctypes.c_size_t),
    ]


class TileIRForeignImpl(ctypes.Structure):
    _fields_ = [
        ("symbol", TileIRStringView),
        ("decl_symbol", TileIRStringView),
        ("execution_context_constraints", TileIRExecutionContext),
    ]


class TileIRResolvedCall(ctypes.Structure):
    _fields_ = [
        ("decl", ctypes.POINTER(TileIRForeignDecl)),
        ("impl", ctypes.POINTER(TileIRForeignImpl)),
        ("resolved_execution_context", TileIRExecutionContext),
        ("const_arguments", ctypes.POINTER(TileIRConstantArgument)),
        ("num_const_arguments", ctypes.c_size_t),
        ("arguments", ctypes.POINTER(TileIRForeignType)),
        ("num_arguments", ctypes.c_size_t),
    ]


class TileIRLPFilterCandidate(ctypes.Structure):
    _fields_ = [
        ("callsite_id", ctypes.c_uint64),
        ("stable_ordinal", ctypes.c_uint64),
        ("provenance", ctypes.c_int),
        ("baseline_score", ctypes.c_int64),
        ("resolved_call", TileIRResolvedCall),
    ]


class TileIRLPFilterAndRankRequest(ctypes.Structure):
    _fields_ = [
        ("format_version", ctypes.c_uint32),
        ("compilation_context", TileIRLPCompilationContext),
        ("candidates", ctypes.POINTER(TileIRLPFilterCandidate)),
        ("num_candidates", ctypes.c_size_t),
    ]


class TileIRLPRankResult(ctypes.Structure):
    _fields_ = [("disposition", ctypes.c_int), ("score", ctypes.c_int64)]


class TileIRCodeBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("output_type", ctypes.c_int),
        ("symbols", ctypes.POINTER(TileIRStringView)),
        ("num_symbols", ctypes.c_size_t),
        ("name", TileIRStringView),
    ]


class TileIRResolvedCallSymbol(ctypes.Structure):
    _fields_ = [("call_index", ctypes.c_size_t), ("symbol", TileIRStringView)]


class TileIRForeignModule(ctypes.Structure):
    _fields_ = [
        ("code_buffers", ctypes.POINTER(TileIRCodeBuffer)),
        ("num_code_buffers", ctypes.c_size_t),
        ("call_symbols", ctypes.POINTER(TileIRResolvedCallSymbol)),
        ("num_call_symbols", ctypes.c_size_t),
    ]


class TileIRLPUnresolvedCall(ctypes.Structure):
    _fields_ = [
        ("call_index", ctypes.c_size_t),
        ("reason", TileIRStringView),
    ]


class TileIRLPUnresolvedCallsError(ctypes.Structure):
    _fields_ = [
        ("calls", ctypes.POINTER(TileIRLPUnresolvedCall)),
        ("num_calls", ctypes.c_size_t),
    ]


class TileIRLPErrorPayload(ctypes.Union):
    _fields_ = [("unresolved_calls", TileIRLPUnresolvedCallsError)]


class TileIRLPError(ctypes.Structure):
    _anonymous_ = ("payload",)
    _fields_ = [
        ("code", ctypes.c_int),
        ("kind", ctypes.c_int),
        ("message", TileIRStringView),
        ("details", TileIRStringView),
        ("payload", TileIRLPErrorPayload),
    ]


TileIRLPErrorPtr = ctypes.POINTER(TileIRLPError)


@dataclass(frozen=True)
class CapturedLPUnresolvedCall:
    call_index: int
    reason: str


@dataclass(frozen=True)
class CapturedLPError:
    code: int
    kind: int
    message: str
    details: str
    unresolved_calls: tuple[CapturedLPUnresolvedCall, ...]


ProvideImplsFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(TileIRLPProvideImplsRequest),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(TileIRLPErrorPtr),
)

FilterAndRankFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(TileIRLPFilterAndRankRequest),
    ctypes.POINTER(TileIRLPRankResult),
    ctypes.c_size_t,
    ctypes.POINTER(TileIRLPErrorPtr),
)

InstantiateFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(TileIRResolvedCall),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(TileIRForeignModule)),
    ctypes.POINTER(TileIRLPErrorPtr),
)

FreeImplsFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

FreeModuleFn = ctypes.CFUNCTYPE(None, ctypes.POINTER(TileIRForeignModule))

FreeErrorFn = ctypes.CFUNCTYPE(None, TileIRLPErrorPtr)

DestroyFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class TileIRLinkageProviderVTable(ctypes.Structure):
    _fields_ = [
        ("api_version", TileIRLinkageProviderAPIVersion),
        ("provide_impls", ProvideImplsFn),
        ("filter_and_rank", FilterAndRankFn),
        ("instantiate", InstantiateFn),
        ("free_impls", FreeImplsFn),
        ("free_module", FreeModuleFn),
        ("free_error", FreeErrorFn),
        ("destroy", DestroyFn),
    ]


class TileIRLinkageProviderInstance(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("vtable", TileIRLinkageProviderVTable),
        ("name", TileIRStringView),
    ]


def _string_view(value: str | bytes) -> TileIRStringView:
    encoded = value.encode() if isinstance(value, str) else value
    return TileIRStringView(encoded, len(encoded))


def _string_view_value(value: TileIRStringView) -> str:
    if not value.data or value.len == 0:
        return ""
    return ctypes.string_at(value.data, value.len).decode()


def _erase_resolved_decl_constant_schema(decl: TileIRForeignDecl) -> None:
    """Mirror the current TileIR resolved-declaration representation."""
    decl.const_params = None
    decl.num_const_params = 0


def _concretize_resolved_call_for_late_callback(
    call: TileIRResolvedCall,
    suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
) -> tuple[bytes, bytes]:
    """Mirror TileIR's late declaration/implementation concretization.

    TileIR may append either concrete parameter bindings (``__...``) or the
    symbol-table uniquifier used when an otherwise-identical concrete symbol is
    already present (``_0``, ``_1``, ...). Late provider callbacks see only
    these concrete spellings, never the parametric discovery symbol.
    """
    assert (suffix.startswith("__") and len(suffix) > 2) or (
        suffix.startswith("_") and suffix[1:].isdigit()
    )
    declaration = (
        f"{_string_view_value(call.decl.contents.symbol)}{suffix}".encode()
    )
    implementation = (
        f"{_string_view_value(call.impl.contents.symbol)}{suffix}".encode()
    )
    call.decl.contents.symbol = _string_view(declaration)
    _erase_resolved_decl_constant_schema(call.decl.contents)
    call.impl.contents.symbol = _string_view(implementation)
    call.impl.contents.decl_symbol = _string_view(declaration)
    call.impl.contents.execution_context_constraints.architecture = (
        _string_view(b"")
    )
    return declaration, implementation


def _copy_lp_error(error: TileIRLPErrorPtr) -> CapturedLPError:
    assert error
    payload = error.contents
    unresolved_calls: tuple[CapturedLPUnresolvedCall, ...] = ()
    if payload.kind == TILEIR_LP_ERROR_KIND_UNRESOLVED_CALLS:
        unresolved_calls = tuple(
            CapturedLPUnresolvedCall(
                payload.unresolved_calls.calls[index].call_index,
                _string_view_value(
                    payload.unresolved_calls.calls[index].reason
                ),
            )
            for index in range(payload.unresolved_calls.num_calls)
        )
    return CapturedLPError(
        payload.code,
        payload.kind,
        _string_view_value(payload.message),
        _string_view_value(payload.details),
        unresolved_calls,
    )


def _copy_and_free_lp_error(instance, error: TileIRLPErrorPtr):
    if not error:
        return None
    try:
        return _copy_lp_error(error)
    finally:
        instance.vtable.free_error(error)


def _api_version() -> TileIRLinkageProviderAPIVersion:
    return TileIRLinkageProviderAPIVersion(*TILEIR_LINKAGE_PROVIDER_API_VERSION)


def _compilation_context(
    architecture: str = "sm_80",
) -> TileIRLPCompilationContext:
    return TileIRLPCompilationContext(
        _string_view(architecture), 4, -1, None, 0
    )


def _execution_context(architecture: str = "sm_80") -> TileIRExecutionContext:
    return TileIRExecutionContext(
        TILEIR_EXEC_SCOPE_CTA, 4, -1, _string_view(architecture)
    )


def _provider_topology_score(block_threads: int) -> int:
    distance = abs(block_threads - 128)
    return (1024 - distance) * 1024 + (1024 - block_threads)


def _fnv1a(value: str) -> int:
    result = 1469598103934665603
    for byte in value.encode():
        result ^= byte
        result = (result * 1099511628211) & ((1 << 64) - 1)
    return result


def _artifact_identity(
    semantic_identity: str, architecture: str = "sm_80"
) -> str:
    """Mirror the provider's exact target-family artifact identity boundary."""
    spelling = architecture
    for prefix in ("sm_", "compute_"):
        if spelling.startswith(prefix):
            spelling = spelling.removeprefix(prefix)
            break
    family = spelling[-1] if spelling.endswith(("a", "f")) else ""
    digits = spelling[:-1] if family else spelling
    target = f"sm_{int(digits)}{family}"
    return f"{semantic_identity}|tile_interface_abi=2|target={target}"


def _native_provider_path() -> Path | None:
    if configured := os.environ.get("CUDA_COOP_TILE_LINKAGE_PROVIDER_PATH"):
        provider_path = Path(configured)
        if not provider_path.is_file():
            pytest.fail(
                "CUDA_COOP_TILE_LINKAGE_PROVIDER_PATH is not a file: "
                f"{provider_path}"
            )
        return provider_path

    candidates = sorted(
        CCCL_ROOT.glob("build/**/libcoop_linkage_provider.so"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    source_mtime = max(
        path.stat().st_mtime for path in _provider_source_paths()
    )
    return next(
        (path for path in candidates if path.stat().st_mtime >= source_mtime),
        None,
    )


def _create_provider(lib) -> TileIRLinkageProviderInstance:
    create = lib.tileir_create_linkage_provider
    create.argtypes = [
        TileIRLinkageProviderAPIVersion,
        ctypes.POINTER(TileIRLinkageProviderInstance),
    ]
    create.restype = ctypes.c_int
    instance = TileIRLinkageProviderInstance()
    assert create(_api_version(), ctypes.byref(instance)) == TILEIR_LP_SUCCESS
    version = instance.vtable.api_version
    assert (version.major, version.minor, version.patch) == (
        TILEIR_LINKAGE_PROVIDER_API_VERSION
    )
    assert _string_view_value(instance.name) == "cuda-coop-tile"
    assert instance.handle
    assert instance.vtable.provide_impls
    assert instance.vtable.filter_and_rank
    assert instance.vtable.instantiate
    assert instance.vtable.free_impls
    assert instance.vtable.free_module
    assert instance.vtable.free_error
    assert instance.vtable.destroy
    return instance


def _load_provider_or_skip(monkeypatch):
    provider_path = _native_provider_path()
    if provider_path is None or not provider_path.is_file():
        pytest.skip(
            "build the native provider or set "
            "CUDA_COOP_TILE_LINKAGE_PROVIDER_PATH"
        )
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(CCCL_ROOT))
    monkeypatch.delenv("CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", raising=False)
    monkeypatch.delenv("CUDA_COOP_TILE_INCLUDE_PATHS", raising=False)
    monkeypatch.delenv("CUDA_COOP_TILE_PROVIDER_SM_ARCH", raising=False)
    monkeypatch.delenv("CUDA_ARCH", raising=False)
    return ctypes.CDLL(str(provider_path)), provider_path


def _provide_impls_result(instance, request: TileIRLPProvideImplsRequest):
    data = ctypes.c_void_p()
    size = ctypes.c_size_t()
    error = TileIRLPErrorPtr()
    result = instance.vtable.provide_impls(
        instance.handle,
        ctypes.byref(request),
        ctypes.byref(data),
        ctypes.byref(size),
        ctypes.byref(error),
    )
    try:
        catalog = (
            ctypes.string_at(data, size.value) if data and size.value else b""
        )
        captured_error = _copy_lp_error(error) if error else None
        return result, catalog, captured_error
    finally:
        if data:
            instance.vtable.free_impls(data)
        if error:
            instance.vtable.free_error(error)


def _provide_impls(instance, request: TileIRLPProvideImplsRequest) -> bytes:
    result, catalog, error = _provide_impls_result(instance, request)
    assert result == TILEIR_LP_SUCCESS
    assert error is None
    return catalog


def _assert_malformed(instance, request, diagnostic: str) -> CapturedLPError:
    result, catalog, error = _provide_impls_result(instance, request)
    assert result == TILEIR_LP_ERROR
    assert catalog == b""
    assert error is not None
    assert error.kind == TILEIR_LP_ERROR_KIND_GENERIC
    assert diagnostic in error.details
    return error


def _assert_soft_miss(instance, request) -> None:
    result, catalog, error = _provide_impls_result(instance, request)
    assert result == TILEIR_LP_SUCCESS
    assert catalog == b""
    assert error is None
