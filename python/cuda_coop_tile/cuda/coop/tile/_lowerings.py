# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Translate cuTile compiler arguments into canonical provider calls.

Public markers reach this module through the cuTile implementation registry.
The block adapters normalize options and validate known values. These
lowerings then separate constant specialization choices from runtime operands
and append preview foreign operations to the active intermediate
representation (IR). They do not run CUB or choose the final kernel workers.

Static-prefix helpers follow the generated declaration's parameter order.
Signed dtype tokens preserve distinctions that TileIR's signless integer
types cannot express. The native provider validates those declarations again
before selecting, generating, and compiling concrete CUB implementations.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
from collections.abc import Mapping
from itertools import pairwise
from math import prod
from pathlib import Path

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.core_ops import Assign
from cuda.tile._ir.ir import Builder, add_operation_variadic
from cuda.tile._ir.op_impl import (
    normalize_axis,
    require_constant_bool,
    require_constant_int_tuple,
    require_optional_constant_str,
    require_tile_type,
)
from cuda.tile._ir.ops import (
    _const,
    astype,
    broadcast_to,
    build_tuple,
    full,
    permute,
    reshape,
)
from cuda.tile._ir.type import TileTy as make_tile_ty
from cuda.tile._ir.type import TokenTy

from ._block.exchange import (
    BlockExchangeBlockedToStriped,
    BlockExchangeStripedToBlocked,
)
from ._block.reduce import (
    BlockReduce,
    BlockReduceOptions,
    BlockReduceSum,
    BlockReduceSumOptions,
)
from ._block.scan import (
    BlockScanExclusiveScan,
    BlockScanExclusiveScanOptions,
    BlockScanExclusiveSum,
    BlockScanExclusiveSumOptions,
    BlockScanInclusiveScan,
    BlockScanInclusiveScanOptions,
    BlockScanInclusiveSum,
    BlockScanInclusiveSumOptions,
)
from ._cpp import _get_tile_library_linker_input
from ._declarations import DECLARATIONS as _PROVIDER_DECL_SCHEMAS
from ._declarations import TileType as _ProviderDeclTileType
from ._dispatch import (
    dispatch_exchange_blocked_to_striped,
    dispatch_exchange_striped_to_blocked,
    dispatch_exclusive_scan,
    dispatch_exclusive_sum,
    dispatch_inclusive_scan,
    dispatch_inclusive_sum,
    dispatch_reduce,
    dispatch_sum,
)

_DTYPE_TO_TOKEN = {
    datatype.bfloat16: "bf16",
    datatype.float16: "f16",
    datatype.float32: "f32",
    datatype.float64: "f64",
    datatype.int8: "i8",
    datatype.uint8: "u8",
    datatype.int16: "i16",
    datatype.uint16: "u16",
    datatype.int32: "i32",
    datatype.uint32: "u32",
    datatype.int64: "i64",
    datatype.uint64: "u64",
}

_REDUCE_OP_TOKEN = {
    "+": "sum",
    "sum": "sum",
    "*": "mul",
    "mul": "mul",
    "multiplies": "mul",
    "min": "min",
    "max": "max",
    "&": "bit_and",
    "|": "bit_or",
    "^": "bit_xor",
    "bit_and": "bit_and",
    "bit_or": "bit_or",
    "bit_xor": "bit_xor",
}

_SCAN_OP_TOKEN = {
    "+": "sum",
    "sum": "sum",
    "*": "mul",
    "mul": "mul",
    "multiplies": "mul",
    "min": "min",
    "max": "max",
    "&": "bit_and",
    "|": "bit_or",
    "^": "bit_xor",
    "bit_and": "bit_and",
    "bit_or": "bit_or",
    "bit_xor": "bit_xor",
}

_PROVIDER_DECL_SUM = "cuda_coop_tile_sum_batched"
_PROVIDER_DECL_REDUCE = "cuda_coop_tile_reduce_batched"
_PROVIDER_DECL_SCAN = "cuda_coop_tile_scan"
_PROVIDER_DECL_HISTOGRAM = "cuda_coop_tile_histogram"
_PROVIDER_DECL_HISTOGRAM_256 = "cuda_coop_tile_histogram_256"
_PROVIDER_DECL_RUN_LENGTH_DECODE = "cuda_coop_tile_run_length_decode"
_PROVIDER_DECL_RUN_LENGTH_DECODE_WITH_OFFSETS = (
    "cuda_coop_tile_run_length_decode_with_offsets"
)
_PROVIDER_DECL_DISCONTINUITY_FLAGS = "cuda_coop_tile_discontinuity_flags"
_PROVIDER_DECL_DISCONTINUITY_HEADS_AND_TAILS = (
    "cuda_coop_tile_discontinuity_heads_and_tails"
)
_PROVIDER_DECL_ADJACENT_DIFFERENCE = "cuda_coop_tile_adjacent_difference"
_PROVIDER_DECL_EXCHANGE = "cuda_coop_tile_exchange"
_PROVIDER_DECL_MERGE_SORT_KEYS = "cuda_coop_tile_merge_sort_keys"
_PROVIDER_DECL_MERGE_SORT_PAIRS = "cuda_coop_tile_merge_sort_pairs"
_PROVIDER_DECL_RADIX_SORT_KEYS = "cuda_coop_tile_radix_sort_keys"
_PROVIDER_DECL_RADIX_SORT_PAIRS = "cuda_coop_tile_radix_sort_pairs"
_PROVIDER_DECL_TOPK_KEYS = "cuda_coop_tile_topk_keys"
_PROVIDER_DECL_TOPK_PAIRS = "cuda_coop_tile_topk_pairs"
_PROVIDER_DECL_SHUFFLE = "cuda_coop_tile_shuffle"
_PROVIDER_DECL_RADIX_RANK = "cuda_coop_tile_radix_rank"

_PROVIDER_REQUIRED_CCCL_HEADERS = (
    "cub/block/block_adjacent_difference.cuh",
    "cub/block/block_discontinuity.cuh",
    "cub/block/block_exchange.cuh",
    "cub/block/block_histogram.cuh",
    "cub/block/block_merge_sort.cuh",
    "cub/block/block_radix_rank.cuh",
    "cub/block/block_radix_sort.cuh",
    "cub/block/block_reduce.cuh",
    "cub/block/block_run_length_decode.cuh",
    "cub/block/block_scan.cuh",
    "cub/block/block_shuffle.cuh",
    "cub/block/block_topk.cuh",
    "cub/thread/thread_operators.cuh",
)
_PROVIDER_INCLUDE_PATH_ENV = "CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS"
_PROVIDER_INCLUDE_PATH_SEPARATOR = "\x1f"

_PROVIDER_SUM_DTYPES = {
    datatype.bfloat16,
    datatype.float16,
    datatype.float32,
    datatype.float64,
    datatype.int8,
    datatype.int16,
    datatype.int32,
    datatype.int64,
}

_RADIX_RANK_DTYPES = {
    datatype.int8,
    datatype.uint8,
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
}

_INTEGER_DTYPES = {
    datatype.int8,
    datatype.uint8,
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
}

_MISSING = object()
_RUNTIME_VALUE = object()


def _require_optional_constant_object(var, name: str):
    if var.is_constant():
        return var.get_constant()
    raise TileTypeError(f"{name} must be a compile-time constant")


def _const_value_or_runtime(var):
    if var.is_constant():
        return var.get_constant()
    return _RUNTIME_VALUE


def _is_const_none(var) -> bool:
    return var.is_constant() and var.get_constant() is None


def _is_provided(var) -> bool:
    return not _is_const_none(var)


def _method_contains(methods, key: str) -> bool:
    if methods is None:
        return False
    if isinstance(methods, Mapping):
        return key in methods
    if isinstance(methods, tuple):
        return any(methods[i] == key for i in range(0, len(methods), 2))
    return False


def _reject_runtime_method_overlap(
    methods, var, key: str, primitive: str
) -> None:
    """Reject two sources for a runtime control before merging methods.

    Constant explicit values can take precedence in the block adapter.
    A runtime IR value cannot enter that constant dictionary, so accepting a
    second value there would silently select a different control.
    """

    if var.is_constant():
        return
    if _method_contains(methods, key):
        raise TileTypeError(
            f"coop.{primitive} cannot combine runtime {key} with "
            f"methods['{key}']"
        )


def _validate_i32_constant(value, *, primitive: str, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TileTypeError(f"coop.{primitive} {name} must be an int")
    return value


def _validate_numeric_constant(value, dtype, *, primitive: str, name: str):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TileTypeError(f"coop.{primitive} {name} must be a numeric value")
    if dtype in _INTEGER_DTYPES and not isinstance(value, int):
        raise TileTypeError(f"coop.{primitive} {name} must be an integer")
    return value


def _constant_scalar_tile(value, dtype, *, shape=()):
    return _const(shape, value, dtype)


def _constant_rank_zero_tile(value, dtype):
    return _const((), value, dtype)


def _scalar_tile_operand(
    var,
    dtype,
    *,
    shape: tuple[int, ...],
    primitive: str,
    name: str,
    default=_MISSING,
    int_value: bool = False,
    numeric_value: bool = False,
    allow_shape_one: bool = True,
):
    """Convert a public scalar control to a provider operand.

    Constants are checked and materialized directly. Runtime one-element
    Tiles are cast and reshaped; scalar IR values are broadcast to ``shape``.
    These operations append IR through the active builder.

    Parameters
    ----------
    var : IR variable
        Public argument, including a constant ``None`` for an omitted value.
    dtype : DType
        Element type required by the foreign declaration.
    shape : tuple of int
        Shape of the provider operand, usually ``()``.
    primitive : str
        Public operation name used in diagnostics.
    name : str
        Argument name used in diagnostics.
    default : object, optional
        Constant substituted for ``None``. With no default, omission is an
        error.
    int_value : bool, optional
        Require an integer Python constant, excluding booleans.
    numeric_value : bool, optional
        Check numeric constants and require integers for integer element types.
    allow_shape_one : bool, optional
        Accept a runtime one-element Tile with nonzero rank. When false,
        runtime Tile controls must already have rank zero.

    Returns
    -------
    operand : IR variable
        Provider operand with the requested dtype and shape.
    constant : object or None
        Validated Python value for a constant input; ``None`` for runtime input.
        Callers use this value for checks without specializing runtime controls.

    Raises
    ------
    TileTypeError
        A required value is omitted or a scalar control has an invalid type
        or shape. Runtime range checks belong to the emitted implementation.
    """

    if var.is_constant():
        value = var.get_constant()
        if value is None:
            if default is _MISSING:
                raise TileTypeError(f"coop.{primitive} {name} must be provided")
            value = default
        if int_value:
            value = _validate_i32_constant(
                value, primitive=primitive, name=name
            )
        elif numeric_value:
            value = _validate_numeric_constant(
                value, dtype, primitive=primitive, name=name
            )
        return _constant_scalar_tile(value, dtype, shape=shape), value

    get_type = getattr(var, "get_type", None)
    var_ty = (
        get_type() if get_type is not None else getattr(var, "tile_type", None)
    )
    if hasattr(var_ty, "numel"):
        if not allow_shape_one and getattr(var_ty, "shape", None) != ():
            raise TileTypeError(
                f"coop.{primitive} {name} must be a scalar or rank-zero tile"
            )
        if var_ty.numel != 1:
            raise TileTypeError(
                f"coop.{primitive} {name} must be a scalar or size-1 tile"
            )
        casted = astype(var, dtype)
        casted_ty = casted.get_type()
        if getattr(casted_ty, "shape", None) == shape:
            return casted, None
        return reshape(casted, shape), None

    return full(shape, astype(var, dtype), dtype), None


def _rank0_scalar_tile_operand(
    var,
    dtype,
    *,
    primitive: str,
    name: str,
    default=_MISSING,
    int_value: bool = False,
    numeric_value: bool = False,
    allow_shape_one: bool = True,
):
    return _scalar_tile_operand(
        var,
        dtype,
        shape=(),
        primitive=primitive,
        name=name,
        default=default,
        int_value=int_value,
        numeric_value=numeric_value,
        allow_shape_one=allow_shape_one,
    )


def _require_empty_methods(var, primitive: str) -> None:
    methods = _require_optional_constant_object(var, "methods")
    if methods is None:
        return
    if isinstance(methods, Mapping) and not methods:
        return
    if isinstance(methods, tuple) and len(methods) == 0:
        return
    raise TileTypeError(f"coop.{primitive} does not support methods")


def _find_defining_op(var):
    """Find a value's producer across the supported cuTile IR forms.

    Use the context's producer map when available, then search the active
    builder backward. Return ``None`` outside a builder or when the value
    has no discoverable producer; callers must then keep the operation.
    """

    producer_map = getattr(var.ctx, "producers", None)
    if producer_map is not None:
        producer = producer_map.get(var.name)
        if producer is not None:
            return producer
    try:
        ops = Builder.get_current().ops
    except AssertionError:
        return None
    for op in reversed(ops):
        if any(res.name == var.name for res in op.result_vars):
            return op
    return None


def _exchange_call_kind(op) -> str | None:
    """Recognize only the canonical Exchange declaration and constants.

    The blocked fragment adapter reverses the provider direction, so translate
    it back to the public direction before folding an inverse call.
    """

    try:
        preview_foreign_call_type = _preview_foreign_call_type()
    except TileTypeError:
        return None
    if (
        not isinstance(op, preview_foreign_call_type)
        or op.symbol_name != _PROVIDER_DECL_EXCHANGE
        or len(op.constant_params) != 5
    ):
        return None
    return {
        "blocked_to_striped": "striped_to_blocked",
        "striped_to_blocked": "blocked_to_striped",
    }.get(op.constant_params[3])


def _find_exchange_origin(var):
    """Find an Exchange that directly produced the requested data result.

    Follow identity assignments but require the first data result. Return its
    public direction and operation, or ``(None, None)`` when provenance cannot
    be established.
    """

    producer = _find_defining_op(var)
    # cuTile represents named intermediate values with identity Assign ops.
    while isinstance(producer, Assign):
        var = producer.value
        producer = _find_defining_op(var)
    if producer is None:
        return None, None
    kind = _exchange_call_kind(producer)
    if kind is None:
        return None, None
    result_vars = tuple(getattr(producer, "result_vars", ()) or ())
    if not result_vars or result_vars[0] is not var:
        return None, None
    return kind, producer


def _exchange_has_topology(op, block_threads: int) -> bool:
    constant_params = getattr(op, "constant_params", ())
    return len(constant_params) >= 3 and constant_params[2] == block_threads


def _foreign_call_first_input(op):
    inputs = getattr(op, "inputs", None)
    if inputs:
        return inputs[0]

    raise TileTypeError("internal error: exchange origin has no input operand")


def _provider_include_paths() -> list[str]:
    try:
        from ._provider_runtime import _include_paths

        return _include_paths(
            required_headers=_PROVIDER_REQUIRED_CCCL_HEADERS,
        )
    except RuntimeError as exc:
        raise TileTypeError(
            "cuda.coop.tile could not resolve source-tree or wheel-bundled "
            "CCCL headers."
        ) from exc


def _ensure_provider_include_paths() -> None:
    """Make selected headers visible to the TileIRAS child process.

    Preserve an explicit provider include-path environment value. Otherwise,
    resolve source or wheel headers and write the encoded path list into this
    process's environment so later compiler children inherit it.
    """

    if os.environ.get(_PROVIDER_INCLUDE_PATH_ENV):
        return
    os.environ[_PROVIDER_INCLUDE_PATH_ENV] = (
        _PROVIDER_INCLUDE_PATH_SEPARATOR.join(_provider_include_paths())
    )


def _unary_static_prefix(
    numel: int,
    dtype,
):
    return (numel, dtype)


def _reduce_sum_static_prefix(
    numel: int,
    dtype,
    *,
    algorithm: str,
):
    return (
        *_unary_static_prefix(
            numel,
            dtype,
        ),
        algorithm,
        _dtype_token(dtype),
    )


def _reduce_generic_static_prefix(
    numel: int,
    dtype,
    *,
    algorithm: str,
    binary_op: str,
):
    return (
        *_unary_static_prefix(
            numel,
            dtype,
        ),
        algorithm,
        binary_op,
        _dtype_token(dtype),
    )


def _scan_static_prefix(
    numel: int,
    dtype,
    *,
    mode: str,
    algorithm: str,
    scan_op: str,
    initial_present: bool,
    prefix_present: bool,
):
    return (
        *_unary_static_prefix(
            numel,
            dtype,
        ),
        mode,
        algorithm,
        scan_op,
        int(initial_present),
        int(prefix_present),
        _dtype_token(dtype),
    )


def _unary_mode_static_prefix(
    numel: int,
    dtype,
    *,
    mode: str,
):
    return (
        *_unary_static_prefix(
            numel,
            dtype,
        ),
        mode,
        _dtype_token(dtype),
    )


def _exchange_static_prefix(
    numel: int,
    dtype,
    *,
    block_threads: int,
    mode: str,
):
    return (numel, dtype, block_threads, mode, _dtype_token(dtype))


def _sort_key_static_prefix(
    numel: int,
    dtype,
):
    return (numel, dtype)


def _sort_pair_static_prefix(
    numel: int,
    key_dtype,
    value_dtype,
):
    return (numel, key_dtype, value_dtype)


def _coop_tile_linker_input() -> tuple[str, str, Path]:
    try:
        distribution, version, path = _get_tile_library_linker_input()
    except RuntimeError as exc:
        raise TileTypeError(str(exc)) from exc

    # cuTile can return a cached cubin before the provider sees changed headers.
    # Preserve path order and include explicit roots even when the serialized
    # include-path escape hatch was supplied by the caller.
    include_inputs = tuple(
        os.environ.get(name, "")
        for name in (
            _PROVIDER_INCLUDE_PATH_ENV,
            "CUDA_COOP_TILE_INCLUDE_PATHS",
            "CUDA_COOP_CCCL_ROOT",
            "CCCL_ROOT",
            "CCCL_DIR",
            "CUDA_COOP_TILE_CUDA_ROOT",
            "CUDA_HOME",
            "CUDA_ROOT",
            "CUDA_PATH",
            "CUDA_TOOLKIT_PATH",
            "CUDACXX",
        )
    )
    path_inputs = list(include_inputs[2:])
    for value in include_inputs[:2]:
        separator = (
            _PROVIDER_INCLUDE_PATH_SEPARATOR
            if _PROVIDER_INCLUDE_PATH_SEPARATOR in value
            else os.pathsep
        )
        path_inputs.extend(value.split(separator))
    relative_paths = any(
        value and not Path(value).is_absolute() for value in path_inputs
    )
    working_directory = os.getcwd() if relative_paths else None
    encoded = json.dumps(
        (include_inputs, working_directory), separators=(",", ":")
    ).encode()
    include_hash = hashlib.sha256(encoded).hexdigest()
    return distribution, f"{version}+includes.{include_hash}", path


def _preview_foreign_call_type():
    try:
        from cuda.tile_preview import PreviewForeignCall
    except ImportError as exc:
        raise TileTypeError(
            "cuda.coop.tile requires cuda.tile_preview.foreign_call support "
            "from the cuTile preview package"
        ) from exc
    return PreviewForeignCall


def _dtype_token(dtype_obj) -> str:
    token = _DTYPE_TO_TOKEN.get(dtype_obj)
    if token is None:
        raise TileTypeError(
            f"unsupported dtype for coop primitive metadata: {dtype_obj}"
        )
    return token


def _reduce_op_token(op: str) -> str:
    token = _REDUCE_OP_TOKEN.get(op)
    if token is None:
        raise TileTypeError(
            f"unsupported reduce op for primitive metadata: {op}"
        )
    return token


def _scan_op_token(op: str) -> str:
    token = _SCAN_OP_TOKEN.get(op)
    if token is None:
        raise TileTypeError(f"unsupported scan op for primitive metadata: {op}")
    return token


def _provider_call_const_args(
    *,
    const_params: tuple[tuple[str, str], ...],
    static_prefix: tuple[object, ...],
) -> dict[str, object]:
    if len(static_prefix) != len(const_params):
        raise TileTypeError(
            "internal error: provider static prefix does not match const params"
        )
    return {
        name: value
        for (name, _), value in zip(const_params, static_prefix, strict=True)
    }


def _resolve_provider_decl_tile_type(
    *,
    decl_sym_name: str,
    tile_type: _ProviderDeclTileType,
    bindings: Mapping[str, object],
    parameter_kinds: Mapping[str, str],
):
    """Resolve a generated Tile type against this call's constants.

    Literal types and dimensions are used directly. Named references must
    have the corresponding ``DType`` or ``Dim`` parameter kind. Return the
    concrete dtype and shape, raising ``TileTypeError`` for an inconsistent
    schema or missing binding before a foreign operation is constructed.
    """

    dtype = tile_type.dtype
    if dtype in ("i8", "i32"):
        dtype = {"i8": datatype.int8, "i32": datatype.int32}[dtype]
    elif isinstance(dtype, str):
        if parameter_kinds.get(dtype) != "DType":
            raise TileTypeError(
                "internal error: canonical provider schema for "
                f"{decl_sym_name} uses {dtype!r} as an element dtype, but "
                "that constant parameter is not declared as DType"
            )
        try:
            dtype = bindings[dtype]
        except KeyError as exc:
            raise TileTypeError(
                "internal error: canonical provider schema for "
                f"{decl_sym_name} references unbound dtype parameter {dtype!r}"
            ) from exc

    shape = []
    for extent in tile_type.shape:
        if isinstance(extent, str):
            if parameter_kinds.get(extent) != "Dim":
                raise TileTypeError(
                    "internal error: canonical provider schema for "
                    f"{decl_sym_name} uses {extent!r} as an extent, but "
                    "that constant parameter is not declared as Dim"
                )
            try:
                extent = bindings[extent]
            except KeyError as exc:
                raise TileTypeError(
                    "internal error: canonical provider schema for "
                    f"{decl_sym_name} references unbound dimension parameter "
                    f"{extent!r}"
                ) from exc
        if isinstance(extent, bool) or not isinstance(extent, numbers.Integral):
            raise TileTypeError(
                "internal error: canonical provider schema for "
                f"{decl_sym_name} resolved a non-integral extent {extent!r}"
            )
        shape.append(int(extent))
    return dtype, tuple(shape)


def _collapsed_provider_dtype(dtype):
    """Return the signless dtype identity exposed by the TileIR provider ABI.

    Integer widths determine ABI compatibility here. Separate dtype-token
    constants retain signedness for native semantics and generated C++ types;
    this comparison must not replace those tokens.
    """

    token = _DTYPE_TO_TOKEN.get(dtype)
    if token is None:
        raise TileTypeError(
            "internal error: canonical provider type validation encountered "
            f"unsupported dtype {dtype!r}"
        )
    if token[:1] in {"i", "u"} and token[1:].isdigit():
        return f"i{token[1:]}"
    return token


def _validate_provider_decl_types(
    *,
    decl_sym_name: str,
    role: str,
    actual_types,
    decl_types,
    bindings: Mapping[str, object],
    parameter_kinds: Mapping[str, str],
) -> None:
    """Check concrete operands or results against the generated ABI schema.

    Resolve symbolic dimensions and dtypes from call constants, then compare
    shapes and signless ABI element types. The caller checks arity before this
    function. A mismatch is an internal lowering error, not a request to cast
    the user's payload to fit a different declaration.
    """

    for index, (actual_type, decl_type) in enumerate(
        zip(actual_types, decl_types, strict=True)
    ):
        expected_dtype, expected_shape = _resolve_provider_decl_tile_type(
            decl_sym_name=decl_sym_name,
            tile_type=decl_type,
            bindings=bindings,
            parameter_kinds=parameter_kinds,
        )
        actual_shape = tuple(actual_type.shape)
        dtype_matches = _collapsed_provider_dtype(
            actual_type.dtype
        ) == _collapsed_provider_dtype(expected_dtype)
        if not dtype_matches or actual_shape != expected_shape:
            raise TileTypeError(
                f"internal error: provider {role} {index} for {decl_sym_name} "
                f"has concrete type ({actual_type.dtype!r}, {actual_shape!r}); "
                "canonical schema and call constants resolve to "
                f"({expected_dtype!r}, {expected_shape!r})"
            )


def _emit_provider_foreign_call(
    *,
    operands,
    output_types,
    decl_sym_name: str,
    static_prefix: tuple[object, ...],
):
    """Validate a declaration and append its preview foreign operation.

    Resolve the TileLib identity and include environment before emission.
    The generated access annotations determine where output placeholders
    are needed and where input values also supply results. An in-place
    operand and its result must have identical Python Tile types.

    Parameters
    ----------
    operands : sequence of IR variable
        Input and input/output operands, in declaration order with pure
        output positions omitted.
    output_types : sequence of TileTy
        Types of the data results, including results of input/output operands.
    decl_sym_name : str
        Symbol in the generated canonical declaration table.
    static_prefix : tuple of object
        Constant arguments in the declaration's exact order. Logical extents
        and signed dtype tokens remain distinct from physical worker choices.

    Returns
    -------
    IR variable
        The single data result, or a cuTile tuple containing multiple results.
        The final effect token remains part of the IR operation and is omitted
        from the public result.

    Raises
    ------
    TileTypeError
        The TileLib, headers, preview support, or declaration contract cannot
        be resolved. No foreign operation is added before schema validation.
    """

    _ensure_provider_include_paths()
    _, tile_library_version, tile_library_path = _coop_tile_linker_input()
    try:
        declaration = _PROVIDER_DECL_SCHEMAS[decl_sym_name]
    except KeyError as exc:
        raise TileTypeError(
            f"internal error: no canonical provider schema for {decl_sym_name}"
        ) from exc
    if sum(mode != "out" for mode in declaration.access) != len(operands):
        raise TileTypeError(
            "internal error: provider declaration operands do not match the "
            f"canonical schema for {decl_sym_name}"
        )
    if len(declaration.results) != len(output_types):
        raise TileTypeError(
            "internal error: provider declaration results do not match the "
            f"canonical schema for {decl_sym_name}"
        )

    inputs = iter(operands)
    results = iter(output_types)
    call_operands = []
    for mode in declaration.access:
        if mode == "in":
            operand = next(inputs)
        else:
            output_type = next(results)
            if mode == "out":
                operand = _const(output_type.shape, 0, output_type.dtype)
            else:
                operand = next(inputs)
                operand_type = operand.get_type()
                if (
                    operand_type.dtype != output_type.dtype
                    or operand_type.shape != output_type.shape
                ):
                    raise TileTypeError(
                        "internal error: provider output/inout "
                        "operand and result types differ"
                    )
        call_operands.append(operand)
    call_operands = tuple(call_operands)

    call_const_args = _provider_call_const_args(
        const_params=declaration.constants,
        static_prefix=static_prefix,
    )
    parameter_kinds = dict(declaration.constants)
    decl_input_types = declaration.operands
    decl_output_types = declaration.results
    _validate_provider_decl_types(
        decl_sym_name=decl_sym_name,
        role="operand",
        actual_types=tuple(operand.get_type() for operand in call_operands),
        decl_types=decl_input_types,
        bindings=call_const_args,
        parameter_kinds=parameter_kinds,
    )
    _validate_provider_decl_types(
        decl_sym_name=decl_sym_name,
        role="result",
        actual_types=output_types,
        decl_types=decl_output_types,
        bindings=call_const_args,
        parameter_kinds=parameter_kinds,
    )

    preview_foreign_call_type = _preview_foreign_call_type()
    operation_kwargs = {
        "tilelibs": ((tile_library_path, tile_library_version),),
        "symbol_name": decl_sym_name,
        "constant_params": tuple(static_prefix),
        "inputs": call_operands,
        "token": None,
    }
    outputs = add_operation_variadic(
        preview_foreign_call_type,
        (*output_types, TokenTy()),
        **operation_kwargs,
    )
    # cuTile orders memory effects through the final token result. Only data
    # results are part of the public cooperative primitive's return value.
    data_outputs = outputs[:-1]
    if len(data_outputs) == 1:
        return data_outputs[0]
    return build_tuple(tuple(data_outputs))


def exclusive_sum_impl(value, prefix_value, algorithm, methods):
    """Lower Exclusive Sum while preserving prefix presence separately.

    Normalize constant methods and validate a concrete block configuration.
    A dynamic prefix uses a validation-only placeholder, then becomes a
    rank-zero operand. Its presence chooses the CUB overload; its value is
    not part of the declaration's specialization constants.
    """

    tile_ty = require_tile_type(value)
    shape = tile_ty.shape

    if tile_ty.ndim != 1:
        raise TileTypeError("coop.exclusive_sum expects a 1D tile")

    algorithm_val = require_optional_constant_str(algorithm)
    methods_val = _require_optional_constant_object(methods, "methods")
    _reject_runtime_method_overlap(
        methods_val, prefix_value, "prefix_value", "exclusive_sum"
    )
    prefix_value_val = _const_value_or_runtime(prefix_value)
    prefix_is_runtime = prefix_value_val is _RUNTIME_VALUE
    options = BlockScanExclusiveSumOptions.from_kwargs(
        # Presence selects the CUB overload, but the value itself is always a
        # runtime operand.  Use a validation-only placeholder for a dynamic
        # value; it never reaches provider metadata or artifact identity.
        prefix_value=0 if prefix_is_runtime else prefix_value_val,
        algorithm=algorithm_val,
        methods=methods_val,
    )

    numel = shape[0]
    decision = dispatch_exclusive_sum(
        numel=numel,
        dtype=tile_ty.dtype,
        requested_algorithm=options.algorithm,
    )
    items_per_thread = numel // decision.block_threads

    BlockScanExclusiveSum(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
        algorithm=decision.algorithm,
        prefix_value=options.prefix_value,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    initial_operand = _constant_rank_zero_tile(0, tile_ty.dtype)
    prefix_present = prefix_is_runtime or options.prefix_value is not None
    prefix_operand = (
        _constant_rank_zero_tile(options.prefix_value, tile_ty.dtype)
        if not _is_provided(prefix_value) and options.prefix_value is not None
        else _rank0_scalar_tile_operand(
            prefix_value,
            tile_ty.dtype,
            primitive="exclusive_sum",
            name="prefix_value",
            default=0,
            numeric_value=True,
        )[0]
    )
    return _emit_provider_foreign_call(
        operands=(value, initial_operand, prefix_operand),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_SCAN,
        static_prefix=_scan_static_prefix(
            numel,
            tile_ty.dtype,
            mode="exclusive",
            algorithm=decision.algorithm,
            scan_op="sum",
            initial_present=False,
            prefix_present=prefix_present,
        ),
    )


def inclusive_sum_impl(value, prefix_value, algorithm, methods):
    """Lower Inclusive Sum with an optional runtime prefix operand.

    The initial-value operand is an unused zero placeholder. Prefix presence
    is a declaration constant, while the actual prefix remains an operand.
    Return a Tile with the input shape and dtype.
    """

    tile_ty = require_tile_type(value)
    shape = tile_ty.shape

    if tile_ty.ndim != 1:
        raise TileTypeError("coop.inclusive_sum expects a 1D tile")

    algorithm_val = require_optional_constant_str(algorithm)
    methods_val = _require_optional_constant_object(methods, "methods")
    _reject_runtime_method_overlap(
        methods_val, prefix_value, "prefix_value", "inclusive_sum"
    )
    prefix_value_val = _const_value_or_runtime(prefix_value)
    prefix_is_runtime = prefix_value_val is _RUNTIME_VALUE
    options = BlockScanInclusiveSumOptions.from_kwargs(
        prefix_value=0 if prefix_is_runtime else prefix_value_val,
        algorithm=algorithm_val,
        methods=methods_val,
    )

    numel = shape[0]
    decision = dispatch_inclusive_sum(
        numel=numel,
        dtype=tile_ty.dtype,
        requested_algorithm=options.algorithm,
    )
    items_per_thread = numel // decision.block_threads

    BlockScanInclusiveSum(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
        algorithm=decision.algorithm,
        prefix_value=options.prefix_value,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    initial_operand = _constant_rank_zero_tile(0, tile_ty.dtype)
    prefix_present = prefix_is_runtime or options.prefix_value is not None
    prefix_operand = (
        _constant_rank_zero_tile(options.prefix_value, tile_ty.dtype)
        if not _is_provided(prefix_value) and options.prefix_value is not None
        else _rank0_scalar_tile_operand(
            prefix_value,
            tile_ty.dtype,
            primitive="inclusive_sum",
            name="prefix_value",
            default=0,
            numeric_value=True,
        )[0]
    )
    return _emit_provider_foreign_call(
        operands=(value, initial_operand, prefix_operand),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_SCAN,
        static_prefix=_scan_static_prefix(
            numel,
            tile_ty.dtype,
            mode="inclusive",
            algorithm=decision.algorithm,
            scan_op="sum",
            initial_present=False,
            prefix_present=prefix_present,
        ),
    )


def exclusive_scan_impl(
    value,
    scan_op,
    initial_value,
    prefix_value,
    algorithm,
    methods,
):
    """Lower Exclusive Scan with a constant operator and runtime seed.

    The option adapter checks the relationship between initial and prefix
    values, including required seeds for non-sum operators. Dynamic controls
    use placeholders only during those checks. Their real values become
    rank-zero operands; presence flags select the provider overload.
    """

    tile_ty = require_tile_type(value)
    shape = tile_ty.shape

    if tile_ty.ndim != 1:
        raise TileTypeError("coop.exclusive_scan expects a 1D tile")

    scan_op_val = require_optional_constant_str(scan_op)
    algorithm_val = require_optional_constant_str(algorithm)
    methods_val = _require_optional_constant_object(methods, "methods")
    _reject_runtime_method_overlap(
        methods_val, initial_value, "initial_value", "exclusive_scan"
    )
    _reject_runtime_method_overlap(
        methods_val, prefix_value, "prefix_value", "exclusive_scan"
    )
    initial_val = _const_value_or_runtime(initial_value)
    prefix_value_val = _const_value_or_runtime(prefix_value)
    initial_is_runtime = initial_val is _RUNTIME_VALUE
    prefix_is_runtime = prefix_value_val is _RUNTIME_VALUE
    options = BlockScanExclusiveScanOptions.from_kwargs(
        scan_op=scan_op_val,
        initial_value=0 if initial_is_runtime else initial_val,
        prefix_value=0 if prefix_is_runtime else prefix_value_val,
        algorithm=algorithm_val,
        methods=methods_val,
    )

    numel = shape[0]
    decision = dispatch_exclusive_scan(
        numel=numel,
        dtype=tile_ty.dtype,
        requested_algorithm=options.algorithm,
    )
    items_per_thread = numel // decision.block_threads

    BlockScanExclusiveScan(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
        scan_op=options.scan_op,
        initial_value=options.initial_value,
        prefix_value=options.prefix_value,
        algorithm=decision.algorithm,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    initial_present = initial_is_runtime or options.initial_value is not None
    prefix_present = prefix_is_runtime or options.prefix_value is not None
    initial_operand = (
        _constant_rank_zero_tile(options.initial_value, tile_ty.dtype)
        if not _is_provided(initial_value) and options.initial_value is not None
        else _rank0_scalar_tile_operand(
            initial_value,
            tile_ty.dtype,
            primitive="exclusive_scan",
            name="initial_value",
            default=0,
            numeric_value=True,
        )[0]
    )
    prefix_operand = (
        _constant_rank_zero_tile(options.prefix_value, tile_ty.dtype)
        if not _is_provided(prefix_value) and options.prefix_value is not None
        else _rank0_scalar_tile_operand(
            prefix_value,
            tile_ty.dtype,
            primitive="exclusive_scan",
            name="prefix_value",
            default=0,
            numeric_value=True,
        )[0]
    )
    return _emit_provider_foreign_call(
        operands=(value, initial_operand, prefix_operand),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_SCAN,
        static_prefix=_scan_static_prefix(
            numel,
            tile_ty.dtype,
            mode="exclusive",
            algorithm=decision.algorithm,
            scan_op=_scan_op_token(options.scan_op),
            initial_present=initial_present,
            prefix_present=prefix_present,
        ),
    )


def inclusive_scan_impl(
    value,
    scan_op,
    initial_value,
    prefix_value,
    algorithm,
    methods,
):
    """Lower Inclusive Scan after rejecting an initial-value seed.

    The option adapter permits a prefix and a built-in scan operator. Pass
    the actual prefix as a rank-zero operand and its presence as a constant;
    the unused initial operand is zero. Preserve the input shape and dtype.
    """

    tile_ty = require_tile_type(value)
    shape = tile_ty.shape

    if tile_ty.ndim != 1:
        raise TileTypeError("coop.inclusive_scan expects a 1D tile")

    scan_op_val = require_optional_constant_str(scan_op)
    algorithm_val = require_optional_constant_str(algorithm)
    methods_val = _require_optional_constant_object(methods, "methods")
    _reject_runtime_method_overlap(
        methods_val, initial_value, "initial_value", "inclusive_scan"
    )
    _reject_runtime_method_overlap(
        methods_val, prefix_value, "prefix_value", "inclusive_scan"
    )
    initial_val = _const_value_or_runtime(initial_value)
    prefix_value_val = _const_value_or_runtime(prefix_value)
    initial_is_runtime = initial_val is _RUNTIME_VALUE
    prefix_is_runtime = prefix_value_val is _RUNTIME_VALUE
    options = BlockScanInclusiveScanOptions.from_kwargs(
        scan_op=scan_op_val,
        initial_value=0 if initial_is_runtime else initial_val,
        prefix_value=0 if prefix_is_runtime else prefix_value_val,
        algorithm=algorithm_val,
        methods=methods_val,
    )

    numel = shape[0]
    decision = dispatch_inclusive_scan(
        numel=numel,
        dtype=tile_ty.dtype,
        requested_algorithm=options.algorithm,
    )
    items_per_thread = numel // decision.block_threads

    BlockScanInclusiveScan(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
        scan_op=options.scan_op,
        initial_value=options.initial_value,
        prefix_value=options.prefix_value,
        algorithm=decision.algorithm,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    prefix_present = prefix_is_runtime or options.prefix_value is not None
    initial_operand = _constant_rank_zero_tile(0, tile_ty.dtype)
    prefix_operand = (
        _constant_rank_zero_tile(options.prefix_value, tile_ty.dtype)
        if not _is_provided(prefix_value) and options.prefix_value is not None
        else _rank0_scalar_tile_operand(
            prefix_value,
            tile_ty.dtype,
            primitive="inclusive_scan",
            name="prefix_value",
            default=0,
            numeric_value=True,
        )[0]
    )
    return _emit_provider_foreign_call(
        operands=(value, initial_operand, prefix_operand),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_SCAN,
        static_prefix=_scan_static_prefix(
            numel,
            tile_ty.dtype,
            mode="inclusive",
            algorithm=decision.algorithm,
            scan_op=_scan_op_token(options.scan_op),
            initial_present=False,
            prefix_present=prefix_present,
        ),
    )


def scan_impl(
    value,
    mode,
    scan_op,
    initial_value,
    prefix_value,
    algorithm,
    methods,
):
    mode_val = require_optional_constant_str(mode)
    if mode_val is None:
        raise TileTypeError(
            "coop.scan mode must be a compile-time constant string"
        )
    if mode_val == "exclusive":
        return exclusive_scan_impl(
            value,
            scan_op,
            initial_value,
            prefix_value,
            algorithm,
            methods,
        )
    if mode_val == "inclusive":
        return inclusive_scan_impl(
            value,
            scan_op,
            initial_value,
            prefix_value,
            algorithm,
            methods,
        )
    raise TileTypeError("coop.scan mode must be one of: exclusive, inclusive")


def _reduce_axes(value, axis, keepdims, broadcast):
    """Normalize axes and arrange independent reductions as contiguous rows."""
    tile_ty = require_tile_type(value)
    shape = tile_ty.shape
    if _is_const_none(axis):
        axes = tuple(range(tile_ty.ndim))
    else:
        axes = tuple(
            sorted(
                normalize_axis(a, tile_ty.ndim, axis)
                for a in require_constant_int_tuple(axis, allow_single_int=True)
            )
        )
        for previous, current in pairwise(axes):
            if previous == current:
                raise TileTypeError(f"Repeated reduction axis {current}")
    retain_dims = require_constant_bool(keepdims)
    broadcast_result = require_constant_bool(broadcast)
    retained = tuple(i for i in range(tile_ty.ndim) if i not in axes)
    retained_shape = tuple(shape[i] for i in retained)
    kept_shape = tuple(1 if i in axes else size for i, size in enumerate(shape))
    result_shape = kept_shape if retain_dims else retained_shape
    order = retained + axes
    if order != tuple(range(tile_ty.ndim)):
        value = permute(value, order)
    return (
        value,
        axes,
        prod(retained_shape),
        prod(shape[i] for i in axes),
        result_shape,
        kept_shape,
        shape if broadcast_result else None,
    )


def _reduce_impl(value, axis, keepdims, broadcast, *, options):
    is_sum = (
        isinstance(options, BlockReduceSumOptions) or options.binary_op == "+"
    )
    tile_ty = require_tile_type(value)
    # Match ct.sum's one widening rule. Other supported dtypes accumulate
    # in their input type, including the generic addition spelling.
    if is_sum and tile_ty.dtype == datatype.bool_:
        value = astype(value, datatype.int32)
        tile_ty = require_tile_type(value)
    (
        value,
        axes,
        batches,
        reduction_size,
        result_shape,
        kept_shape,
        broadcast_shape,
    ) = _reduce_axes(value, axis, keepdims, broadcast)
    dispatch = dispatch_sum if is_sum else dispatch_reduce
    decision = dispatch(
        numel=reduction_size,
        dtype=tile_ty.dtype,
        requested_algorithm=options.algorithm,
    )
    kwargs = {
        "items_per_thread": (reduction_size + decision.block_threads - 1)
        // decision.block_threads,
        "dtype": tile_ty.dtype,
        "block_threads": decision.block_threads,
        "algorithm": decision.algorithm,
    }
    if is_sum:
        BlockReduceSum(**kwargs)
    else:
        BlockReduce(binary_op=options.binary_op, **kwargs)
    # An empty axis tuple (including axis=None on a scalar) is a no-op.
    # Still validate operator/dtype/options so invalid calls cannot hide here.
    if not axes:
        return value
    numel = batches * reduction_size
    if is_sum:
        declaration = _PROVIDER_DECL_SUM
        prefix = _reduce_sum_static_prefix(
            numel,
            tile_ty.dtype,
            algorithm=decision.algorithm,
        )
    else:
        declaration = _PROVIDER_DECL_REDUCE
        prefix = _reduce_generic_static_prefix(
            numel,
            tile_ty.dtype,
            algorithm=decision.algorithm,
            binary_op=_reduce_op_token(options.binary_op),
        )
    result = _emit_provider_foreign_call(
        operands=(reshape(value, (numel,)),),
        output_types=(make_tile_ty(tile_ty.dtype, (batches,)),),
        decl_sym_name=declaration,
        static_prefix=(*prefix, batches),
    )
    if broadcast_shape is not None:
        return broadcast_to(reshape(result, kept_shape), broadcast_shape)
    return reshape(result, result_shape)


def sum_impl(value, axis, keepdims, algorithm, methods, broadcast):
    options = BlockReduceSumOptions.from_kwargs(
        algorithm=require_optional_constant_str(algorithm),
        methods=_require_optional_constant_object(methods, "methods"),
    )
    return _reduce_impl(value, axis, keepdims, broadcast, options=options)


def reduce_impl(
    value, axis, keepdims, binary_op, algorithm, methods, broadcast
):
    options = BlockReduceOptions.from_kwargs(
        binary_op=require_optional_constant_str(binary_op),
        algorithm=require_optional_constant_str(algorithm),
        methods=_require_optional_constant_object(methods, "methods"),
    )
    return _reduce_impl(value, axis, keepdims, broadcast, options=options)


def exchange_striped_to_blocked_impl(value, block_threads, methods):
    """Lower the flat striped-to-blocked permutation at its explicit width.

    Fold a directly preceding inverse Exchange only when its width matches.
    Otherwise emit the opposite CUB direction because the foreign-call
    fragment adapter itself uses blocked storage. The width remains a
    semantic declaration constant.
    """

    _require_empty_methods(methods, "exchange_striped_to_blocked")
    tile_ty = require_tile_type(value)
    shape = tile_ty.shape
    if tile_ty.ndim != 1:
        raise TileTypeError(
            "coop.exchange_striped_to_blocked expects a 1D tile"
        )

    requested_threads = _require_optional_constant_object(
        block_threads, "coop.exchange_striped_to_blocked block_threads"
    )
    numel = shape[0]
    decision = dispatch_exchange_striped_to_blocked(
        numel=numel,
        dtype=tile_ty.dtype,
        block_threads=requested_threads,
    )
    items_per_thread = numel // decision.block_threads

    origin_kind, origin_op = _find_exchange_origin(value)
    if origin_kind == "blocked_to_striped" and _exchange_has_topology(
        origin_op, decision.block_threads
    ):
        return _foreign_call_first_input(origin_op)

    # Foreign-call copy paths currently require unit atom strides. The
    # opposite CUB primitive is intentionally load-bearing here: it preserves
    # the user-facing flat exchange direction after the blocked layout adapter.
    BlockExchangeBlockedToStriped(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    return _emit_provider_foreign_call(
        operands=(value,),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_EXCHANGE,
        static_prefix=_exchange_static_prefix(
            numel,
            tile_ty.dtype,
            block_threads=decision.block_threads,
            mode="blocked_to_striped",
        ),
    )


def exchange_blocked_to_striped_impl(value, block_threads, methods):
    """Lower the flat blocked-to-striped permutation at its explicit width.

    A directly preceding inverse with the same width returns its original
    input. Otherwise emit the inverse CUB direction required by the blocked
    fragment adapter, preserving the input Tile's shape and dtype.
    """

    _require_empty_methods(methods, "exchange_blocked_to_striped")
    tile_ty = require_tile_type(value)
    shape = tile_ty.shape
    if tile_ty.ndim != 1:
        raise TileTypeError(
            "coop.exchange_blocked_to_striped expects a 1D tile"
        )

    requested_threads = _require_optional_constant_object(
        block_threads, "coop.exchange_blocked_to_striped block_threads"
    )
    numel = shape[0]
    decision = dispatch_exchange_blocked_to_striped(
        numel=numel,
        dtype=tile_ty.dtype,
        block_threads=requested_threads,
    )
    items_per_thread = numel // decision.block_threads

    origin_kind, origin_op = _find_exchange_origin(value)
    if origin_kind == "striped_to_blocked" and _exchange_has_topology(
        origin_op, decision.block_threads
    ):
        return _foreign_call_first_input(origin_op)

    BlockExchangeStripedToBlocked(
        items_per_thread=items_per_thread,
        dtype=tile_ty.dtype,
        block_threads=decision.block_threads,
    )

    out_ty = make_tile_ty(tile_ty.dtype, shape)
    return _emit_provider_foreign_call(
        operands=(value,),
        output_types=(out_ty,),
        decl_sym_name=_PROVIDER_DECL_EXCHANGE,
        static_prefix=_exchange_static_prefix(
            numel,
            tile_ty.dtype,
            block_threads=decision.block_threads,
            mode="striped_to_blocked",
        ),
    )
