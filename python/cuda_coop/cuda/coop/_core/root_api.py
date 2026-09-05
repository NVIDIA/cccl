# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light, backend-dispatching implementation of the common API."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from numbers import Integral
from types import ModuleType
from typing import Any, Protocol, TypeVar, runtime_checkable

from .block import (
    BlockAdjacentDifferenceDirection,
    BlockDiscontinuityMode,
    BlockShuffleMode,
    validate_block_histogram_output_capacity,
)
from .dtype_policy import (
    validate_common_v1_integer_key_dtype_name,
    validate_common_v1_integer_value_dtype_name,
    validate_common_v1_numeric_dtype_name,
)
from .thread_group import (
    CoopCompilerContextRequiredError,
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
)
from .thread_group import (
    this_block as _core_this_block,
)
from .thread_group import (
    this_cluster as _core_this_cluster,
)
from .thread_group import (
    this_grid as _core_this_grid,
)
from .thread_group import (
    this_thread as _core_this_thread,
)
from .thread_group import (
    this_warp as _core_this_warp,
)

_ACTIVE_BACKEND_MODULE: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_backend_module",
    default=None,
)
_QUALIFIED_BACKEND_MODULE: str | None = None
_ACTIVE_COMMON_ROOT_OPERATION: ContextVar[str | None] = ContextVar(
    "cuda_coop_active_common_root_operation",
    default=None,
)
_GROUP_OPERATIONS = (
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "scan",
    "shuffle",
    "store",
    "sum",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
)

_BLOCK_AND_WARP_GROUPS = ("block", "warp", "threads_within_warp")
_COMMON_V1_MAX_EXCHANGE_ITEMS_PER_THREAD = 5
_REDUCTION_GROUPS = (
    "thread",
    "warp",
    "threads_within_warp",
    "block",
    "cluster",
)
_BLOCK_ONLY = ("block",)
_COMMON_V1_OPERATION_GROUPS = {
    "load": _BLOCK_AND_WARP_GROUPS,
    "store": _BLOCK_AND_WARP_GROUPS,
    "reduce": _REDUCTION_GROUPS,
    "sum": _REDUCTION_GROUPS,
    "scan": _BLOCK_AND_WARP_GROUPS,
    "exclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "inclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "exclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "inclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "exchange": _BLOCK_AND_WARP_GROUPS,
    "adjacent_difference": _BLOCK_ONLY,
    "discontinuity": _BLOCK_ONLY,
    "shuffle": _BLOCK_ONLY,
    "merge_sort_keys": _BLOCK_AND_WARP_GROUPS,
    "merge_sort_pairs": _BLOCK_AND_WARP_GROUPS,
    "radix_sort_keys": _BLOCK_ONLY,
    "radix_sort_pairs": _BLOCK_ONLY,
    "radix_rank": _BLOCK_ONLY,
    "histogram": _BLOCK_ONLY,
    "run_length_decode": _BLOCK_ONLY,
    "topk_max_keys": _BLOCK_ONLY,
    "topk_max_pairs": _BLOCK_ONLY,
    "topk_min_keys": _BLOCK_ONLY,
    "topk_min_pairs": _BLOCK_ONLY,
}
_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)
_REDUCE_ALGORITHMS = frozenset({"raking_commutative_only", "raking", "warp_reductions"})
_SCAN_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})
_SCAN_MODES = frozenset({"exclusive", "inclusive"})
_EXCHANGE_MODES = frozenset({"striped_to_blocked", "blocked_to_striped"})
_ADJACENT_DIFFERENCE_DIRECTIONS = frozenset({"left", "right"})
_DISCONTINUITY_MODES = frozenset({"heads", "tails"})
_SHUFFLE_MODES = frozenset({"down", "up"})
_HISTOGRAM_ALGORITHMS = frozenset({"atomic", "sort"})

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class ThreadDataLike(Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by certified backends."""

    items_per_thread: int
    dtype: Any | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit cooperative scratch descriptor understood by certified backends."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: str


class UnsupportedCoopBackendOperationError(NotImplementedError):
    """The selected compiler backend does not implement a root operation."""

    def __init__(self, backend_module: str, operation: str) -> None:
        self.backend_module = backend_module
        self.operation = operation
        self.reason_code = "cuda-coop-backend-operation-unavailable"
        super().__init__(
            f"cuda.coop.{operation} is not implemented by {backend_module!r}"
        )


@contextmanager
def _compiler_scope(backend_module: str) -> Iterator[None]:
    """Activate one backend for the current compiler trace."""

    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")

    token = _ACTIVE_BACKEND_MODULE.set(backend_module)
    try:
        yield
    finally:
        _ACTIVE_BACKEND_MODULE.reset(token)


def _register_qualified_backend(backend_module: str) -> None:
    """Use one imported qualified backend outside compiler-owned scopes."""

    if not isinstance(backend_module, str):
        raise TypeError("backend_module must be a string")
    if not backend_module.strip():
        raise ValueError("backend_module must be a non-empty string")

    global _QUALIFIED_BACKEND_MODULE
    if _QUALIFIED_BACKEND_MODULE not in {None, backend_module}:
        raise RuntimeError(
            "cuda.coop common-root fallback is already activated by "
            f"{_QUALIFIED_BACKEND_MODULE!r}; cannot also activate "
            f"{backend_module!r}"
        )
    _QUALIFIED_BACKEND_MODULE = backend_module


def _backend_module_name() -> str | None:
    return _ACTIVE_BACKEND_MODULE.get() or _QUALIFIED_BACKEND_MODULE


@contextmanager
def _common_root_operation_scope(operation: str) -> Iterator[None]:
    """Identify one root dispatch without changing backend selection."""

    token = _ACTIVE_COMMON_ROOT_OPERATION.set(operation)
    try:
        yield
    finally:
        _ACTIVE_COMMON_ROOT_OPERATION.reset(token)


def _common_root_operation_name() -> str | None:
    """Return the common-root operation currently delegated to a backend."""

    return _ACTIVE_COMMON_ROOT_OPERATION.get()


def _active_backend(feature: str) -> tuple[str, ModuleType]:
    module_name = _backend_module_name()
    if module_name is None:
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{feature} requires a Python DSL compiler context "
            "(compiler-owned activation) or a qualified backend import before "
            "compilation; for example, import "
            "cuda.coop.cutlass or cuda.coop.numba_mlir"
        )
    return module_name, import_module(module_name)


def _backend_member(name: str) -> Any:
    module_name, backend = _active_backend(name)
    try:
        return getattr(backend, name)
    except AttributeError as exc:
        raise UnsupportedCoopBackendOperationError(module_name, name) from exc


def _portable_selector(
    operation: str,
    parameter: str,
    value: Any,
    allowed: frozenset[str],
    *,
    allow_none: bool = False,
) -> Any:
    """Normalize one common selector while a tracing backend is active."""

    if _backend_module_name() is None:
        return value
    if value is None and allow_none:
        return None
    token = getattr(value, "value", value)
    if isinstance(token, str):
        token = token.strip().lower().replace("-", "_")
    if token not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"cuda.coop.{operation} {parameter} must be one of: {choices}; "
            "use a backend-qualified import for backend-only controls"
        )
    return token


def _common_v1_group_name(kind: str) -> str:
    """Return the public profile spelling for one internal group kind."""

    return "physical_warp" if kind == "warp" else kind


def _validate_common_v1_operation_group(
    operation: str,
    group: Any,
) -> None:
    """Enforce the common V1 support matrix for a common-root call."""

    if not isinstance(group, ThreadGroup):
        return
    supported = _COMMON_V1_OPERATION_GROUPS[operation]
    if group.kind in supported:
        return
    group_name = _common_v1_group_name(group.kind)
    supported_names = ", ".join(map(_common_v1_group_name, supported))
    raise NotImplementedError(
        f"cuda.coop.{operation} does not support group kind {group_name!r} in "
        f"common V1; supported group kinds: {supported_names}; use a "
        "backend-qualified import for backend-specific group support"
    )


def _validate_common_thread_data_payload(
    operation: str,
    parameter: str,
    value: Any,
) -> None:
    """Require the portable fixed-size payload representation."""

    if isinstance(value, ThreadDataLike):
        return
    raise TypeError(
        f"cuda.coop.{operation} requires a fixed-size ThreadData {parameter} "
        "payload; use a backend-qualified import for backend-specific scalar "
        "or register payloads"
    )


def _common_thread_data_extent(
    operation: str,
    parameter: str,
    value: ThreadDataLike[Any],
) -> int:
    """Return one positive trace-static portable payload extent."""

    extent = value.items_per_thread
    if isinstance(extent, bool) or not isinstance(extent, Integral):
        raise TypeError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    normalized = int(extent)
    if normalized <= 0:
        raise ValueError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must be a "
            "compile-time positive integer"
        )
    if len(value) != normalized:
        raise ValueError(
            f"cuda.coop.{operation} {parameter}.items_per_thread must match "
            "the payload item count"
        )
    return normalized


def _common_payload_dtype(value: ThreadDataLike[Any]) -> Any:
    """Return the declared or item-inferred dtype for a portable payload."""

    dtype = value.dtype
    if dtype is None and len(value) > 0:
        item = value[0]
        dtype = getattr(item, "dtype", None)
        if dtype is None:
            dtype = type(item)
    return dtype


def _common_integer_dtype_name(dtype: Any) -> str:
    """Normalize a Python or structural compiler integer dtype to its name."""

    if dtype is int:
        return "int32"
    if dtype is float:
        return "float32"

    dtype_name = getattr(dtype, "name", None)
    if not isinstance(dtype_name, str):
        dtype_name = getattr(dtype, "__name__", None)
    if isinstance(dtype_name, str):
        dtype_name = dtype_name.lower()
        for prefix in ("int", "uint", "float", "complex"):
            if dtype_name.startswith(prefix) and dtype_name[len(prefix) :].isdigit():
                return dtype_name
        if dtype_name in {"bool", "boolean"}:
            return dtype_name

    width = getattr(dtype, "width", None)
    if width is None:
        width = getattr(dtype, "bitwidth", None)
    signed = getattr(dtype, "signed", None)
    if isinstance(width, Integral) and not isinstance(width, bool):
        if isinstance(signed, bool):
            prefix = "int" if signed else "uint"
            return f"{prefix}{int(width)}"

    if isinstance(dtype_name, str):
        return dtype_name
    return str(dtype).lower()


def _validate_common_run_length_decode_dtype(
    parameter: str,
    value: ThreadDataLike[Any],
    *,
    allow_uint8: bool,
) -> tuple[int, bool]:
    """Require one portable decode dtype and return its width and signedness."""

    dtype_name = _common_integer_dtype_name(_common_payload_dtype(value))
    validator = (
        validate_common_v1_integer_value_dtype_name
        if allow_uint8
        else validate_common_v1_integer_key_dtype_name
    )
    dtype_name = validator(
        dtype_name,
        operation="run_length_decode",
        parameter=parameter,
    )
    return int(
        dtype_name.removeprefix("u").removeprefix("int")
    ), not dtype_name.startswith("u")


def _is_compiler_integer(value: Any) -> bool:
    """Return whether a dynamic value exposes the portable integer protocol."""

    missing = object()
    width = getattr(value, "width", missing)
    signed = getattr(value, "signed", missing)
    dtype = getattr(value, "dtype", missing)
    ir_value = getattr(value, "ir_value", missing)
    return (
        isinstance(width, Integral)
        and not isinstance(width, bool)
        and width > 0
        and isinstance(signed, bool)
        and dtype is not missing
        and callable(ir_value)
    )


def _validate_common_merge_sort_oob_default(
    operation: str,
    keys: ThreadDataLike[Any],
    oob_default: Any,
) -> None:
    """Require a key-typed, losslessly representable partial-tile sentinel."""

    key_dtype_name = _common_integer_dtype_name(_common_payload_dtype(keys))
    if isinstance(oob_default, Integral) and not isinstance(oob_default, bool):
        value = int(oob_default)
        width = int(key_dtype_name.removeprefix("u").removeprefix("int"))
        if key_dtype_name.startswith("uint"):
            lower, upper = 0, (1 << width) - 1
        else:
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(
                f"cuda.coop.{operation} oob_default={value} is not "
                f"representable in keys dtype {key_dtype_name}"
            )

    if _is_compiler_integer(oob_default):
        oob_dtype_name = _common_integer_dtype_name(oob_default)
    else:
        oob_dtype_name = _common_integer_dtype_name(type(oob_default))
    if oob_dtype_name != key_dtype_name:
        raise TypeError(
            f"cuda.coop.{operation} oob_default must have the same integer "
            f"dtype as keys ({key_dtype_name}); got {oob_dtype_name}"
        )


def _validate_common_run_length_decode_controls(
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any,
    run_length_width: int,
    run_length_signed: bool,
) -> None:
    """Validate trace-static output shape and the portable window scalar."""

    if isinstance(decoded_items_per_thread, bool) or not isinstance(
        decoded_items_per_thread, Integral
    ):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_items_per_thread must be a "
            "compile-time positive integer"
        )
    if int(decoded_items_per_thread) <= 0:
        raise ValueError(
            "cuda.coop.run_length_decode decoded_items_per_thread must be a "
            "compile-time positive integer"
        )

    if isinstance(decoded_window_offset, bool):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_window_offset must be an "
            "int-like scalar"
        )
    if isinstance(decoded_window_offset, Integral):
        normalized_offset = int(decoded_window_offset)
        if normalized_offset < 0:
            raise ValueError(
                "cuda.coop.run_length_decode decoded_window_offset must be non-negative"
            )
        value_bits = run_length_width - 1 if run_length_signed else run_length_width
        if normalized_offset >= 1 << value_bits:
            raise ValueError(
                "cuda.coop.run_length_decode decoded_window_offset must be "
                "representable in the run_lengths dtype"
            )
        return
    if not _is_compiler_integer(decoded_window_offset):
        raise TypeError(
            "cuda.coop.run_length_decode decoded_window_offset must be an "
            "int-like scalar"
        )


def _validate_common_integer_key_dtype(
    operation: str,
    keys: ThreadDataLike[Any],
) -> int:
    """Require the backend-neutral integer-key profile and return its width."""

    dtype_name = validate_common_v1_integer_key_dtype_name(
        _common_integer_dtype_name(_common_payload_dtype(keys)),
        operation=operation,
    )
    return int(dtype_name.removeprefix("u").removeprefix("int"))


def _validate_common_numeric_payload_dtype(
    operation: str,
    parameter: str,
    value: ThreadDataLike[Any],
) -> None:
    """Require one portable numeric payload dtype."""

    validate_common_v1_numeric_dtype_name(
        _common_integer_dtype_name(_common_payload_dtype(value)),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_pair_payloads(
    operation: str,
    keys: Any,
    values: Any,
) -> tuple[int, int]:
    """Validate matching portable pair payloads and return key width/extent."""

    _validate_common_thread_data_payload(operation, "keys", keys)
    _validate_common_thread_data_payload(operation, "values", values)
    key_extent = _common_thread_data_extent(operation, "keys", keys)
    value_extent = _common_thread_data_extent(operation, "values", values)
    if key_extent != value_extent:
        raise ValueError(
            f"cuda.coop.{operation} keys and values must have matching items_per_thread"
        )
    key_width = _validate_common_integer_key_dtype(operation, keys)
    _validate_common_numeric_payload_dtype(operation, "value", values)
    return key_width, key_extent


def _validate_common_integer_dtype(
    operation: str,
    parameter: str,
    dtype: Any,
    *,
    allow_uint8: bool,
) -> None:
    """Require one portable integral dtype used by a specialized operation."""

    validator = (
        validate_common_v1_integer_value_dtype_name
        if allow_uint8
        else validate_common_v1_integer_key_dtype_name
    )
    validator(
        _common_integer_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_radix_sort_controls(
    operation: str,
    *,
    key_width: int,
    begin_bit: Any,
    end_bit: Any | None,
    descending: Any,
) -> None:
    """Validate the statically visible portion of portable radix controls."""

    if not isinstance(descending, bool):
        raise TypeError(f"cuda.coop.{operation} descending must be a compile-time bool")

    def static_bound(name: str, value: Any) -> int | None:
        if isinstance(value, bool):
            raise TypeError(f"cuda.coop.{operation} {name} must be an int-like scalar")
        if isinstance(value, Integral):
            return int(value)
        if _is_compiler_integer(value):
            return None
        raise TypeError(f"cuda.coop.{operation} {name} must be an int-like scalar")

    static_begin = static_bound("begin_bit", begin_bit)
    resolved_end = key_width if end_bit is None else end_bit
    static_end = static_bound("end_bit", resolved_end)
    if static_begin is not None and static_begin < 0:
        raise ValueError(f"cuda.coop.{operation} begin_bit must be non-negative")
    if static_begin is not None and static_begin >= key_width:
        raise ValueError(f"cuda.coop.{operation} begin_bit must be < {key_width}")
    if static_end is not None and static_end > key_width:
        raise ValueError(f"cuda.coop.{operation} end_bit must be <= {key_width}")
    if static_end is not None and static_end < 1:
        raise ValueError(f"cuda.coop.{operation} end_bit must be positive")
    if (
        static_begin is not None
        and static_end is not None
        and static_end <= static_begin
    ):
        raise ValueError(
            f"cuda.coop.{operation} end_bit must be greater than begin_bit"
        )


def _validate_common_topk_controls(
    operation: str,
    *,
    group: ThreadGroup,
    keys: Any,
    values: Any = None,
    k: Any,
    valid_items: Any | None,
    begin_bit: Any,
    end_bit: Any | None,
) -> None:
    """Validate the statically visible portable TopK contract."""

    if values is None:
        _validate_common_thread_data_payload(operation, "keys", keys)
        items_per_thread = _common_thread_data_extent(operation, "keys", keys)
        key_width = _validate_common_integer_key_dtype(operation, keys)
    else:
        key_width, items_per_thread = _validate_common_pair_payloads(
            operation, keys, values
        )

    if isinstance(group, ThreadGroup):
        hierarchy = group.hierarchy
        assert hierarchy is not None
        block_dim = hierarchy.block_dim
        if block_dim is not None and block_dim[1:] != (1, 1):
            raise ValueError(f"cuda.coop.{operation} requires a one-dimensional block")
        if group.static_size is not None and group.static_size > 1024:
            raise ValueError(
                f"cuda.coop.{operation} block thread count must be <= 1024"
            )

    def static_int(name: str, value: Any) -> int | None:
        if isinstance(value, bool):
            raise TypeError(f"cuda.coop.{operation} {name} must be an int-like scalar")
        if isinstance(value, Integral):
            return int(value)
        if _is_compiler_integer(value):
            return None
        raise TypeError(f"cuda.coop.{operation} {name} must be an int-like scalar")

    static_k = static_int("k", k)
    if static_k is not None and static_k <= 0:
        raise ValueError(f"cuda.coop.{operation} k must be positive")

    tile_size = None
    if isinstance(group, ThreadGroup) and group.static_size is not None:
        tile_size = group.static_size * items_per_thread

    if valid_items is None:
        static_valid_items = tile_size
    else:
        static_valid_items = static_int("valid_items", valid_items)
        if static_valid_items is not None and static_valid_items <= 0:
            raise ValueError(f"cuda.coop.{operation} valid_items must be positive")
        if (
            static_valid_items is not None
            and tile_size is not None
            and static_valid_items > tile_size
        ):
            raise ValueError(
                f"cuda.coop.{operation} valid_items must be <= tile size {tile_size}"
            )

    if (
        static_k is not None
        and static_valid_items is not None
        and static_k > static_valid_items
    ):
        raise ValueError(f"cuda.coop.{operation} k must be <= valid_items")

    static_begin = static_int("begin_bit", begin_bit)
    resolved_end = key_width if end_bit is None else end_bit
    static_end = static_int("end_bit", resolved_end)
    if static_begin is not None and static_begin < 0:
        raise ValueError(f"cuda.coop.{operation} begin_bit must be non-negative")
    if static_begin is not None and static_begin >= key_width:
        raise ValueError(f"cuda.coop.{operation} begin_bit must be < {key_width}")
    if static_end is not None and static_end > key_width:
        raise ValueError(f"cuda.coop.{operation} end_bit must be <= {key_width}")
    if static_end is not None and static_end < 1:
        raise ValueError(f"cuda.coop.{operation} end_bit must be positive")
    if (
        static_begin is not None
        and static_end is not None
        and static_end <= static_begin
    ):
        raise ValueError(
            f"cuda.coop.{operation} end_bit must be greater than begin_bit"
        )


def _validate_common_radix_rank_controls(
    *,
    key_width: int,
    begin_bit: Any,
    end_bit: Any | None,
    radix_bits: Any | None,
    descending: Any,
) -> None:
    """Validate the trace-static portable BlockRadixRank interval."""

    if not isinstance(descending, bool):
        raise TypeError("cuda.coop.radix_rank descending must be a compile-time bool")

    def static_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"cuda.coop.radix_rank {name} must be a compile-time integer"
            )
        return int(value)

    begin = static_int("begin_bit", begin_bit)
    if begin < 0:
        raise ValueError("cuda.coop.radix_rank begin_bit must be non-negative")
    if begin >= key_width:
        raise ValueError(f"cuda.coop.radix_rank begin_bit must be < {key_width}")

    width = 4 if radix_bits is None else static_int("radix_bits", radix_bits)
    if width <= 0:
        raise ValueError("cuda.coop.radix_rank radix_bits must be positive")
    if end_bit is None:
        end = begin + width
    else:
        end = static_int("end_bit", end_bit)
        if radix_bits is not None and end != begin + width:
            raise ValueError(
                "cuda.coop.radix_rank radix_bits must match end_bit - begin_bit"
            )
        width = end - begin

    if end <= begin:
        raise ValueError("cuda.coop.radix_rank end_bit must be greater than begin_bit")
    if end > key_width:
        raise ValueError(f"cuda.coop.radix_rank end_bit must be <= {key_width}")
    if width > 8:
        raise ValueError("cuda.coop.radix_rank bit width must be <= 8")


def _group_constructor(
    name: str,
    fallback: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _backend_module_name() is None:
        return fallback(*args, **kwargs)
    return _backend_member(name)(*args, **kwargs)


def this_thread() -> ThreadGroup:
    """Describe the current thread."""

    return _group_constructor("this_thread", _core_this_thread)


def this_warp() -> ThreadGroup:
    """Describe the current physical warp."""

    return _group_constructor("this_warp", _core_this_warp)


def this_block() -> ThreadGroup:
    """Describe the current CTA."""

    return _group_constructor("this_block", _core_this_block)


def this_cluster() -> ThreadGroup:
    """Describe the current cluster."""

    return _group_constructor("this_cluster", _core_this_cluster)


def this_grid() -> ThreadGroup:
    """Describe the current grid."""

    group = _group_constructor("this_grid", _core_this_grid)
    if _backend_module_name() is not None and isinstance(group, ThreadGroup):
        assert group.hierarchy is not None
        return group.with_hierarchy(group.hierarchy, source="common_root")
    return group


def ThreadData(items_per_thread: int, dtype: Any = None) -> ThreadDataLike[Any]:
    """Construct the selected backend's per-thread payload container."""

    with _common_root_operation_scope("ThreadData"):
        return _backend_member("ThreadData")(items_per_thread, dtype=dtype)


def TempStorage(
    size_in_bytes: Any = None,
    alignment: Any = None,
    auto_sync: Any = None,
    sharing: str = "shared",
) -> TempStorageLike:
    """Construct the selected backend's explicit scratch override."""

    return _backend_member("TempStorage")(
        size_in_bytes=size_in_bytes,
        alignment=alignment,
        auto_sync=auto_sync,
        sharing=sharing,
    )


def _group_primitive_marker(
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if _backend_module_name() is None:
        del args, kwargs
        raise CoopCompilerContextRequiredError(
            f"cuda.coop.{operation} requires a Python DSL compiler context "
            "(compiler-owned activation) or a qualified backend import before "
            "compilation; for example, import "
            "cuda.coop.cutlass or cuda.coop.numba_mlir"
        )
    _validate_common_v1_operation_group(operation, args[0] if args else None)
    with _common_root_operation_scope(operation):
        return _backend_member(operation)(*args, **kwargs)


def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadDataLike[Any]:
    """Load values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "load", "algorithm", algorithm, _LOAD_STORE_ALGORITHMS
    )

    return _group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "store", "algorithm", algorithm, _LOAD_STORE_ALGORITHMS
    )

    _group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


def reduce(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    binary_op: Any = None,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Reduce values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "reduce",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "reduce",
        group,
        value,
        binary_op=binary_op,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    broadcast: bool = True,
    valid_items: Any = None,
    algorithm: Any = None,
) -> Any:
    """Sum values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "sum",
        "algorithm",
        algorithm,
        _REDUCE_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "sum",
        group,
        value,
        broadcast=broadcast,
        valid_items=valid_items,
        algorithm=algorithm,
    )


def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Scan values across a group through the compiler-selected backend.

    ``mode="exclusive"`` returns a fully defined prefix: the default sum uses
    zero when ``initial_value`` is omitted, while every other ``scan_op``
    requires an explicit ``initial_value``. ``mode="inclusive"`` does not
    accept ``initial_value``.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("scan", "mode", mode, _SCAN_MODES)
    algorithm = _portable_selector(
        "scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an exclusive prefix sum through the compiler-selected backend.

    The first flattened output position is zero.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "exclusive_sum",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "exclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an inclusive prefix sum through the compiler-selected backend.

    Every output position is defined.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "inclusive_sum",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "inclusive_sum",
        group,
        value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an exclusive scan through the compiler-selected backend.

    The default sum uses zero when ``initial_value`` is omitted. Every other
    ``scan_op`` requires an explicit ``initial_value`` so that the first
    flattened output position is defined.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "exclusive_scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "exclusive_scan",
        group,
        value,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return an inclusive scan through the compiler-selected backend.

    Every output position is defined; inclusive scans do not accept an initial
    value in the portable profile.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "inclusive_scan",
        "algorithm",
        algorithm,
        _SCAN_ALGORITHMS,
        allow_none=True,
    )

    return _group_primitive_marker(
        "inclusive_scan",
        group,
        value,
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


def exchange(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "striped_to_blocked",
) -> Any:
    """Exchange values across a group through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("exchange", "mode", mode, _EXCHANGE_MODES)

    return _group_primitive_marker(
        "exchange",
        group,
        value,
        mode=mode,
    )


def adjacent_difference(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    direction: Any = BlockAdjacentDifferenceDirection.LEFT,
    valid_items: Any = None,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Compute groupwise differences through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    direction = _portable_selector(
        "adjacent_difference",
        "direction",
        direction,
        _ADJACENT_DIFFERENCE_DIRECTIONS,
    )

    return _group_primitive_marker(
        "adjacent_difference",
        group,
        value,
        direction=direction,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


def discontinuity(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockDiscontinuityMode.HEADS,
    tile_predecessor_item: Any = None,
    tile_successor_item: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Compute groupwise discontinuities through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    mode = _portable_selector("discontinuity", "mode", mode, _DISCONTINUITY_MODES)

    return _group_primitive_marker(
        "discontinuity",
        group,
        value,
        mode=mode,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
        temp_storage=temp_storage,
    )


def shuffle(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: Any = BlockShuffleMode.DOWN,
    distance: Any = 1,
) -> Any:
    """Unit-shift a block payload through the compiler-selected backend.

    The common profile accepts only a fixed-size per-thread payload, ``up`` or
    ``down`` mode, and the unit distance ``1``. The vacated first item for
    ``up`` or last item for ``down`` is undefined. Use the qualified
    ``cuda.coop.<backend>`` API for scalar, boundary-output, or other-distance
    behavior.
    """

    mode = _portable_selector("shuffle", "mode", mode, _SHUFFLE_MODES)
    if _backend_module_name() is not None:
        normalized_distance = getattr(distance, "value", distance)
        if (
            isinstance(normalized_distance, bool)
            or not isinstance(normalized_distance, Integral)
            or int(normalized_distance) != 1
        ):
            raise ValueError(
                "cuda.coop.shuffle distance must be exactly 1 in common V1; "
                "use a backend-qualified import for other distances"
            )
        distance = 1

    return _group_primitive_marker(
        "shuffle",
        group,
        value,
        mode=mode,
        distance=distance,
    )


def merge_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Return merge-sorted keys through the compiler-selected backend.

    Complete physical blocks and warps accept fixed-size integral ``ThreadData``
    and return a shape-preserving payload without mutating ``keys``. A block
    must contain a power-of-two number of threads. For a partial tile, provide
    ``valid_items`` and ``oob_default`` together. The sentinel must sort after
    every valid key: greater for ascending order and less for descending order.
    Only the valid sorted prefix is defined.

    Use the qualified ``cuda.coop.<backend>`` API for custom comparators,
    backend-specific payloads, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("merge_sort_keys", group)
        _validate_common_thread_data_payload("merge_sort_keys", "keys", keys)
        _validate_common_integer_key_dtype("merge_sort_keys", keys)
        if (valid_items is None) != (oob_default is None):
            raise ValueError(
                "cuda.coop.merge_sort_keys valid_items and oob_default must be "
                "provided together"
            )
        if oob_default is not None:
            _validate_common_merge_sort_oob_default(
                "merge_sort_keys", keys, oob_default
            )

    return _group_primitive_marker(
        "merge_sort_keys",
        group,
        keys,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def merge_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    descending: bool = False,
    valid_items: Any = None,
    oob_default: Any = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Return merge-sorted pairs through the compiler-selected backend.

    Complete physical blocks and warps accept matching fixed-size ``ThreadData``
    payloads. Keys use the portable integral-key profile and values use the
    portable numeric profile. Values remain attached to their keys, equal-key
    ordering is unspecified, and neither input is mutated. For a partial tile,
    provide ``valid_items`` and a key-typed ``oob_default`` together; only the
    valid sorted prefix is defined.

    Use the qualified ``cuda.coop.<backend>`` API for custom comparators,
    scalar/register payloads, or backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("merge_sort_pairs", group)
        _validate_common_pair_payloads("merge_sort_pairs", keys, values)
        if (valid_items is None) != (oob_default is None):
            raise ValueError(
                "cuda.coop.merge_sort_pairs valid_items and oob_default must "
                "be provided together"
            )
        if oob_default is not None:
            _validate_common_merge_sort_oob_default(
                "merge_sort_pairs", keys, oob_default
            )

    return _group_primitive_marker(
        "merge_sort_pairs",
        group,
        keys,
        values,
        descending=descending,
        valid_items=valid_items,
        oob_default=oob_default,
        temp_storage=temp_storage,
    )


def radix_sort_keys(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> Any:
    """Return radix-sorted integral keys through the compiler-selected backend.

    ``group`` must be a complete physical block and ``keys`` must be a
    fixed-size ``ThreadData`` payload of Python, NumPy, or compiler 32- or
    64-bit signed or unsigned integers. ``begin_bit`` and ``end_bit`` select a
    half-open interval in CUB's bit-ordered key representation. Omitting
    ``end_bit`` selects the key width, including when ``begin_bit`` is nonzero.
    The returned payload preserves the input item type and item count without
    mutating ``keys``.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads,
    striped output, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("radix_sort_keys", group)
        _validate_common_thread_data_payload("radix_sort_keys", "keys", keys)
        key_width = _validate_common_integer_key_dtype("radix_sort_keys", keys)
        _validate_common_radix_sort_controls(
            "radix_sort_keys",
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )

    return _group_primitive_marker(
        "radix_sort_keys",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


def radix_sort_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    descending: bool = False,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Return radix-sorted pairs through the compiler-selected backend.

    ``group`` is a complete physical block. Matching ``ThreadData`` payloads
    preserve both item types and key/value association without mutation.
    ``begin_bit`` and ``end_bit`` select the same portable bit interval as
    :func:`radix_sort_keys`.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads,
    striped output, launch controls, or other backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("radix_sort_pairs", group)
        key_width, _ = _validate_common_pair_payloads("radix_sort_pairs", keys, values)
        _validate_common_radix_sort_controls(
            "radix_sort_pairs",
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )

    return _group_primitive_marker(
        "radix_sort_pairs",
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        temp_storage=temp_storage,
    )


def radix_rank(
    group: ThreadGroup,
    keys: Any,
    /,
    *,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    radix_bits: Any | None = None,
    descending: bool = False,
) -> Any:
    """Return stable ranks through the compiler-selected backend.

    ``group`` must be a complete physical block and ``keys`` must be a
    fixed-size ``ThreadData`` payload of Python, NumPy, or compiler 32- or
    64-bit signed or unsigned integers. The digit is extracted from CUB's
    bit-ordered key representation, so signed keys are ordered by value rather
    than by their raw two's-complement sign bit. The selected half-open
    interval defaults to four bits starting at ``begin_bit`` and may contain at
    most eight bits. Equal digits retain flattened blocked input order. The
    returned payload contains shape-preserving signed 32-bit ranks and does not
    mutate ``keys``.

    Use the qualified ``cuda.coop.<backend>`` API for scalar/register payloads
    or an exclusive digit-prefix side output.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("radix_rank", group)
        _validate_common_thread_data_payload("radix_rank", "keys", keys)
        key_width = _validate_common_integer_key_dtype("radix_rank", keys)
        _validate_common_radix_rank_controls(
            key_width=key_width,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=radix_bits,
            descending=descending,
        )

    return _group_primitive_marker(
        "radix_rank",
        group,
        keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
    )


def histogram(
    group: ThreadGroup,
    samples: Any,
    /,
    *,
    bins: Any,
    bins_per_thread: Any = 1,
    counter_dtype: Any = None,
    algorithm: Any = "atomic",
) -> Any:
    """Compute a groupwise histogram through the compiler-selected backend.

    ``samples`` is a compiler-produced, fixed-size per-thread payload. Each
    member receives ``bins_per_thread`` striped counters for bin indices
    ``rank + i * group_size``; slots beyond ``bins`` are zero. The static
    projection must have enough capacity for all bins. Portable V1 certifies
    uint8/int32/uint32/int64/uint64 samples and int32/uint32/int64/uint64
    counters; the Python ``int`` dtype spelling maps to int32. Every sample
    must satisfy CUB's ``0 <= sample < bins`` precondition; violating it is
    undefined behavior.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("histogram", group)
        _validate_common_thread_data_payload("histogram", "samples", samples)
        _validate_common_integer_dtype(
            "histogram",
            "sample",
            _common_payload_dtype(samples),
            allow_uint8=True,
        )
        if counter_dtype is not None:
            _validate_common_integer_dtype(
                "histogram",
                "counter",
                counter_dtype,
                allow_uint8=False,
            )
        if (
            isinstance(group, ThreadGroup)
            and group.kind == "block"
            and group.static_size is not None
        ):
            validate_block_histogram_output_capacity(
                bins=bins,
                bins_per_thread=bins_per_thread,
                block_threads=group.static_size,
                scope=f"{_backend_module_name()}.histogram",
            )

    algorithm = _portable_selector(
        "histogram", "algorithm", algorithm, _HISTOGRAM_ALGORITHMS
    )

    return _group_primitive_marker(
        "histogram",
        group,
        samples,
        bins=bins,
        bins_per_thread=bins_per_thread,
        counter_dtype=counter_dtype,
        algorithm=algorithm,
    )


def run_length_decode(
    group: ThreadGroup,
    run_values: Any,
    run_lengths: Any,
    /,
    *,
    decoded_items_per_thread: Any,
    decoded_window_offset: Any = 0,
) -> Any:
    """Decode run-length values through the compiler-selected backend.

    ``group`` must be a complete physical block. ``run_values`` and
    ``run_lengths`` are matching, positive-size ``ThreadData`` payloads whose
    runs are owned in blocked order. ``decoded_items_per_thread`` is a positive
    trace-static integer. Member ``rank`` receives decoded positions
    ``decoded_window_offset + rank * decoded_items_per_thread + i``. The
    uniform window offset must be nonnegative and representable in the
    run-length dtype; callers providing a dynamic offset guarantee that range.
    Positions beyond the decoded stream are zero.

    The result is a new ``ThreadData`` payload with
    ``decoded_items_per_thread`` values per member and the run-value dtype.
    Neither input is mutated. Actual runs must have positive lengths; zero
    lengths are permitted only as one trailing padding suffix, and the
    block-wide sum must be positive and representable in the run-length dtype.
    Use the qualified
    ``cuda.coop.<backend>`` API for relative offsets, total decoded size, or
    backend-specific payloads.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("run_length_decode", group)
        _validate_common_thread_data_payload(
            "run_length_decode", "run_values", run_values
        )
        _validate_common_thread_data_payload(
            "run_length_decode", "run_lengths", run_lengths
        )
        value_extent = _common_thread_data_extent(
            "run_length_decode", "run_values", run_values
        )
        length_extent = _common_thread_data_extent(
            "run_length_decode", "run_lengths", run_lengths
        )
        if value_extent != length_extent:
            raise ValueError(
                "cuda.coop.run_length_decode run_values and run_lengths must "
                "have matching items_per_thread"
            )
        _validate_common_run_length_decode_dtype(
            "run_values", run_values, allow_uint8=True
        )
        run_length_width, run_length_signed = _validate_common_run_length_decode_dtype(
            "run_lengths", run_lengths, allow_uint8=False
        )
        _validate_common_run_length_decode_controls(
            decoded_items_per_thread=decoded_items_per_thread,
            decoded_window_offset=decoded_window_offset,
            run_length_width=run_length_width,
            run_length_signed=run_length_signed,
        )

    return _group_primitive_marker(
        "run_length_decode",
        group,
        run_values,
        run_lengths,
        decoded_items_per_thread=decoded_items_per_thread,
        decoded_window_offset=decoded_window_offset,
    )


def topk_max_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select largest keys through the compiler-selected backend.

    Only the first ``k`` positions are defined. ``temp_storage`` supplies
    optional reusable scratch. Use the qualified ``cuda.coop.<backend>`` API
    for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("topk_max_keys", group)
        _validate_common_topk_controls(
            "topk_max_keys",
            group=group,
            keys=keys,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_max_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_min_keys(
    group: ThreadGroup,
    keys: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> Any:
    """Select smallest keys through the compiler-selected backend.

    Only the first ``k`` positions are defined. ``temp_storage`` supplies
    optional reusable scratch. Use the qualified ``cuda.coop.<backend>`` API
    for backend-specific behavior.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("topk_min_keys", group)
        _validate_common_topk_controls(
            "topk_min_keys",
            group=group,
            keys=keys,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_min_keys",
        group,
        keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_max_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select largest-key pairs through the compiler-selected backend.

    Only the first ``k`` flattened blocked pairs are defined. That prefix is
    unordered, each value remains attached to its key, and neither input is
    mutated. Use the qualified ``cuda.coop.<backend>`` API for backend-specific
    payloads. ``temp_storage`` supplies optional reusable scratch.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("topk_max_pairs", group)
        _validate_common_topk_controls(
            "topk_max_pairs",
            group=group,
            keys=keys,
            values=values,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_max_pairs",
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


def topk_min_pairs(
    group: ThreadGroup,
    keys: Any,
    values: Any,
    k: Any,
    /,
    *,
    valid_items: Any = None,
    begin_bit: Any = 0,
    end_bit: Any | None = None,
    temp_storage: Any = None,
) -> tuple[Any, Any]:
    """Select smallest-key pairs through the compiler-selected backend.

    Only the first ``k`` flattened blocked pairs are defined. That prefix is
    unordered, each value remains attached to its key, and neither input is
    mutated. Use the qualified ``cuda.coop.<backend>`` API for backend-specific
    payloads. ``temp_storage`` supplies optional reusable scratch.
    """

    if _backend_module_name() is not None:
        _validate_common_v1_operation_group("topk_min_pairs", group)
        _validate_common_topk_controls(
            "topk_min_pairs",
            group=group,
            keys=keys,
            values=values,
            k=k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    return _group_primitive_marker(
        "topk_min_pairs",
        group,
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        temp_storage=temp_storage,
    )


for _member_name in (
    "TempStorage",
    "ThreadData",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
):
    setattr(
        globals()[_member_name],
        "__cuda_coop_backend_member__",
        _member_name,
    )
del _member_name

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
]
