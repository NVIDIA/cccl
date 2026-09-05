# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared payload and dtype validation for portable root calls.

Family frontends use these import-light helpers before delegating to a compiler
backend. The validators define the conservative portable contract and do not
infer backend-specific types or construct lowering plans.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..dtype_policy import (
    validate_portable_integer_key_dtype_name,
    validate_portable_integer_value_dtype_name,
    validate_portable_numeric_dtype_name,
)
from ..thread_group import ThreadGroup

_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})

_ItemT = TypeVar("_ItemT")


@runtime_checkable
class _ReadableThreadDataLike(Protocol[_ItemT]):
    """Readable fixed-size per-thread payload understood by supported backends."""

    items_per_thread: int
    dtype: Any | None

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _ItemT: ...


@runtime_checkable
class ThreadDataLike(_ReadableThreadDataLike[_ItemT], Protocol[_ItemT]):
    """Mutable fixed-size per-thread payload understood by supported backends."""

    def __setitem__(self, index: int, value: _ItemT) -> None: ...


@runtime_checkable
class TempStorageLike(Protocol):
    """Explicit cooperative scratch descriptor understood by supported backends."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool | None
    sharing: str


def _validate_common_temp_storage(operation: str, value: Any) -> None:
    """Require the portable explicit temporary-storage representation."""

    if isinstance(value, TempStorageLike):
        return
    raise TypeError(
        f"cuda.coop.{operation} temp_storage must satisfy TempStorageLike; "
        "construct it with cuda.coop.TempStorage()"
    )


def _validate_common_thread_data_payload(
    operation: str,
    parameter: str,
    value: Any,
    *,
    allow_readonly: bool = False,
) -> None:
    """Require the portable fixed-size payload representation."""

    protocol = _ReadableThreadDataLike if allow_readonly else ThreadDataLike
    if isinstance(value, protocol):
        return
    raise TypeError(
        f"cuda.coop.{operation} requires a fixed-size ThreadData {parameter} "
        "payload; use a backend-qualified import for backend-specific scalar "
        "or register payloads"
    )


def _common_thread_data_extent(
    operation: str,
    parameter: str,
    value: _ReadableThreadDataLike[Any],
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


def _common_payload_dtype(
    operation: str,
    parameter: str,
    value: _ReadableThreadDataLike[Any],
) -> Any:
    """Return the declared or item-inferred dtype for a portable payload."""

    dtype = value.dtype
    if dtype is None and len(value) > 0:
        item = value[0]
        dtype = getattr(item, "dtype", None)
        if dtype is None:
            dtype = type(item)
        dtype_name = _common_numeric_dtype_name(dtype)
        for index in range(1, len(value)):
            item = value[index]
            item_dtype = getattr(item, "dtype", None)
            if item_dtype is None:
                item_dtype = type(item)
            if _common_numeric_dtype_name(item_dtype) != dtype_name:
                raise TypeError(
                    f"cuda.coop.{operation} {parameter} items must have one "
                    "common dtype"
                )
    return dtype


def _common_numeric_dtype_name(dtype: Any) -> str:
    """Normalize a Python, NumPy, or structural compiler numeric dtype."""

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
            suffix = dtype_name[len(prefix) :] if dtype_name.startswith(prefix) else ""
            if suffix.isdigit():
                return dtype_name
        if dtype_name in {"bool", "boolean"}:
            return dtype_name

    width = getattr(dtype, "width", None)
    if width is None:
        width = getattr(dtype, "bitwidth", None)
    signed = getattr(dtype, "signed", None)
    if isinstance(width, Integral) and not isinstance(width, bool):
        if isinstance(signed, bool):
            return f"{'int' if signed else 'uint'}{int(width)}"

    if isinstance(dtype_name, str):
        return dtype_name
    return str(dtype).lower()


def _common_integer_dtype_name(dtype: Any) -> str:
    """Normalize a Python or structural compiler integer dtype to its name."""

    return _common_numeric_dtype_name(dtype)


def _is_common_numeric_scalar(value: Any) -> bool:
    """Return whether ``value`` has the portable scalar representation."""

    if type(value) in {int, float}:
        return True
    value_type = type(value)
    if (value_type.__module__ or "").split(".", 1)[0] == "numpy":
        return value_type.__name__ in {
            "uint8",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        }
    width = getattr(value, "width", None)
    return (
        isinstance(width, Integral)
        and not isinstance(width, bool)
        and width > 0
        and getattr(value, "dtype", None) is not None
        and callable(getattr(value, "ir_value", None))
    )


def _validate_common_numeric_scalar(
    operation: str,
    parameter: str,
    value: Any,
) -> str:
    """Require one portable numeric scalar rather than a thread payload."""

    if not _is_common_numeric_scalar(value):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be a portable numeric "
            "scalar; use a backend-qualified import for backend-specific values"
        )
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = type(value)
    return validate_portable_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_integer_value(
    operation: str,
    parameter: str,
    value: Any,
) -> int | None:
    """Validate a static or compiler-owned portable integer value."""

    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if not (
        _is_common_numeric_scalar(value)
        and isinstance(getattr(value, "signed", None), bool)
    ):
        raise TypeError(
            f"cuda.coop.{operation} {parameter} must be a portable integer value"
        )
    dtype = getattr(value, "dtype", None)
    assert dtype is not None
    validate_portable_integer_value_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )
    return None


def _validate_common_numeric_value(
    operation: str,
    parameter: str,
    value: Any,
    *,
    allow_untyped_thread_data: bool = False,
    allow_readonly_thread_data: bool = False,
    require_thread_data: bool = False,
) -> str | None:
    """Require one portable scalar or fixed-size per-thread payload."""

    protocol = _ReadableThreadDataLike if allow_readonly_thread_data else ThreadDataLike
    if isinstance(value, protocol):
        _common_thread_data_extent(operation, parameter, value)
        if value.dtype is None and allow_untyped_thread_data:
            return
        dtype = _common_payload_dtype(operation, parameter, value)
    else:
        if require_thread_data:
            _validate_common_thread_data_payload(
                operation,
                parameter,
                value,
                allow_readonly=allow_readonly_thread_data,
            )
            raise AssertionError("unreachable")
        if not _is_common_numeric_scalar(value):
            raise TypeError(
                f"cuda.coop.{operation} requires a portable numeric scalar or "
                f"fixed-size ThreadData {parameter} payload; use a "
                "backend-qualified import for backend-specific payloads"
            )
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = type(value)
    return validate_portable_numeric_dtype_name(
        _common_numeric_dtype_name(dtype),
        operation=operation,
        parameter=parameter,
    )


def _validate_common_numeric_operator(
    operation: str,
    parameter: str,
    value: Any,
    operator: Any,
    *,
    allow_readonly_thread_data: bool = False,
) -> None:
    """Require an operator supported for the portable payload dtype."""

    dtype_name = _validate_common_numeric_value(
        operation,
        parameter,
        value,
        allow_readonly_thread_data=allow_readonly_thread_data,
    )
    assert dtype_name is not None
    if operator in _BITWISE_OPERATORS:
        validate_portable_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter=parameter,
        )


def _validate_common_run_length_decode_dtype(
    parameter: str,
    value: ThreadDataLike[Any],
    *,
    allow_uint8: bool,
) -> tuple[int, bool]:
    """Require one portable decode dtype and return its width and signedness."""

    dtype_name = _common_integer_dtype_name(
        _common_payload_dtype("run_length_decode", parameter, value)
    )
    validator = (
        validate_portable_integer_value_dtype_name
        if allow_uint8
        else validate_portable_integer_key_dtype_name
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
    """Require a matching typed integer or representable Python sentinel."""

    key_dtype_name = _common_integer_dtype_name(
        _common_payload_dtype(operation, "keys", keys)
    )
    # Only exact Python ints are untyped across both supported backends.
    if type(oob_default) is int:
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
        return

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

    dtype_name = validate_portable_integer_key_dtype_name(
        _common_integer_dtype_name(_common_payload_dtype(operation, "keys", keys)),
        operation=operation,
    )
    return int(dtype_name.removeprefix("u").removeprefix("int"))


def _validate_common_numeric_payload_dtype(
    operation: str,
    parameter: str,
    value: ThreadDataLike[Any],
) -> None:
    """Require one portable numeric payload dtype."""

    validate_portable_numeric_dtype_name(
        _common_numeric_dtype_name(_common_payload_dtype(operation, parameter, value)),
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
        validate_portable_integer_value_dtype_name
        if allow_uint8
        else validate_portable_integer_key_dtype_name
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


__all__ = ["TempStorageLike", "ThreadDataLike"]
