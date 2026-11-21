# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass

from cuda.tile import _datatype as datatype
from cuda.tile._exception import TileTypeError

from ._base import BlockPrimitive

_SUPPORTED_DTYPES = {
    datatype.bfloat16,
    datatype.float16,
    datatype.float32,
    datatype.float64,
    datatype.int8,
    datatype.uint8,
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
}

_INT_DTYPES = {
    datatype.int8,
    datatype.uint8,
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
}

_SUPPORTED_ALGORITHMS = {
    None,
    "raking",
    "raking_memoize",
    "warp_scans",
}

_SCAN_OP_TO_KIND = {
    None: "sum",
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

_INCLUSIVE_SCAN_INITIAL_VALUE_ERROR = (
    "coop.inclusive_scan does not support initial_value; omit initial_value or "
    "use prefix_value for a block prefix"
)


def _normalize_dtype(dtype):
    if isinstance(dtype, datatype.DType):
        return dtype
    if hasattr(dtype, "dtype"):
        return dtype.dtype
    raise TileTypeError(f"Unsupported dtype {dtype}")


def _merge_methods(
    *,
    primitive: str,
    methods: dict | tuple | None,
    allowed_keys: tuple[str, ...],
    explicit_values: dict[str, object | None],
) -> dict[str, object]:
    """Merge constant method options without changing the caller's mapping.

    Accept a mapping or alternating key/value tuple, reject unknown keys,
    and overlay non-None explicit values. Repeated tuple keys keep the last
    value. The lowering separately rejects runtime arguments that would
    conflict with a method entry; runtime IR values do not belong here.
    """

    if methods is None:
        methods_dict: dict[str, object] = {}
    elif isinstance(methods, tuple):
        if len(methods) % 2 != 0:
            raise TileTypeError(
                f"coop.{primitive} methods tuple must have even length "
                "(key/value pairs)"
            )
        methods_dict = {}
        for i in range(0, len(methods), 2):
            key = methods[i]
            value = methods[i + 1]
            if not isinstance(key, str):
                raise TileTypeError(
                    f"coop.{primitive} methods tuple keys must be strings"
                )
            methods_dict[key] = value
    else:
        if not isinstance(methods, Mapping):
            raise TileTypeError(
                f"coop.{primitive} methods must be a dict or tuple "
                f"of key/value pairs"
            )
        methods_dict = dict(methods)

    unknown = sorted(k for k in methods_dict if k not in allowed_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise TileTypeError(
            f"coop.{primitive} methods contains unsupported keys: {joined}"
        )

    merged = dict(methods_dict)
    for key, value in explicit_values.items():
        if value is not None:
            merged[key] = value
    return merged


def _validate_algorithm(algorithm: str | None, *, primitive: str) -> None:
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise TileTypeError(
            f"coop.{primitive} algorithm must be one of: "
            "raking, raking_memoize, warp_scans"
        )


def _validate_numeric_value(
    value,
    *,
    primitive: str,
    arg_name: str,
    dtype: datatype.DType,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TileTypeError(
            f"coop.{primitive} {arg_name} must be a numeric constant"
        )
    if dtype in _INT_DTYPES and not isinstance(value, int):
        raise TileTypeError(
            f"coop.{primitive} {arg_name} must be an integer for integer dtypes"
        )


def _normalize_scan_op(scan_op: str | None, *, primitive: str) -> str:
    if scan_op not in _SCAN_OP_TO_KIND:
        raise TileTypeError(
            f"coop.{primitive} scan_op must be one of: +, *, min, max, &, |, ^"
        )
    if scan_op is None or scan_op == "sum":
        return "+"
    return scan_op


def _validate_scan_op_dtype(
    *,
    scan_op_kind: str,
    dtype: datatype.DType,
    primitive: str,
) -> None:
    if (
        scan_op_kind in {"bit_and", "bit_or", "bit_xor"}
        and dtype not in _INT_DTYPES
    ):
        raise TileTypeError(
            f"coop.{primitive} bitwise scan_op requires an integer dtype"
        )


class _BlockScanBase(BlockPrimitive):
    """Validate the dtype and algorithm shared by Scan variants.

    Subclasses check seed values and operator compatibility. Dynamic seeds
    are represented by safe constants during this phase; their actual values
    are passed to the provider separately.
    """

    def __init__(
        self,
        *,
        primitive: str,
        name: str,
        items_per_thread: int,
        dtype,
        block_threads: int,
        algorithm: str | None,
    ) -> None:
        super().__init__(name=name, block_threads=block_threads)
        if items_per_thread <= 0:
            raise TileTypeError("items_per_thread must be positive")
        self.primitive = primitive
        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        _validate_algorithm(algorithm, primitive=primitive)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.{primitive} does not support dtype {self.dtype}"
            )
        self.algorithm = algorithm


class BlockScanExclusiveSum(_BlockScanBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        algorithm: str | None = None,
        prefix_value: float | None = None,
    ) -> None:
        super().__init__(
            primitive="exclusive_sum",
            name="block_scan_exclusive_sum",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )
        _validate_numeric_value(
            prefix_value,
            primitive=self.primitive,
            arg_name="prefix_value",
            dtype=self.dtype,
        )
        self.prefix_value = prefix_value


class BlockScanInclusiveSum(_BlockScanBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        algorithm: str | None = None,
        prefix_value: float | None = None,
    ) -> None:
        super().__init__(
            primitive="inclusive_sum",
            name="block_scan_inclusive_sum",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )
        _validate_numeric_value(
            prefix_value,
            primitive=self.primitive,
            arg_name="prefix_value",
            dtype=self.dtype,
        )
        self.prefix_value = prefix_value


class BlockScanExclusiveScan(_BlockScanBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        scan_op: str | None = "+",
        initial_value: float | None = None,
        prefix_value: float | None = None,
        algorithm: str | None = None,
    ) -> None:
        super().__init__(
            primitive="exclusive_scan",
            name="block_scan_exclusive_scan",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )
        self.scan_op = _normalize_scan_op(scan_op, primitive=self.primitive)
        self.scan_op_kind = _SCAN_OP_TO_KIND[self.scan_op]
        _validate_scan_op_dtype(
            scan_op_kind=self.scan_op_kind,
            dtype=self.dtype,
            primitive=self.primitive,
        )
        _validate_numeric_value(
            initial_value,
            primitive=self.primitive,
            arg_name="initial_value",
            dtype=self.dtype,
        )
        _validate_numeric_value(
            prefix_value,
            primitive=self.primitive,
            arg_name="prefix_value",
            dtype=self.dtype,
        )
        if initial_value is not None and prefix_value is not None:
            raise TileTypeError(
                "coop.exclusive_scan does not support both initial_value and "
                "prefix_value"
            )
        if (
            initial_value is None
            and prefix_value is None
            and self.scan_op_kind != "sum"
        ):
            raise TileTypeError(
                "coop.exclusive_scan requires initial_value or "
                "prefix_value for "
                "a non-sum scan_op"
            )
        self.initial_value = initial_value
        self.prefix_value = prefix_value


class BlockScanInclusiveScan(_BlockScanBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        scan_op: str | None = "+",
        initial_value: float | None = None,
        prefix_value: float | None = None,
        algorithm: str | None = None,
    ) -> None:
        super().__init__(
            primitive="inclusive_scan",
            name="block_scan_inclusive_scan",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )
        self.scan_op = _normalize_scan_op(scan_op, primitive=self.primitive)
        self.scan_op_kind = _SCAN_OP_TO_KIND[self.scan_op]
        _validate_scan_op_dtype(
            scan_op_kind=self.scan_op_kind,
            dtype=self.dtype,
            primitive=self.primitive,
        )
        _validate_numeric_value(
            initial_value,
            primitive=self.primitive,
            arg_name="initial_value",
            dtype=self.dtype,
        )
        _validate_numeric_value(
            prefix_value,
            primitive=self.primitive,
            arg_name="prefix_value",
            dtype=self.dtype,
        )
        if initial_value is not None:
            raise TileTypeError(_INCLUSIVE_SCAN_INITIAL_VALUE_ERROR)
        self.initial_value = initial_value
        self.prefix_value = prefix_value


@dataclass(frozen=True)
class BlockScanExclusiveSumOptions:
    prefix_value: object | None = None
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        prefix_value: object | None,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockScanExclusiveSumOptions:
        merged = _merge_methods(
            primitive="exclusive_sum",
            methods=methods,
            allowed_keys=("algorithm", "prefix_value"),
            explicit_values={
                "algorithm": algorithm,
                "prefix_value": prefix_value,
            },
        )
        algorithm = merged.get("algorithm")
        prefix_value = merged.get("prefix_value")
        if prefix_value is not None and (
            isinstance(prefix_value, bool)
            or not isinstance(prefix_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.exclusive_sum prefix_value must be a numeric constant"
            )
        _validate_algorithm(algorithm, primitive="exclusive_sum")
        return cls(
            prefix_value=prefix_value,
            algorithm=algorithm,
            methods=merged or None,
        )


@dataclass(frozen=True)
class BlockScanInclusiveSumOptions:
    prefix_value: object | None = None
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        prefix_value: object | None,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockScanInclusiveSumOptions:
        merged = _merge_methods(
            primitive="inclusive_sum",
            methods=methods,
            allowed_keys=("algorithm", "prefix_value"),
            explicit_values={
                "algorithm": algorithm,
                "prefix_value": prefix_value,
            },
        )
        algorithm = merged.get("algorithm")
        prefix_value = merged.get("prefix_value")
        if prefix_value is not None and (
            isinstance(prefix_value, bool)
            or not isinstance(prefix_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.inclusive_sum prefix_value must be a numeric constant"
            )
        _validate_algorithm(algorithm, primitive="inclusive_sum")
        return cls(
            prefix_value=prefix_value,
            algorithm=algorithm,
            methods=merged or None,
        )


@dataclass(frozen=True)
class BlockScanExclusiveScanOptions:
    """Normalize the exclusive operator, seed presence, and algorithm.

    An initial value and prefix cannot both be supplied. A non-sum operator
    needs one of them. ``from_kwargs`` checks these relationships before
    dtype-specific validation and foreign-call emission.
    """

    scan_op: str = "+"
    initial_value: object | None = None
    prefix_value: object | None = None
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        scan_op: str | None,
        initial_value: object | None,
        prefix_value: object | None,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockScanExclusiveScanOptions:
        merged = _merge_methods(
            primitive="exclusive_scan",
            methods=methods,
            allowed_keys=(
                "scan_op",
                "initial_value",
                "prefix_value",
                "algorithm",
            ),
            explicit_values={
                "scan_op": scan_op,
                "initial_value": initial_value,
                "prefix_value": prefix_value,
                "algorithm": algorithm,
            },
        )
        scan_op = merged.get("scan_op")
        initial_value = merged.get("initial_value")
        prefix_value = merged.get("prefix_value")
        algorithm = merged.get("algorithm")
        scan_op = _normalize_scan_op(scan_op, primitive="exclusive_scan")
        if initial_value is not None and (
            isinstance(initial_value, bool)
            or not isinstance(initial_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.exclusive_scan initial_value must be a numeric constant"
            )
        if prefix_value is not None and (
            isinstance(prefix_value, bool)
            or not isinstance(prefix_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.exclusive_scan prefix_value must be a numeric constant"
            )
        if initial_value is not None and prefix_value is not None:
            raise TileTypeError(
                "coop.exclusive_scan does not support both initial_value and "
                "prefix_value"
            )
        if initial_value is None and prefix_value is None and scan_op != "+":
            raise TileTypeError(
                "coop.exclusive_scan requires initial_value or "
                "prefix_value for "
                "a non-sum scan_op"
            )
        _validate_algorithm(algorithm, primitive="exclusive_scan")
        return cls(
            scan_op=scan_op,
            initial_value=initial_value,
            prefix_value=prefix_value,
            algorithm=algorithm,
            methods=merged or None,
        )


@dataclass(frozen=True)
class BlockScanInclusiveScanOptions:
    """Normalize inclusive Scan options and reject initial-value seeds.

    A prefix is accepted; it is distinct from an exclusive initial value.
    The lowering preserves that distinction in the declaration's presence
    flags even for runtime controls.
    """

    scan_op: str = "+"
    initial_value: object | None = None
    prefix_value: object | None = None
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        scan_op: str | None,
        initial_value: object | None,
        prefix_value: object | None,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockScanInclusiveScanOptions:
        merged = _merge_methods(
            primitive="inclusive_scan",
            methods=methods,
            allowed_keys=(
                "scan_op",
                "initial_value",
                "prefix_value",
                "algorithm",
            ),
            explicit_values={
                "scan_op": scan_op,
                "initial_value": initial_value,
                "prefix_value": prefix_value,
                "algorithm": algorithm,
            },
        )
        scan_op = merged.get("scan_op")
        initial_value = merged.get("initial_value")
        prefix_value = merged.get("prefix_value")
        algorithm = merged.get("algorithm")
        scan_op = _normalize_scan_op(scan_op, primitive="inclusive_scan")
        if initial_value is not None and (
            isinstance(initial_value, bool)
            or not isinstance(initial_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.inclusive_scan initial_value must be a numeric constant"
            )
        if initial_value is not None:
            raise TileTypeError(_INCLUSIVE_SCAN_INITIAL_VALUE_ERROR)
        if prefix_value is not None and (
            isinstance(prefix_value, bool)
            or not isinstance(prefix_value, numbers.Real)
        ):
            raise TileTypeError(
                "coop.inclusive_scan prefix_value must be a numeric constant"
            )
        _validate_algorithm(algorithm, primitive="inclusive_scan")
        return cls(
            scan_op=scan_op,
            initial_value=initial_value,
            prefix_value=prefix_value,
            algorithm=algorithm,
            methods=merged or None,
        )
