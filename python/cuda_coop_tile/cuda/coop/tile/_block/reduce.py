# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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
    "raking_commutative_only",
    "raking",
    "warp_reductions",
}

_BINARY_OP_TO_KIND = {
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
            "raking_commutative_only, raking, warp_reductions"
        )


def _normalize_binary_op(binary_op: str | None) -> str:
    if binary_op not in _BINARY_OP_TO_KIND:
        raise TileTypeError(
            "coop.reduce binary_op must be one of: +, *, min, max, &, |, ^"
        )
    if binary_op is None or binary_op == "sum":
        return "+"
    return binary_op


def _validate_binary_op_dtype(
    *,
    binary_op_kind: str,
    dtype: datatype.DType,
) -> None:
    if (
        binary_op_kind in {"bit_and", "bit_or", "bit_xor"}
        and dtype not in _INT_DTYPES
    ):
        raise TileTypeError(
            "coop.reduce bitwise binary_op requires an integer dtype"
        )


class _BlockReduceBase(BlockPrimitive):
    """Validate common Reduce dtype, item count, and algorithm choices.

    The lowering supplies its candidate topology. Subclasses add operator
    checks; native selection later checks the chosen implementation width.
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
        _validate_algorithm(algorithm, primitive=primitive)
        self.primitive = primitive
        self.items_per_thread = items_per_thread
        self.dtype = _normalize_dtype(dtype)
        if self.dtype not in _SUPPORTED_DTYPES:
            raise TileTypeError(
                f"coop.{primitive} does not support dtype {self.dtype}"
            )
        self.algorithm = algorithm


class BlockReduceSum(_BlockReduceBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        algorithm: str | None = None,
    ) -> None:
        super().__init__(
            primitive="sum",
            name="block_reduce_sum",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )


class BlockReduce(_BlockReduceBase):
    def __init__(
        self,
        *,
        items_per_thread: int,
        dtype,
        block_threads: int = 128,
        binary_op: str | None = "+",
        algorithm: str | None = None,
    ) -> None:
        super().__init__(
            primitive="reduce",
            name="block_reduce",
            items_per_thread=items_per_thread,
            dtype=dtype,
            block_threads=block_threads,
            algorithm=algorithm,
        )
        self.binary_op = _normalize_binary_op(binary_op)
        self.binary_op_kind = _BINARY_OP_TO_KIND[self.binary_op]
        _validate_binary_op_dtype(
            binary_op_kind=self.binary_op_kind,
            dtype=self.dtype,
        )
        if self.binary_op_kind == "sum":
            raise TileTypeError("coop.reduce binary_op '+' should use coop.sum")


@dataclass(frozen=True)
class BlockReduceSumOptions:
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockReduceSumOptions:
        merged = _merge_methods(
            primitive="sum",
            methods=methods,
            allowed_keys=("algorithm",),
            explicit_values={"algorithm": algorithm},
        )
        algorithm = merged.get("algorithm")
        _validate_algorithm(algorithm, primitive="sum")
        return cls(algorithm=algorithm, methods=merged or None)


@dataclass(frozen=True)
class BlockReduceOptions:
    """Hold normalized built-in operator and algorithm options.

    ``from_kwargs`` combines explicit constants with method entries and
    checks their names. Dtype-dependent operator checks run when the lowering
    constructs ``BlockReduce``.
    """

    binary_op: str = "+"
    algorithm: str | None = None
    methods: dict | tuple | None = None

    @classmethod
    def from_kwargs(
        cls,
        *,
        binary_op: str | None,
        algorithm: str | None,
        methods: dict | tuple | None,
    ) -> BlockReduceOptions:
        merged = _merge_methods(
            primitive="reduce",
            methods=methods,
            allowed_keys=("binary_op", "algorithm"),
            explicit_values={
                "binary_op": binary_op,
                "algorithm": algorithm,
            },
        )
        binary_op = merged.get("binary_op")
        algorithm = merged.get("algorithm")
        normalized_binary_op = _normalize_binary_op(binary_op)
        _validate_algorithm(algorithm, primitive="reduce")
        return cls(
            binary_op=normalized_binary_op,
            algorithm=algorithm,
            methods=merged or None,
        )
