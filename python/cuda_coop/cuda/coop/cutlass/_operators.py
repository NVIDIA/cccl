# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in operator normalization shared by CUTLASS Reduce and Scan."""

import operator
from enum import Enum

import numpy as np

from ._compiler._types import INTEGER_VALUE_TYPES

OPERATOR_CPP = {
    "sum": "::cuda::std::plus<T>",
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}
_ALIASES = {
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_CALLABLE_ALIASES = {
    operator.add: "sum",
    operator.mul: "multiplies",
    operator.and_: "bit_and",
    operator.or_: "bit_or",
    operator.xor: "bit_xor",
    np.add: "sum",
    np.multiply: "multiplies",
    np.minimum: "min",
    np.maximum: "max",
    np.bitwise_and: "bit_and",
    np.bitwise_or: "bit_or",
    np.bitwise_xor: "bit_xor",
}


def normalize_operator(value, *, primitive="reduce"):
    if value is None:
        return "sum"
    if isinstance(value, Enum):
        raise TypeError(f"cuda.coop.cutlass.{primitive} operator must be a built-in")
    if isinstance(value, str):
        token = value.strip().lower().replace("-", "_")
        try:
            return _ALIASES[token]
        except KeyError:
            raise ValueError(
                f"cuda.coop.cutlass.{primitive} operator must be one of: "
                + ", ".join(OPERATOR_CPP)
            ) from None
    for known, op in _CALLABLE_ALIASES.items():
        if value is known:
            return op
    raise NotImplementedError(
        f"cuda.coop.cutlass.{primitive} supports built-in operators; "
        "custom callbacks are not supported"
    )


def validate_operator_dtype(op, value_type, *, primitive="reduce"):
    if op not in OPERATOR_CPP:
        raise ValueError(f"unsupported built-in operator {op!r}")
    if op.startswith("bit_") and value_type not in INTEGER_VALUE_TYPES:
        raise TypeError(f"cuda.coop.cutlass.{primitive} {op} requires an integer dtype")


def operator_expression(op):
    return OPERATOR_CPP[op].replace("<T>", "<>") + "{}"
