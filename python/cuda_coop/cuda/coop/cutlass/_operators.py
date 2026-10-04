# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Map supported operator names and known callables to C++ functors.

Callable aliases are recognized by identity; their Python bodies are never
compiled or invoked. The shared plan receives a typed functor descriptor,
while generated calls construct a functor with deduced operand types.
"""

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
    """Resolve a built-in string or known callable to one operator token.

    None selects sum. Compare callable objects by identity so an arbitrary
    callback cannot acquire built-in semantics merely by sharing a name.
    """

    if value is None:
        return "sum"
    if isinstance(value, Enum):
        raise TypeError(
            f"cuda.coop.cutlass.{primitive} operator must be a built-in"
        )
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
    """Require a known operator and integer values for bitwise operations."""

    if op not in OPERATOR_CPP:
        raise ValueError(f"unsupported built-in operator {op!r}")
    if op.startswith("bit_") and value_type not in INTEGER_VALUE_TYPES:
        raise TypeError(
            f"cuda.coop.cutlass.{primitive} {op} requires an integer dtype"
        )


def operator_expression(op):
    """Construct a C++ functor whose operand types are deduced."""

    return OPERATOR_CPP[op].replace("<T>", "<>") + "{}"
