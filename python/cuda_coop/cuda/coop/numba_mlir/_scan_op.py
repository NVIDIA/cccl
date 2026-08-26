# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalize Python scan operators for CUB provider generation."""

import operator
from enum import Enum

import numpy as np

from ._semantic import _normalize_numba_callable


class ScanOpCategory(Enum):
    """Normalized category for block and warp scan operators."""

    SUM = "sum"
    KNOWN = "known"
    CALLABLE = "callable"


class ScanOp:
    """Describe a scan operator as a C++ functor or device callback."""

    _SUM_OPS = {"+", "add", "plus", "sum", np.add, operator.add}
    _KNOWN_OPS = {
        "mul": "::cuda::std::multiplies<T>",
        "multiply": "::cuda::std::multiplies<T>",
        "multiplies": "::cuda::std::multiplies<T>",
        "min": "::cuda::minimum<T>",
        "minimum": "::cuda::minimum<T>",
        "max": "::cuda::maximum<T>",
        "maximum": "::cuda::maximum<T>",
        "bit_and": "::cuda::std::bit_and<T>",
        "bit_or": "::cuda::std::bit_or<T>",
        "bit_xor": "::cuda::std::bit_xor<T>",
        "*": "::cuda::std::multiplies<T>",
        "&": "::cuda::std::bit_and<T>",
        "|": "::cuda::std::bit_or<T>",
        "^": "::cuda::std::bit_xor<T>",
        np.maximum: "::cuda::maximum<T>",
        np.minimum: "::cuda::minimum<T>",
        np.multiply: "::cuda::std::multiplies<T>",
        np.bitwise_and: "::cuda::std::bit_and<T>",
        np.bitwise_or: "::cuda::std::bit_or<T>",
        np.bitwise_xor: "::cuda::std::bit_xor<T>",
        operator.mul: "::cuda::std::multiplies<T>",
        operator.and_: "::cuda::std::bit_and<T>",
        operator.or_: "::cuda::std::bit_or<T>",
        operator.xor: "::cuda::std::bit_xor<T>",
    }

    def __init__(self, op):
        if isinstance(op, ScanOp):
            self.op = op.op
            self.op_category = op.op_category
            self.op_cpp = op.op_cpp
            return
        self.op = _normalize_numba_callable(op)
        try:
            is_sum = op in self._SUM_OPS
            is_known = op in self._KNOWN_OPS
        except TypeError:
            is_sum = is_known = False
        if is_sum:
            self.op_category = ScanOpCategory.SUM
            self.op_cpp = "::cuda::std::plus<T>"
        elif is_known:
            self.op_category = ScanOpCategory.KNOWN
            self.op_cpp = self._KNOWN_OPS[op]
        elif callable(op):
            self.op_category = ScanOpCategory.CALLABLE
            self.op_cpp = None
        else:
            raise ValueError(f"Unsupported scan operator: {op!r}")

    @property
    def is_sum(self):
        return self.op_category is ScanOpCategory.SUM

    @property
    def is_known(self):
        return self.op_category is ScanOpCategory.KNOWN

    @property
    def is_callable(self):
        return self.op_category is ScanOpCategory.CALLABLE
