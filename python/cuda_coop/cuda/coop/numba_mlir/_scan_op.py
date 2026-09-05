# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import operator
from enum import Enum

import numpy as np


class ScanOpCategory(Enum):
    """Normalized category for block and warp scan operators."""

    SUM = "sum"
    KNOWN = "known"
    CALLABLE = "callable"


CUDA_STD_PLUS = "::cuda::std::plus<T>"
CUDA_STD_MULTIPLIES = "::cuda::std::multiplies<T>"
CUDA_STD_BIT_AND = "::cuda::std::bit_and<T>"
CUDA_STD_BIT_OR = "::cuda::std::bit_or<T>"
CUDA_STD_BIT_XOR = "::cuda::std::bit_xor<T>"
CUDA_MINIMUM = "::cuda::minimum<T>"
CUDA_MAXIMUM = "::cuda::maximum<T>"


class ScanOp:
    """Normalize a Python scan operator into a CUB-compatible descriptor.

    Known operators map directly to C++ functors. Other Python callables are
    compiled as device support functions and linked into the generated CUB
    wrapper.
    """

    # Operators interpreted as sum operations.
    SUM_OPS = {
        "+",
        "add",
        "plus",
        "sum",
        np.add,
        operator.add,
    }

    # Known non-sum operators that map to C++ functors.
    KNOWN_OPS = {
        "mul": CUDA_STD_MULTIPLIES,
        "multiply": CUDA_STD_MULTIPLIES,
        "multiplies": CUDA_STD_MULTIPLIES,
        "min": CUDA_MINIMUM,
        "minimum": CUDA_MINIMUM,
        "max": CUDA_MAXIMUM,
        "maximum": CUDA_MAXIMUM,
        "bit_and": CUDA_STD_BIT_AND,
        "bit_or": CUDA_STD_BIT_OR,
        "bit_xor": CUDA_STD_BIT_XOR,
        "*": CUDA_STD_MULTIPLIES,
        "&": CUDA_STD_BIT_AND,
        "|": CUDA_STD_BIT_OR,
        "^": CUDA_STD_BIT_XOR,
        np.maximum: CUDA_MAXIMUM,
        np.minimum: CUDA_MINIMUM,
        np.multiply: CUDA_STD_MULTIPLIES,
        np.bitwise_and: CUDA_STD_BIT_AND,
        np.bitwise_or: CUDA_STD_BIT_OR,
        np.bitwise_xor: CUDA_STD_BIT_XOR,
        operator.mul: CUDA_STD_MULTIPLIES,
        operator.and_: CUDA_STD_BIT_AND,
        operator.or_: CUDA_STD_BIT_OR,
        operator.xor: CUDA_STD_BIT_XOR,
    }

    def __init__(self, op):
        if isinstance(op, ScanOp):
            self.op = op.op
            self.op_category = op.op_category
            self.op_cpp = op.op_cpp
            return

        self.op = op
        self.op_category = None
        self.op_cpp = None

        if isinstance(op, str):
            if op in self.SUM_OPS:
                self.op_category = ScanOpCategory.SUM
                self.op_cpp = CUDA_STD_PLUS
            elif op in self.KNOWN_OPS:
                self.op_category = ScanOpCategory.KNOWN
                self.op_cpp = self.KNOWN_OPS[op]
            else:
                raise ValueError(f"Unsupported scan operator: {op}")
        elif callable(op):
            if op in self.SUM_OPS:
                self.op_category = ScanOpCategory.SUM
                self.op_cpp = CUDA_STD_PLUS
            elif op in self.KNOWN_OPS:
                self.op_category = ScanOpCategory.KNOWN
                self.op_cpp = self.KNOWN_OPS[op]
            else:
                self.op_category = ScanOpCategory.CALLABLE
        else:
            raise ValueError(f"Unsupported scan op type: {type(op)}")

    @property
    def is_sum(self):
        return self.op_category == ScanOpCategory.SUM

    @property
    def is_known(self):
        return self.op_category == ScanOpCategory.KNOWN

    @property
    def is_callable(self):
        return self.op_category == ScanOpCategory.CALLABLE
