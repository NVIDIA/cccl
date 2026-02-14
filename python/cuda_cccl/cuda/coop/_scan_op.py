# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
cuda.coop._scan_op
==================

This module implements the ``ScanOp`` class and related functions.
"""

import operator
from enum import Enum

import numpy as np

from ._typing import (
    ScanOpType,
)


class ScanOpCategory(Enum):
    """
    Represents the category of a scan operator.  This is used to guide
    specialization of the C++ API.
    """

    Sum = "Sum"
    """
    Represents a sum operator.
    """

    Known = "Known"
    """
    Represents one of the known, non-sum associative operators: multiply,
    minimum, maximum, bitwise AND, bitwise OR, and bitwise XOR.
    """

    Callable = "Callable"
    """
    Represents a user-defined callable operator (e.g. a Python function).
    """


CUDA_STD_PLUS = "::cuda::std::plus<T>"
CUDA_STD_MULTIPLIES = "::cuda::std::multiplies<T>"
CUDA_STD_BIT_AND = "::cuda::std::bit_and<T>"
CUDA_STD_BIT_OR = "::cuda::std::bit_or<T>"
CUDA_STD_BIT_XOR = "::cuda::std::bit_xor<T>"
# `std::min` and `std::max` are STL functions, not operators, so there aren't
# `::cuda`-prefixed versions.  Instead, CUDA provides `::cuda::minimum<T>` and
# `::cuda::maximum<T>`, which *are* function objects, which is what we want.
CUDA_MINIMUM = "::cuda::minimum<T>"
CUDA_MAXIMUM = "::cuda::maximum<T>"


class ScanOp:
    """
    Represents an associative binary operator for a prefix scan operation.
    This helper class is used to *normalize* the operator provided by a user,
    which may be a string, NumPy/Python operator, or callable, to a form that
    can be used in Numba CUDA JIT'd kernels.
    """

    # Set of all ops interpreted as sum (for (inclusive|exclusive)_sum).
    SUM_OPS = {
        "+",
        "add",
        "plus",
        np.add,
        operator.add,
    }
    """
    Set of all ops interpreted as sum (for (inclusive|exclusive)_sum).
    """

    # Map of all known (non-sum) operators to their C++ type representations.
    KNOWN_OPS = {
        # String names
        "mul": CUDA_STD_MULTIPLIES,
        "multiplies": CUDA_STD_MULTIPLIES,
        "min": CUDA_MINIMUM,
        "minimum": CUDA_MINIMUM,
        "max": CUDA_MAXIMUM,
        "maximum": CUDA_MAXIMUM,
        "bit_and": CUDA_STD_BIT_AND,
        "bit_or": CUDA_STD_BIT_OR,
        "bit_xor": CUDA_STD_BIT_XOR,
        # String operators
        "*": CUDA_STD_MULTIPLIES,
        "&": CUDA_STD_BIT_AND,
        "|": CUDA_STD_BIT_OR,
        "^": CUDA_STD_BIT_XOR,
        # NumPy functions
        np.maximum: CUDA_MAXIMUM,
        np.minimum: CUDA_MINIMUM,
        np.multiply: CUDA_STD_MULTIPLIES,
        np.bitwise_and: CUDA_STD_BIT_AND,
        np.bitwise_or: CUDA_STD_BIT_OR,
        np.bitwise_xor: CUDA_STD_BIT_XOR,
        # Python operator module functions.
        operator.mul: CUDA_STD_MULTIPLIES,
        operator.and_: CUDA_STD_BIT_AND,
        operator.or_: CUDA_STD_BIT_OR,
        operator.xor: CUDA_STD_BIT_XOR,
    }
    """
    Map of all known (non-sum) operators to their C++ type representations.
    """

    def __init__(self, op: ScanOpType):
        """
        Initializes the ScanOp instance.

        :param op: Supplies the :ref:`ScanOpType` scan operator to use
            for the block-wide scan.
        :type op: ScanOpType

        :raises ValueError: If the provided operator is not supported.

        """
        if isinstance(op, ScanOp):
            # If op is already a ScanOp instance, just prime this instance
            # with its values and return.
            self.op = op.op
            self.op_category = op.op_category
            self.op_cpp = op.op_cpp
            return

        self.op = op
        self.op_category = None
        self.op_cpp = None

        # Handle string names and operators.
        if isinstance(op, str):
            if op in self.SUM_OPS:
                self.op_category = ScanOpCategory.Sum
                self.op_cpp = CUDA_STD_PLUS
            elif op in self.KNOWN_OPS:
                self.op_category = ScanOpCategory.Known
                self.op_cpp = self.KNOWN_OPS[op]
            else:
                raise ValueError(f"Unsupported scan operator: {op}")

        # Handle NumPy functions or other callables.
        elif callable(op):
            if op in self.SUM_OPS:
                self.op_category = ScanOpCategory.Sum
                self.op_cpp = CUDA_STD_PLUS
            elif op in self.KNOWN_OPS:
                self.op_category = ScanOpCategory.Known
                self.op_cpp = self.KNOWN_OPS[op]
            else:
                # Custom callable; no op_cpp representation.
                self.op_category = ScanOpCategory.Callable
        else:
            raise ValueError(f"Unsupported scan op type: {type(op)}")

    def __repr__(self):
        return f"ScanOp({self.op})"

    @property
    def is_sum(self):
        """
        Returns ``True`` if the scan operator is a sum operator.
        """
        return self.op_category == ScanOpCategory.Sum

    @property
    def is_known(self):
        """
        Returns ``True`` if the scan operator is a known operator.
        """
        return self.op_category == ScanOpCategory.Known

    @property
    def is_callable(self):
        """
        Returns ``True`` if the scan operator is a callable.
        """
        return self.op_category == ScanOpCategory.Callable
