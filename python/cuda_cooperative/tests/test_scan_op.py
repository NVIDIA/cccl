# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
test_scan_op.py

This file contains unit tests for the cuda.cooperative.experimental._scan_op
module's ``ScanOp`` class and related functionality.
"""

import operator

import numpy as np
import pytest

from cuda.cooperative.experimental._scan_op import (
    CUDA_MAXIMUM,
    CUDA_MINIMUM,
    CUDA_STD_BIT_AND,
    CUDA_STD_BIT_OR,
    CUDA_STD_BIT_XOR,
    CUDA_STD_MULTIPLIES,
    CUDA_STD_PLUS,
    ScanOp,
    ScanOpCategory,
)


class TestScanOp:
    @pytest.mark.parametrize(
        "op, expected_category, expected_cpp",
        [
            ("add", ScanOpCategory.Sum, CUDA_STD_PLUS),
            ("mul", ScanOpCategory.Known, CUDA_STD_MULTIPLIES),
            ("multiplies", ScanOpCategory.Known, CUDA_STD_MULTIPLIES),
            ("min", ScanOpCategory.Known, CUDA_MINIMUM),
            ("minimum", ScanOpCategory.Known, CUDA_MINIMUM),
            ("max", ScanOpCategory.Known, CUDA_MAXIMUM),
            ("maximum", ScanOpCategory.Known, CUDA_MAXIMUM),
            ("bit_and", ScanOpCategory.Known, CUDA_STD_BIT_AND),
            ("bit_or", ScanOpCategory.Known, CUDA_STD_BIT_OR),
            ("bit_xor", ScanOpCategory.Known, CUDA_STD_BIT_XOR),
        ],
    )
    def test_string_names(self, op, expected_category, expected_cpp):
        """Test that string operators are correctly processed."""
        scan_op = ScanOp(op)
        assert scan_op.op == op
        assert scan_op.op_category == expected_category
        assert scan_op.op_cpp == expected_cpp

    @pytest.mark.parametrize(
        "op, expected_category, expected_cpp",
        [
            ("+", ScanOpCategory.Sum, CUDA_STD_PLUS),
            ("*", ScanOpCategory.Known, CUDA_STD_MULTIPLIES),
            ("&", ScanOpCategory.Known, CUDA_STD_BIT_AND),
            ("|", ScanOpCategory.Known, CUDA_STD_BIT_OR),
            ("^", ScanOpCategory.Known, CUDA_STD_BIT_XOR),
        ],
    )
    def test_string_operators(self, op, expected_category, expected_cpp):
        """Test that string operators are correctly processed."""
        scan_op = ScanOp(op)
        assert scan_op.op == op
        assert scan_op.op_category == expected_category
        assert scan_op.op_cpp == expected_cpp

    @pytest.mark.parametrize(
        "op, expected_category, expected_cpp",
        [
            (np.add, ScanOpCategory.Sum, CUDA_STD_PLUS),
            (np.multiply, ScanOpCategory.Known, CUDA_STD_MULTIPLIES),
            (np.minimum, ScanOpCategory.Known, CUDA_MINIMUM),
            (np.maximum, ScanOpCategory.Known, CUDA_MAXIMUM),
            (np.bitwise_and, ScanOpCategory.Known, CUDA_STD_BIT_AND),
            (np.bitwise_or, ScanOpCategory.Known, CUDA_STD_BIT_OR),
            (np.bitwise_xor, ScanOpCategory.Known, CUDA_STD_BIT_XOR),
        ],
    )
    def test_numpy_functions(self, op, expected_category, expected_cpp):
        """Test that NumPy functions are correctly processed."""
        scan_op = ScanOp(op)
        assert scan_op.op == op
        assert scan_op.op_category == expected_category
        assert scan_op.op_cpp == expected_cpp

    @pytest.mark.parametrize(
        "op, expected_category, expected_cpp",
        [
            (operator.add, ScanOpCategory.Sum, CUDA_STD_PLUS),
            (operator.mul, ScanOpCategory.Known, CUDA_STD_MULTIPLIES),
            (operator.and_, ScanOpCategory.Known, CUDA_STD_BIT_AND),
            (operator.or_, ScanOpCategory.Known, CUDA_STD_BIT_OR),
            (operator.xor, ScanOpCategory.Known, CUDA_STD_BIT_XOR),
        ],
    )
    def test_python_operator_module_functions(
        self, op, expected_category, expected_cpp
    ):
        """Test that Python operator module functions are correctly processed."""
        scan_op = ScanOp(op)
        assert scan_op.op == op
        assert scan_op.op_category == expected_category
        assert scan_op.op_cpp == expected_cpp

    def test_custom_callable(self):
        """Test that custom callables are correctly processed."""

        def custom_add(a, b):
            return a + b

        scan_op = ScanOp(custom_add)
        assert scan_op.op == custom_add
        assert scan_op.op_category == ScanOpCategory.Callable
        assert scan_op.op_cpp is None

    @pytest.mark.parametrize(
        "op",
        [
            "unsupported_op",
            123,
            [1, 2, 3],
            {"op": "+"},
            None,
        ],
    )
    def test_invalid_operators(self, op):
        """Test that invalid operators raise ValueError."""
        with pytest.raises(ValueError):
            ScanOp(op)

    def test_repr(self):
        """Test the string representation of ScanOp."""
        scan_op = ScanOp("+")
        assert repr(scan_op) == "ScanOp(+)"

        scan_op = ScanOp(np.add)
        assert "ScanOp(" in repr(scan_op)
        assert "add" in repr(scan_op)
