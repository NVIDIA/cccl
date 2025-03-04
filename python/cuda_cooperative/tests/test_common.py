# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest

from cuda.cooperative.experimental._common import normalize_dtype_param


class TestNormalizeDtypeParam:
    def test_numba_type(self):
        """Test that numba types are returned as-is."""
        numba_types = [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ]

        for dtype in numba_types:
            assert normalize_dtype_param(dtype) is dtype

    def test_numpy_dtype(self):
        """Test conversion of numpy.dtype objects."""
        numpy_dtypes = [
            np.dtype(np.int32),
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.uint8),
            np.dtype(np.complex64),
        ]

        expected_numba_types = [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ]

        for dtype, expected in zip(numpy_dtypes, expected_numba_types):
            result = normalize_dtype_param(dtype)
            assert result == expected

    def test_numpy_type(self):
        """Test conversion of numpy type objects."""
        numpy_types = [
            np.int32,
            np.float32,
            np.float64,
            np.uint8,
            np.complex64,
        ]

        expected_numba_types = [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ]

        for dtype, expected in zip(numpy_types, expected_numba_types):
            result = normalize_dtype_param(dtype)
            assert result == expected

    def test_string_with_np_prefix(self):
        """Test conversion of strings with 'np.' prefix."""
        dtype_strings = [
            "np.int32",
            "np.float32",
            "np.float64",
            "np.uint8",
            "np.complex64",
        ]

        expected_numba_types = [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ]

        for dtype, expected in zip(dtype_strings, expected_numba_types):
            result = normalize_dtype_param(dtype)
            assert result == expected

    def test_string_numba_type(self):
        """Test conversion of strings representing numba types."""
        dtype_strings = [
            "int32",
            "float32",
            "float64",
            "uint8",
            "complex64",
        ]

        expected_numba_types = [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ]

        for dtype, expected in zip(dtype_strings, expected_numba_types):
            result = normalize_dtype_param(dtype)
            assert result == expected

    def test_invalid_string_with_period(self):
        """Test that strings with periods not starting with 'np.' raise ValueError."""
        invalid_strings = [
            "invalid.type",
            "numpy.int32",
            "something.else",
        ]

        for dtype in invalid_strings:
            with pytest.raises(ValueError) as excinfo:
                normalize_dtype_param(dtype)
            assert "String with period must start with 'np.'" in str(excinfo.value)

    def test_invalid_numpy_type(self):
        """Test that invalid numpy type names raise ValueError."""
        invalid_strings = [
            "np.invalid_type",
            "np.not_a_type",
        ]

        for dtype in invalid_strings:
            with pytest.raises(ValueError) as excinfo:
                normalize_dtype_param(dtype)
            assert "Invalid numpy dtype:" in str(excinfo.value)

    def test_invalid_numba_type(self):
        """Test that invalid numba type names raise ValueError."""
        invalid_strings = [
            "invalid_type",
            "not_a_type",
        ]

        for dtype in invalid_strings:
            with pytest.raises(ValueError) as excinfo:
                normalize_dtype_param(dtype)
            assert "Invalid numba type name:" in str(excinfo.value)

    def test_unrecognized_format(self):
        """Test that unrecognized dtype formats raise ValueError."""
        invalid_dtypes = [
            123,
            [1, 2, 3],
            {"type": "int32"},
            None,
        ]

        for dtype in invalid_dtypes:
            with pytest.raises(ValueError) as excinfo:
                normalize_dtype_param(dtype)
            assert "Unrecognized dtype format:" in str(excinfo.value)

    def test_python_builtin_types(self):
        """Test conversion of Python built-in types."""
        python_types = [
            int,
            float,
        ]

        expected_numba_types = [
            numba.int64,
            numba.float64,
        ]

        for dtype, expected in zip(python_types, expected_numba_types):
            result = normalize_dtype_param(dtype)
            assert result == expected
