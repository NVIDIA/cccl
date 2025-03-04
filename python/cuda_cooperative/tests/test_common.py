# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest

from cuda.cooperative.experimental._common import normalize_dtype_param


class TestNormalizeDtypeParam:
    @pytest.mark.parametrize(
        "dtype",
        [
            numba.int32,
            numba.float32,
            numba.float64,
            numba.uint8,
            numba.complex64,
        ],
    )
    def test_numba_type(self, dtype):
        """Test that numba types are returned as-is."""
        assert normalize_dtype_param(dtype) is dtype

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (np.dtype(np.int32), numba.int32),
            (np.dtype(np.float32), numba.float32),
            (np.dtype(np.float64), numba.float64),
            (np.dtype(np.uint8), numba.uint8),
            (np.dtype(np.complex64), numba.complex64),
        ],
    )
    def test_numpy_dtype(self, dtype, expected):
        """Test conversion of numpy.dtype objects."""
        result = normalize_dtype_param(dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (np.int32, numba.int32),
            (np.float32, numba.float32),
            (np.float64, numba.float64),
            (np.uint8, numba.uint8),
            (np.complex64, numba.complex64),
        ],
    )
    def test_numpy_type(self, dtype, expected):
        """Test conversion of numpy type objects."""
        result = normalize_dtype_param(dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("np.int32", numba.int32),
            ("np.float32", numba.float32),
            ("np.float64", numba.float64),
            ("np.uint8", numba.uint8),
            ("np.complex64", numba.complex64),
        ],
    )
    def test_string_with_np_prefix(self, dtype, expected):
        """Test conversion of strings with 'np.' prefix."""
        result = normalize_dtype_param(dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            ("int32", numba.int32),
            ("float32", numba.float32),
            ("float64", numba.float64),
            ("uint8", numba.uint8),
            ("complex64", numba.complex64),
        ],
    )
    def test_string_numba_type(self, dtype, expected):
        """Test conversion of strings representing numba types."""
        result = normalize_dtype_param(dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "dtype",
        [
            "invalid.type",
            "numpy.int32",
            "something.else",
        ],
    )
    def test_invalid_string_with_period(self, dtype):
        """Test that strings with periods not starting with 'np.' raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dtype_param(dtype)
        assert "String with period must start with 'np.'" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dtype",
        [
            "np.invalid_type",
            "np.not_a_type",
        ],
    )
    def test_invalid_numpy_type(self, dtype):
        """Test that invalid numpy type names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dtype_param(dtype)
        assert "Invalid numpy dtype:" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dtype",
        [
            "invalid_type",
            "not_a_type",
        ],
    )
    def test_invalid_numba_type(self, dtype):
        """Test that invalid numba type names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dtype_param(dtype)
        assert "Invalid numba type name:" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dtype",
        [
            123,
            [1, 2, 3],
            {"type": "int32"},
            None,
        ],
    )
    def test_unrecognized_format(self, dtype):
        """Test that unrecognized dtype formats raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dtype_param(dtype)
        assert "Unrecognized dtype format:" in str(excinfo.value)
