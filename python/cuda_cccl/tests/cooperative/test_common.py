# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
import numpy as np
import pytest

from cuda.cccl.cooperative.experimental._common import (
    CudaSharedMemConfig,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
)


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


class TestNormalizeDimParam:
    def test_dim3_instance(self):
        """Test that dim3 instances are returned as-is."""
        dim = dim3(2, 3, 4)
        assert normalize_dim_param(dim) is dim

    @pytest.mark.parametrize(
        "value",
        [
            1,
            10,
            256,
        ],
    )
    def test_positive_integer(self, value):
        """Test conversion of positive integers to 1D dim3."""
        result = normalize_dim_param(value)
        assert isinstance(result, dim3)
        assert result.x == value
        assert result.y == 1
        assert result.z == 1

    @pytest.mark.parametrize(
        "value",
        [
            -1,
            -10,
        ],
    )
    def test_negative_integer(self, value):
        """Test that negative integers raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dim_param(value)
        assert "Dimension value must be non-negative" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dim_tuple,expected",
        [
            ((2, 3), dim3(2, 3, 1)),
            ((10, 20), dim3(10, 20, 1)),
            ((256, 128), dim3(256, 128, 1)),
        ],
    )
    def test_2d_tuple(self, dim_tuple, expected):
        """Test conversion of 2D tuples to dim3."""
        result = normalize_dim_param(dim_tuple)
        assert isinstance(result, dim3)
        assert result.x == expected.x
        assert result.y == expected.y
        assert result.z == expected.z

    @pytest.mark.parametrize(
        "dim_tuple",
        [
            (2, -3),
            (-10, 20),
            (-1, -1),
        ],
    )
    def test_2d_tuple_negative(self, dim_tuple):
        """Test that 2D tuples with negative values raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dim_param(dim_tuple)
        assert "Dimension values must be non-negative" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dim_tuple,expected",
        [
            ((2, 3, 4), dim3(2, 3, 4)),
            ((10, 20, 30), dim3(10, 20, 30)),
            ((256, 128, 64), dim3(256, 128, 64)),
        ],
    )
    def test_3d_tuple(self, dim_tuple, expected):
        """Test conversion of 3D tuples to dim3."""
        result = normalize_dim_param(dim_tuple)
        assert isinstance(result, dim3)
        assert result.x == expected.x
        assert result.y == expected.y
        assert result.z == expected.z

    @pytest.mark.parametrize(
        "dim_tuple",
        [
            (2, 3, -4),
            (10, -20, 30),
            (-1, -1, -1),
        ],
    )
    def test_3d_tuple_negative(self, dim_tuple):
        """Test that 3D tuples with negative values raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dim_param(dim_tuple)
        assert "Dimension values must be non-negative" in str(excinfo.value)

    @pytest.mark.parametrize(
        "dim_tuple",
        [
            (),
            (1,),
            (1, 2, 3, 4),
        ],
    )
    def test_invalid_tuple_length(self, dim_tuple):
        """Test that tuples with invalid lengths raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dim_param(dim_tuple)
        assert "Tuple dimension must have 2 or 3 elements" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [
            "string",
            [1, 2, 3],
            {"x": 1, "y": 2, "z": 3},
            None,
        ],
    )
    def test_unsupported_type(self, value):
        """Test that unsupported types raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            normalize_dim_param(value)
        assert "Unsupported dimension type:" in str(excinfo.value)


class TestCudaSharedMemConfig:
    @pytest.mark.parametrize(
        "enum_member, expected_value",
        [
            (CudaSharedMemConfig.BankSizeDefault, 0),
            (CudaSharedMemConfig.BankSizeFourByte, 1),
            (CudaSharedMemConfig.BankSizeEightByte, 2),
        ],
    )
    def test_enum_values(self, enum_member, expected_value):
        """Test that enum values are correctly defined."""
        assert enum_member.value == expected_value

    @pytest.mark.parametrize(
        "enum_member, expected_name",
        [
            (CudaSharedMemConfig.BankSizeDefault, "BankSizeDefault"),
            (CudaSharedMemConfig.BankSizeFourByte, "BankSizeFourByte"),
            (CudaSharedMemConfig.BankSizeEightByte, "BankSizeEightByte"),
        ],
    )
    def test_enum_names(self, enum_member, expected_name):
        """Test that enum names are correctly defined."""
        assert enum_member.name == expected_name

    @pytest.mark.parametrize(
        "enum_member, expected_str",
        [
            (CudaSharedMemConfig.BankSizeDefault, "cudaSharedMemBankSizeDefault"),
            (CudaSharedMemConfig.BankSizeFourByte, "cudaSharedMemBankSizeFourByte"),
            (CudaSharedMemConfig.BankSizeEightByte, "cudaSharedMemBankSizeEightByte"),
        ],
    )
    def test_string_representation(self, enum_member, expected_str):
        """Test the string representation of enum values."""
        assert str(enum_member) == expected_str

    def test_enum_iteration(self):
        """Test that we can iterate through all enum values."""
        expected_values = [
            CudaSharedMemConfig.BankSizeDefault,
            CudaSharedMemConfig.BankSizeFourByte,
            CudaSharedMemConfig.BankSizeEightByte,
        ]
        assert list(CudaSharedMemConfig) == expected_values

    @pytest.mark.parametrize(
        "name, expected_enum",
        [
            ("BankSizeDefault", CudaSharedMemConfig.BankSizeDefault),
            ("BankSizeFourByte", CudaSharedMemConfig.BankSizeFourByte),
            ("BankSizeEightByte", CudaSharedMemConfig.BankSizeEightByte),
        ],
    )
    def test_enum_lookup_by_name(self, name, expected_enum):
        """Test that we can look up enum values by name."""
        assert CudaSharedMemConfig[name] == expected_enum

    @pytest.mark.parametrize(
        "value, expected_enum",
        [
            (0, CudaSharedMemConfig.BankSizeDefault),
            (1, CudaSharedMemConfig.BankSizeFourByte),
            (2, CudaSharedMemConfig.BankSizeEightByte),
        ],
    )
    def test_enum_lookup_by_value(self, value, expected_enum):
        """Test that we can look up enum values by value."""
        assert CudaSharedMemConfig(value) == expected_enum
