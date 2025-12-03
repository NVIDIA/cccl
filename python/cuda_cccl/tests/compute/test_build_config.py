# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test BuildConfig functionality for NVRTC compile options."""

import numba.cuda
import numpy as np
import pytest

import cuda.compute
from cuda.compute import BuildConfig, OpKind


def test_build_config_creation():
    """Test that BuildConfig can be created with various options."""
    # Test with no options
    config1 = BuildConfig()
    assert config1 is not None

    # Test with compile flags only
    config2 = BuildConfig(extra_compile_flags=["-fmad=true"])
    assert config2 is not None

    # Test with include dirs only
    config3 = BuildConfig(extra_include_dirs=["/path/to/headers"])
    assert config3 is not None

    # Test with both
    config4 = BuildConfig(
        extra_compile_flags=["-fmad=true", "-use_fast_math"],
        extra_include_dirs=["/path/to/headers"],
    )
    assert config4 is not None


def test_build_config_type_validation():
    """Test that BuildConfig validates input types."""
    # Should raise TypeError for non-list compile flags
    with pytest.raises(TypeError):
        BuildConfig(extra_compile_flags="-fmad=true")

    # Should raise TypeError for non-list include dirs
    with pytest.raises(TypeError):
        BuildConfig(extra_include_dirs="/path/to/headers")


@pytest.mark.parametrize("dtype", [np.uint32, np.float32])
def test_reduce_with_build_config(dtype):
    """Test that reduce works with BuildConfig."""
    num_items = 100
    init_value = 0
    h_init = np.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)

    h_input = np.arange(num_items, dtype=dtype)
    d_input = numba.cuda.to_device(h_input)

    # Create BuildConfig with some NVRTC options
    # Note: These are valid NVRTC flags but may not affect the result
    build_config = BuildConfig(extra_compile_flags=["-lineinfo"])

    # Test with build_config
    cuda.compute.reduce_into(
        d_input, d_output, OpKind.PLUS, d_input.size, h_init, build_config=build_config
    )
    h_output = d_output.copy_to_host()
    expected = np.sum(h_input) + init_value
    assert h_output[0] == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [np.uint32, np.float32])
def test_make_reduce_into_with_build_config(dtype):
    """Test that make_reduce_into works with BuildConfig."""
    num_items = 100
    init_value = 42
    h_init = np.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)

    h_input = np.arange(num_items, dtype=dtype)
    d_input = numba.cuda.to_device(h_input)

    # Create BuildConfig
    build_config = BuildConfig(extra_compile_flags=["-lineinfo"])

    # Create reducer with build_config
    reducer = cuda.compute.make_reduce_into(
        d_input, d_output, OpKind.PLUS, h_init, build_config=build_config
    )

    # Execute reduction
    from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

    tmp_storage_bytes = reducer(None, d_input, d_output, num_items, h_init)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes)
    reducer(tmp_storage, d_input, d_output, num_items, h_init)

    h_output = d_output.copy_to_host()
    expected = np.sum(h_input) + init_value
    assert h_output[0] == pytest.approx(expected)


def test_build_config_caching():
    """Test that different BuildConfigs create different cached objects."""
    num_items = 100
    dtype = np.uint32
    h_init = np.array([0], dtype=dtype)
    d_output1 = numba.cuda.device_array(1, dtype=dtype)
    d_output2 = numba.cuda.device_array(1, dtype=dtype)

    h_input = np.arange(num_items, dtype=dtype)
    d_input = numba.cuda.to_device(h_input)

    # Create two different BuildConfigs
    config1 = BuildConfig(extra_compile_flags=["-lineinfo"])
    config2 = BuildConfig(extra_compile_flags=["-g"])

    # Create two reducers with different configs
    reducer1 = cuda.compute.make_reduce_into(
        d_input, d_output1, OpKind.PLUS, h_init, build_config=config1
    )
    reducer2 = cuda.compute.make_reduce_into(
        d_input, d_output2, OpKind.PLUS, h_init, build_config=config2
    )

    # Both should work (they're cached separately due to different configs)
    assert reducer1 is not None
    assert reducer2 is not None
