# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from numba import cuda

from cuda import coop
from cuda.coop import _launch_config
from cuda.coop._rewrite import (
    LaunchConfigUnavailableError,
    get_kernel_param_value,
    get_kernel_param_value_safe,
)


def test_launch_config_temporary_toggle_restores_state():
    before = _launch_config.LAUNCH_CONFIG_ENABLED
    with _launch_config.temporary_launch_config_enabled(not before):
        assert _launch_config.LAUNCH_CONFIG_ENABLED is (not before)
    assert _launch_config.LAUNCH_CONFIG_ENABLED is before


def test_launch_config_disabled_wrappers_return_none():
    with _launch_config.temporary_launch_config_enabled(False):
        assert _launch_config.current_launch_config() is None
        assert _launch_config.ensure_current_launch_config() is None


def test_launch_config_reset_reads_env(monkeypatch):
    old = _launch_config.LAUNCH_CONFIG_ENABLED
    try:
        monkeypatch.setenv("NUMBA_CCCL_COOP_ENABLE_LAUNCH_CONFIG", "0")
        _launch_config.reset_launch_config_enabled()
        assert _launch_config.LAUNCH_CONFIG_ENABLED is False

        monkeypatch.setenv("NUMBA_CCCL_COOP_ENABLE_LAUNCH_CONFIG", "1")
        _launch_config.reset_launch_config_enabled()
        assert _launch_config.LAUNCH_CONFIG_ENABLED is True
    finally:
        _launch_config.set_launch_config_enabled(old)


def test_kernel_param_lookup_graceful_when_launch_config_disabled():
    with _launch_config.temporary_launch_config_enabled(False):
        assert get_kernel_param_value_safe("x", None) is None
        with pytest.raises(
            LaunchConfigUnavailableError,
            match="numba.cuda.launchconfig",
        ):
            get_kernel_param_value("x", None)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_two_phase_global_primitive_runs_when_launch_config_disabled():
    dim = 32
    items_per_thread = 2
    block_load = coop.block.load(np.int32, dim, items_per_thread)
    block_store = coop.block.store(np.int32, dim, items_per_thread)

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        block_load(d_in, thread_data)
        block_store(d_out, thread_data)

    h_input = np.arange(dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    with _launch_config.temporary_launch_config_enabled(False):
        kernel[1, dim](d_input, d_output)

    np.testing.assert_array_equal(d_output.copy_to_host(), h_input)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_single_phase_runs_with_explicit_threads_per_block_when_disabled():
    dim = 32
    items_per_thread = 2

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
            threads_per_block=dim,
        )
        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
            threads_per_block=dim,
        )

    h_input = np.arange(dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    with _launch_config.temporary_launch_config_enabled(False):
        kernel[1, dim](d_input, d_output)

    np.testing.assert_array_equal(d_output.copy_to_host(), h_input)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_single_phase_runs_with_dim_alias_when_launch_config_disabled():
    dim = 32
    items_per_thread = 2

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(
            d_in,
            thread_data,
            items_per_thread=items_per_thread,
            dim=dim,
        )
        coop.block.store(
            d_out,
            thread_data,
            items_per_thread=items_per_thread,
            dim=dim,
        )

    h_input = np.arange(dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    with _launch_config.temporary_launch_config_enabled(False):
        kernel[1, dim](d_input, d_output)

    np.testing.assert_array_equal(d_output.copy_to_host(), h_input)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_single_phase_requires_explicit_dim_if_launch_config_disabled():
    dim = 32
    items_per_thread = 2

    @cuda.jit
    def kernel(d_in, d_out):
        thread_data = coop.local.array(items_per_thread, dtype=d_in.dtype)
        coop.block.load(d_in, thread_data, items_per_thread=items_per_thread)
        coop.block.store(d_out, thread_data, items_per_thread=items_per_thread)

    h_input = np.arange(dim * items_per_thread, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array_like(d_input)

    with _launch_config.temporary_launch_config_enabled(False):
        with pytest.raises(
            LaunchConfigUnavailableError,
            match="threads_per_block",
        ):
            kernel[1, dim](d_input, d_output)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_single_phase_block_sum_runs_with_explicit_dim_when_disabled():
    dim = 32

    @cuda.jit
    def kernel(d_in, d_out):
        item = d_in[cuda.threadIdx.x]
        aggregate = coop.block.sum(item, items_per_thread=1, dim=dim)
        if cuda.threadIdx.x == 0:
            d_out[0] = aggregate

    h_input = np.arange(dim, dtype=np.int32)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(1, dtype=np.int32)

    with _launch_config.temporary_launch_config_enabled(False):
        kernel[1, dim](d_input, d_output)

    np.testing.assert_array_equal(d_output.copy_to_host(), np.array([h_input.sum()]))
