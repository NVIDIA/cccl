# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared test helpers for the STF test suite."""

import pytest


def require_vmm():
    """Composite/localized allocations build a localized array, which needs
    CUDA virtual address management. Skip rather than fail where it is absent
    (the C test at c/experimental/stf/test/test_placement.cpp gates the same
    way)."""
    from cuda.bindings import driver as cu

    assert cu.cuInit(0)[0] == cu.CUresult.CUDA_SUCCESS
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, supported = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        dev,
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if not supported:
        pytest.skip("device 0 does not support CUDA VMM (virtual address management)")
