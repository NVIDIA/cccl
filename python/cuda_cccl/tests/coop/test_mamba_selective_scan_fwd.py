# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from mamba_selective_scan_fwd import (
    make_kernel_traits,
    make_selective_scan_fwd_kernel,
    selective_scan_fwd_reference,
)
from numba import cuda


def test_mamba_selective_scan_fwd_simple():
    threads_per_block = 128
    items_per_thread = 4
    seqlen = threads_per_block * items_per_thread

    rng = np.random.default_rng(0)
    h_u = rng.standard_normal(seqlen).astype(np.float32)
    h_delta = (0.1 * rng.standard_normal(seqlen)).astype(np.float32)

    A = np.float32(-0.2)
    B = np.float32(0.7)
    C = np.float32(-0.3)
    D = np.float32(0.5)
    delta_bias = np.float32(0.01)

    traits = make_kernel_traits(np.float32, threads_per_block, items_per_thread)

    d_u = cuda.to_device(h_u)
    d_delta = cuda.to_device(h_delta)
    d_out = cuda.device_array_like(d_u)

    k = make_selective_scan_fwd_kernel(traits)[1, threads_per_block]
    k(d_u, d_delta, d_out, A, B, C, D, delta_bias, traits)
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    h_ref = selective_scan_fwd_reference(h_u, h_delta, A, B, C, D, delta_bias)

    np.testing.assert_allclose(h_out, h_ref, rtol=1e-5, atol=1e-5)
