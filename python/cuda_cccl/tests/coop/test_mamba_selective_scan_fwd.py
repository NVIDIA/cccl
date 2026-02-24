# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import numpy as np
import pytest
from mamba_selective_scan_fwd import (
    make_kernel_traits,
    make_selective_scan_fwd_kernel,
    make_selective_scan_fwd_kernel_single_phase_bleeding_edge_qol,
    make_selective_scan_fwd_kernel_single_phase_temp_storage,
)
from numba import cuda

from cuda import coop
from cuda.coop import BlockLoadAlgorithm, BlockStoreAlgorithm

_REF_PATH = Path(__file__).parent / "data" / "mamba_selective_scan_fwd_ref.npz"


def _load_ref_data():
    if not _REF_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference data: {_REF_PATH}. "
            "Regenerate with tests/coop/generate_mamba_selective_scan_fwd_ref.py."
        )
    data = np.load(_REF_PATH)
    return {
        "u": data["u"].astype(np.float32),
        "delta": data["delta"].astype(np.float32),
        "A": np.float32(data["A"][0]),
        "B": np.float32(data["B"][0]),
        "C": np.float32(data["C"][0]),
        "D": np.float32(data["D"][0]),
        "delta_bias": np.float32(data["delta_bias"][0]),
        "out": data["out"].astype(np.float32),
        "seqlen": int(data["seqlen"][0]),
    }


@pytest.mark.parametrize(
    "kernel_variant",
    [
        "traits_gpu_dataclass",
        "single_phase_temp_storage",
        "single_phase_bleeding_edge_qol",
    ],
)
def test_mamba_selective_scan_fwd_simple(kernel_variant):
    threads_per_block = 128
    items_per_thread = 4

    ref = _load_ref_data()
    seqlen = ref["seqlen"]
    expected_seqlen = threads_per_block * items_per_thread
    if seqlen != expected_seqlen:
        raise AssertionError(
            f"Reference seqlen {seqlen} != expected {expected_seqlen}."
        )

    d_u = cuda.to_device(ref["u"])
    d_delta = cuda.to_device(ref["delta"])
    d_out = cuda.device_array_like(d_u)

    if kernel_variant == "traits_gpu_dataclass":
        traits = make_kernel_traits(np.float32, threads_per_block, items_per_thread)
        k = make_selective_scan_fwd_kernel(traits)[1, threads_per_block]
        k(
            d_u,
            d_delta,
            d_out,
            ref["A"],
            ref["B"],
            ref["C"],
            ref["D"],
            ref["delta_bias"],
            traits,
        )
    elif kernel_variant == "single_phase_temp_storage":
        k = make_selective_scan_fwd_kernel_single_phase_temp_storage(
            threads_per_block,
            items_per_thread,
        )[1, threads_per_block]
        k(
            d_u,
            d_delta,
            d_out,
            ref["A"],
            ref["B"],
            ref["C"],
            ref["D"],
            ref["delta_bias"],
        )
    elif kernel_variant == "single_phase_bleeding_edge_qol":
        k = make_selective_scan_fwd_kernel_single_phase_bleeding_edge_qol(
            threads_per_block,
            items_per_thread,
        )[1, threads_per_block]
        k(
            d_u,
            d_delta,
            d_out,
            ref["A"],
            ref["B"],
            ref["C"],
            ref["D"],
            ref["delta_bias"],
        )
    else:
        raise AssertionError(f"Unknown kernel_variant: {kernel_variant}")
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    np.testing.assert_allclose(h_out, ref["out"], rtol=1e-5, atol=1e-5)


def test_mamba_dual_temp_storage_staging_kernel():
    threads_per_block = 128
    items_per_thread = 4

    ref = _load_ref_data()
    seqlen = ref["seqlen"]
    expected_seqlen = threads_per_block * items_per_thread
    if seqlen != expected_seqlen:
        raise AssertionError(
            f"Reference seqlen {seqlen} != expected {expected_seqlen}."
        )

    d_u = cuda.to_device(ref["u"])
    d_delta = cuda.to_device(ref["delta"])
    d_u_out = cuda.device_array_like(d_u)
    d_delta_out = cuda.device_array_like(d_delta)

    @cuda.jit
    def kernel(u, delta, u_out, delta_out):
        u_vals = coop.local.array(items_per_thread, dtype=u.dtype)
        delta_vals = coop.local.array(items_per_thread, dtype=delta.dtype)
        temp_load = coop.TempStorage()
        temp_store = coop.TempStorage()

        # Reuse temp_load across two block.load calls and temp_store across two
        # block.store calls, relying on TempStorage auto-sync insertion.
        coop.block.load(
            u,
            u_vals,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            temp_storage=temp_load,
        )
        coop.block.load(
            delta,
            delta_vals,
            items_per_thread=items_per_thread,
            algorithm=BlockLoadAlgorithm.WARP_TRANSPOSE,
            temp_storage=temp_load,
        )
        coop.block.store(
            u_out,
            u_vals,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.WARP_TRANSPOSE,
            temp_storage=temp_store,
        )
        coop.block.store(
            delta_out,
            delta_vals,
            items_per_thread=items_per_thread,
            algorithm=BlockStoreAlgorithm.WARP_TRANSPOSE,
            temp_storage=temp_store,
        )

    kernel[1, threads_per_block](d_u, d_delta, d_u_out, d_delta_out)
    cuda.synchronize()

    np.testing.assert_allclose(d_u_out.copy_to_host(), ref["u"], rtol=0, atol=0)
    np.testing.assert_allclose(d_delta_out.copy_to_host(), ref["delta"], rtol=0, atol=0)
