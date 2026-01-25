# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import numpy as np
from mamba_selective_scan_fwd import (
    make_kernel_traits,
    make_selective_scan_fwd_kernel,
)
from numba import cuda

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


def test_mamba_selective_scan_fwd_simple():
    threads_per_block = 128
    items_per_thread = 4

    ref = _load_ref_data()
    seqlen = ref["seqlen"]
    expected_seqlen = threads_per_block * items_per_thread
    if seqlen != expected_seqlen:
        raise AssertionError(
            f"Reference seqlen {seqlen} != expected {expected_seqlen}."
        )

    traits = make_kernel_traits(np.float32, threads_per_block, items_per_thread)

    d_u = cuda.to_device(ref["u"])
    d_delta = cuda.to_device(ref["delta"])
    d_out = cuda.device_array_like(d_u)

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
    cuda.synchronize()

    h_out = d_out.copy_to_host()
    np.testing.assert_allclose(h_out, ref["out"], rtol=1e-5, atol=1e-5)
