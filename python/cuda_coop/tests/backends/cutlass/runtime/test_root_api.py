# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from cuda import coop

from ....support.paths import TESTS_ROOT

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.usefixtures("cutlass_runtime_available")]

from_dlpack = runtime.from_dlpack
_COLD_ACTIVATION_PROBE = (
    TESTS_ROOT / "support" / "fixtures" / "cutlass_root_sum_cold_activation.py"
)


@cute.kernel
def _root_block_sum_kernel(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block = coop.this_block()
    values_out[tidx] = coop.reduce(block, values_in[tidx])


@cute.jit
def _run_root_block_sum(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    _root_block_sum_kernel(values_in, values_out).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@cute.kernel
def _control_kernel(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    values_out[tidx] = values_in[tidx] + 1


@cute.jit
def _run_control(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    _control_kernel(values_in, values_out).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


@cute.kernel
def _root_load_store_kernel(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
    totals_out: cute.Tensor,
):
    block = coop.this_block()
    items = coop.ThreadData(2)
    coop.load(block, values_in, items)
    coop.store(block, values_out, items)
    total = coop.reduce(block, items)
    coop.store(block, totals_out, total)


@cute.jit
def _run_root_load_store(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
    totals_out: cute.Tensor,
):
    _root_load_store_kernel(values_in, values_out, totals_out).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


def test_cutlass_root_activates_automatically_in_fresh_process(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment["CUDA_CACHE_PATH"] = str(tmp_path / "cuda-cache")
    environment["CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR"] = str(
        tmp_path / "provider-cache"
    )
    environment.setdefault("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    result = subprocess.run(
        [sys.executable, str(_COLD_ACTIVATION_PROBE)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "CUTLASS root-Sum cold-activation subprocess failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_cutlass_common_root_runs_repeatedly_with_qualified_control() -> None:
    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, 33, dtype=torch.int32)
    values_in = values_host.cuda()
    values_out = torch.zeros_like(values_in)

    _run_root_block_sum(from_dlpack(values_in), from_dlpack(values_out))
    torch.cuda.synchronize()
    expected = torch.full_like(values_host, int(values_host.sum().item()))
    torch.testing.assert_close(values_out.cpu(), expected, atol=0, rtol=0)

    values_out.zero_()
    _run_root_block_sum(from_dlpack(values_in), from_dlpack(values_out))
    torch.cuda.synchronize()
    torch.testing.assert_close(values_out.cpu(), expected, atol=0, rtol=0)

    control_out = torch.zeros_like(values_in)
    _run_control(from_dlpack(values_in), from_dlpack(control_out))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        control_out.cpu(),
        values_host + 1,
        atol=0,
        rtol=0,
    )

    items_host = torch.arange(1, 65, dtype=torch.int32)
    items_in = items_host.cuda()
    items_out = torch.zeros_like(items_in)
    totals_out = torch.zeros((32,), dtype=torch.int32, device="cuda")
    _run_root_load_store(
        from_dlpack(items_in),
        from_dlpack(items_out),
        from_dlpack(totals_out),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(items_out.cpu(), items_host, atol=0, rtol=0)
    torch.testing.assert_close(
        totals_out.cpu(),
        torch.full((32,), int(items_host.sum()), dtype=torch.int32),
        atol=0,
        rtol=0,
    )
