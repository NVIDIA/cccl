# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    ROW_SUM_TEMP_STORAGE as _ROW_SUM_TEMP_STORAGE,
)
from ..support.runtime import (
    ROW_SUM_TEMP_STORAGE_2 as _ROW_SUM_TEMP_STORAGE_2,
)
from ..support.runtime import (
    coop,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)
from ..support.runtime import (
    has_cub_row_reduce_headers as _has_cub_row_reduce_headers,
)

pytestmark = runtime_pytestmark


@cute.kernel
def _row_sum_kernel(
    values_in: cute.Tensor,
    total_out: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    value = values_in[tidx]
    total = coop._block.row_sum(
        value,
        rows_per_block=1,
        warps_per_row=4,
        temp_storage=_ROW_SUM_TEMP_STORAGE,
        launch_metadata={"threads_per_block": 128},
    )
    repeated_total = coop._block.row_sum(
        value,
        rows_per_block=1,
        warps_per_row=4,
        temp_storage=_ROW_SUM_TEMP_STORAGE_2,
        launch_metadata={"threads_per_block": 128},
    )
    total_out[tidx] = total + (repeated_total - total)


@cute.jit
def _run_row_sum(
    values_in: cute.Tensor,
    total_out: cute.Tensor,
):
    _row_sum_kernel(values_in, total_out).launch(grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.skipif(
    not _has_cub_row_reduce_headers(),
    reason="requires CUB block_row_reduce.cuh",
)
def test_provider_row_sum_runtime():
    cutlass.cuda.initialize_cuda_context()
    _ROW_SUM_TEMP_STORAGE.reset_uses()
    _ROW_SUM_TEMP_STORAGE_2.reset_uses()

    values_host = torch.arange(128, dtype=torch.float32)
    values_in = values_host.cuda()
    total_out = torch.zeros((128,), dtype=torch.float32, device="cuda")

    _run_row_sum(from_dlpack(values_in), from_dlpack(total_out))
    torch.cuda.synchronize()

    expected_total = torch.full(
        (128,), float(values_host.sum().item()), dtype=torch.float32
    )
    torch.testing.assert_close(total_out.cpu(), expected_total, atol=0, rtol=0)
