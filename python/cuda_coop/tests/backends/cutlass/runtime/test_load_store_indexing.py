# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ..support.runtime import (
    Int32,
    coop,
    cute,
    cutlass,
    from_dlpack,
    runtime_pytestmark,
    torch,
)

pytestmark = runtime_pytestmark


@cute.kernel
def _noncontiguous_load_adapter_kernel(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    items = coop.ThreadData(2, dtype=Int32)
    coop._block.load(
        values_in,
        items,
        valid_items=55,
        oob_default=-777,
        offset=3,
    )
    coop._block.store(values_out, items, valid_items=55, offset=5)


@cute.jit
def _run_noncontiguous_load_adapter(
    values_in: cute.Tensor,
    values_out: cute.Tensor,
):
    _noncontiguous_load_adapter_kernel(values_in, values_out).launch(
        grid=(1, 1, 1),
        block=(32, 1, 1),
    )


def test_provider_noncontiguous_load_retains_indexing_payload_adapter():
    """Keep strided tensors on the indexing-payload provider route."""

    cutlass.cuda.initialize_cuda_context()

    storage_host = torch.arange(160, dtype=torch.int32)
    values_host = storage_host[::2]
    assert not values_host.is_contiguous()
    values_in = values_host.cuda()
    sentinel = -999
    output_storage = torch.full((160,), sentinel, dtype=torch.int32, device="cuda")
    values_out = output_storage[::2]
    assert not values_out.is_contiguous()

    _run_noncontiguous_load_adapter(
        from_dlpack(values_in),
        from_dlpack(values_out),
    )
    torch.cuda.synchronize()

    expected = torch.full((80,), sentinel, dtype=torch.int32)
    expected[5:60] = values_host[3:58]
    torch.testing.assert_close(values_out.cpu(), expected, atol=0, rtol=0)
