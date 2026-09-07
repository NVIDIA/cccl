# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fresh-process probe for automatic CUTLASS activation through root Sum."""

from __future__ import annotations

import sys

from cuda import coop
from cuda.coop._core import root_api

_BACKEND_MODULE = "cuda.coop.cutlass"
_THREADS = 32
_COMMON_PAIR_OPERATIONS = (
    "merge_sort_pairs",
    "radix_sort_pairs",
    "topk_max_pairs",
    "topk_min_pairs",
)


def main() -> None:
    assert _BACKEND_MODULE in sys.modules
    assert all(callable(getattr(coop, name)) for name in _COMMON_PAIR_OPERATIONS)

    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    globals()["cute"] = cute

    @cute.kernel
    def kernel(values_in: cute.Tensor, values_out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        values_out[tidx] = coop.sum(coop.this_block(), values_in[tidx])

    @cute.jit
    def run(values_in: cute.Tensor, values_out: cute.Tensor):
        kernel(values_in, values_out).launch(
            grid=(1, 1, 1),
            block=(_THREADS, 1, 1),
        )

    cutlass.cuda.initialize_cuda_context()
    values_host = torch.arange(1, _THREADS + 1, dtype=torch.int32)
    values_in = values_host.cuda()
    values_out = torch.zeros_like(values_in)
    run(from_dlpack(values_in), from_dlpack(values_out))
    torch.cuda.synchronize()

    assert _BACKEND_MODULE in sys.modules
    expected = torch.full_like(values_host, int(values_host.sum().item()))
    torch.testing.assert_close(values_out.cpu(), expected, atol=0, rtol=0)
    assert root_api._ACTIVE_BACKEND_MODULE.get() is None
    assert root_api._backend_module_name() == _BACKEND_MODULE


if __name__ == "__main__":
    main()
