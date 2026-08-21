# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""EXAMPLE: localized model weights with module-owned lifetime.

The end-to-end idiom this package recommends for framework weights, in
one screen:

1. Build a place grid (here two views of one device; on multi-domain
   parts, one place per locality domain).
2. Allocate each weight with :func:`localized_parameter`: ONE ordinary
   ``torch.nn.Parameter`` whose physical pages are striped over the
   grid's places by a cute-partition spec. Checkpoint loaders, views,
   and every torch op see a plain tensor.
3. Ownership is the module's, via DLPack (``lifetime="gc"``, the
   default for parameters): module -> parameter -> storage ->
   allocation. Dropping the module frees the VMM and the placement
   metadata — no registry pin, no explicit release, no leak on model
   swap/unload.
4. Compiler-side passes read the placement through :func:`get_meta`
   (keyed by storage, so it survives views and Parameter wrapping).
"""

import gc
import weakref

import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
torch = pytest.importorskip("torch")

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop import pytorch as tp  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


class LocalizedMLP(torch.nn.Module):
    """Two matmul weights, each blocked over the grid's places along the
    outermost axis (rows land whole on one place)."""

    def __init__(self, grid, d_in=64, d_hidden=16384, d_out=64):
        super().__init__()
        self.w1 = tp.localized_parameter((d_hidden, d_in), torch.float32, grid)
        self.w2 = tp.localized_parameter((d_out, d_hidden), torch.float32, grid)

    def forward(self, x):
        return torch.relu(x @ self.w1.T) @ self.w2.T


@requires_cuda
def test_localized_weights_lifecycle_example():
    stf.machine_init()
    grid = stf.exec_place_grid.create([stf.exec_place.device(0)] * 2)

    model = LocalizedMLP(grid).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)

    # ordinary tensors to every consumer: checkpoint-style copy, forward
    x = torch.randn(8, 64, device="cuda")
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 64)

    # the structured channel a compiler pass reads
    meta = tp.get_meta(model.w1)
    assert meta.partition is not None and meta.lifetime == "gc"
    report = tp.placement_report(model.w1)
    assert report.accuracy == 1.0  # page-aligned rows: exact placement

    # module-owned lifetime: unloading the model frees pages AND metadata
    w1_meta = weakref.ref(meta)
    del meta, model
    gc.collect()
    torch.cuda.synchronize()
    assert w1_meta() is None
