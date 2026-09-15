# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The localized programming model for pytorch users, as runnable examples.

The premise: allocate with a placement (``torch.localized.*`` factories) and
compute with ordinary pytorch -- a localized tensor is a plain tensor. Where
compute should follow the pages, ``torch.localized.views(t)`` hands out one
plain strided view per die (the placement-aware ``chunk``), and the user
writes the per-die loop with stock torch.

The examples walk the spectrum deliberately:
  1. per-die views partition the tensor exactly;
  2. pointwise, in place, per die -- the AXPY shape;
  3. reductions over the SPLIT dim -- per-die partials + a fold;
  4. an ``nn.Module`` with localized parameters end to end.
"""

import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
torch = pytest.importorskip("torch")

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop import pytorch as tp  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

N_PLACES = 2
SHAPE = (4096, 1024)  # rows split across dies; big enough for real striping


@pytest.fixture(params=["devices", "locality_domains"])
def grid(request):
    """Every example runs at two granularities: a device grid (repeat), and
    the machine's locality domains -- the substrate the placement work is
    for. The domain flavor skips cleanly where the locality-domain
    bindings (PR #10703) are not in the build."""
    stf.machine_init()
    if request.param == "devices":
        return stf.exec_place_grid.from_devices([0] * N_PLACES)
    eg = stf.exec_place_grid
    if hasattr(eg, "machine"):
        return eg.machine(granularity="locality_domain")
    if hasattr(eg, "locality_domains"):
        return eg.locality_domains(0)
    pytest.skip("locality-domain bindings not available (PR #10703)")


# -- 1. per-die views partition the tensor exactly ---------------------------


@requires_cuda
def test_views_partition_exactly(grid):
    x = tp.localized_empty(SHAPE, torch.float32, grid)
    vs = tp.views(x)
    assert len(vs) == grid.size
    for die, v in enumerate(vs):
        assert v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        v.fill_(float(die))  # writes land in x through the view
    torch.cuda.synchronize()
    # every element was written exactly once, with its die's id
    counts = torch.bincount(x.flatten().long(), minlength=len(vs))
    assert counts.sum().item() == x.numel()
    assert (counts > 0).all()
    tp.release(x)


# -- 2. pointwise, in place, per die -----------------------------------------


@requires_cuda
def test_axpy_per_die(grid):
    """``y.add_(x, alpha=a)`` is the idiomatic torch AXPY; run per die it
    touches exactly the pages the die owns."""
    x = tp.localized_ones(SHAPE, torch.float32, grid)
    y = tp.localized_full(SHAPE, 2.0, torch.float32, grid)
    for xv, yv in zip(tp.views(x), tp.views(y)):
        yv.add_(xv, alpha=3.0)
    torch.cuda.synchronize()
    assert torch.equal(y, torch.full(SHAPE, 5.0, device="cuda"))
    tp.release(x)
    tp.release(y)


# -- 3. reductions over the SPLIT dim ----------------------------------------


@requires_cuda
def test_split_dim_reduction_is_partials_plus_fold(grid):
    # A global sum reduces OVER the split dim. The construct is per-die
    # partials (each die reduces its own elements into its own slot)
    # followed by a fold of the P partials.
    x = tp.localized_ones(SHAPE, torch.float32, grid)
    partials = torch.stack([v.sum() for v in tp.views(x)])
    total = partials.sum()
    torch.cuda.synchronize()
    assert total.item() == SHAPE[0] * SHAPE[1]
    assert partials.numel() == len(tp.views(x))  # one partial per grid place
    tp.release(x)


# -- 4. the motivator: an nn.Module with localized parameters ----------------


@requires_cuda
def test_localized_mlp_module(grid):
    """A pytorch-user-shaped module: weights are localized parameters, the
    forward is ordinary pytorch (placement-transparent tier), the in-place
    activation stage additionally runs per die over the views."""

    BATCH, D_IN, D_H = 64, 256, 512

    class TinyMLP(torch.nn.Module):
        # Sizes are kept small for test speed: placement *physics* needs
        # tensors past the 2 MiB block (see the README); the mechanism --
        # placed parameters, a placed activation buffer, per-die views for
        # the pointwise stage, plain matmuls for the rest -- is what this
        # shows.
        def __init__(self, grid):
            super().__init__()
            self.w1 = tp.localized_parameter((D_H, D_IN), torch.float32, grid)
            self.w2 = tp.localized_parameter((D_IN, D_H), torch.float32, grid)
            # the activation buffer is placed too (batch-blocked)
            self.h = tp.localized_empty((BATCH, D_H), torch.float32, grid)
            with torch.no_grad():
                self.w1.normal_(std=0.02)
                self.w2.normal_(std=0.02)

        def forward(self, x):
            # matmuls contract over a split dim: they run as ordinary
            # whole-device pytorch -- localized tensors are plain tensors --
            # writing into the placed buffer via out=
            torch.matmul(x, self.w1.t(), out=self.h)
            # the pointwise stage runs per die over the batch split
            for v in tp.views(self.h):
                v.relu_()
            return self.h @ self.w2.t()

    m = TinyMLP(grid)
    x = torch.randn(BATCH, D_IN, device="cuda")
    with torch.no_grad():
        out = m(x)
        ref = ((x @ m.w1.t()).relu_()) @ m.w2.t()
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5)
    # placement is discoverable on every placed piece
    assert tp.spec_of(m.w1) is not None
    assert tp.grid_of(m.h) is grid
