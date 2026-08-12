# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The localized programming model for pytorch users, as runnable examples.

The premise: allocate with a placement (``torch.localized.*`` factories),
compute with ``torch.localized.map`` -- one fused expression (eager, or a
stock ``torch.compile`` artifact) applied per die over exactly the elements
that die owns. Placement lives with the tensors; ``map`` infers it.

The examples walk the validity spectrum deliberately:
  1. trivially independent (fused pointwise chains) -- always valid;
  2. dim-wise ops along UNSPLIT dims (softmax over hidden with the batch
     split) -- valid, and the transformer inner-loop shape;
  3. reductions over SPLIT dims -- NOT a map: shown done right with
     per-die partials + a fold (the write-dual boundary, made explicit);
  4. misalignment -- rejected eagerly from registry metadata;
  5. CUDA-graph capture of the whole fork/join -- the "works in real
     life" requirement;
  6. an ``nn.Module`` with localized parameters end to end.
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


def _compiled(fn):
    """Stock torch.compile when triton is available, eager otherwise --
    map treats both identically (any callable)."""
    try:
        import triton  # noqa: F401, PLC0415

        return torch.compile(fn)
    except ImportError:
        return fn


# -- 1. trivially independent: a fused pointwise chain ----------------------


@requires_cuda
def test_map_pointwise_chain(grid):
    def body(x, y):
        # several pointwise ops; inductor fuses them into one kernel
        x.mul_(2.0).add_(y).relu_().sub_(0.5)

    x = tp.localized_empty(SHAPE, torch.float32, grid)
    y = tp.localized_empty(SHAPE, torch.float32, grid)
    x.normal_()
    y.normal_()
    ref_x = x.clone()  # plain tensor: the whole-device reference
    ref_y = y.clone()

    tp.map(_compiled(body), x, y)
    body(ref_x, ref_y)
    torch.cuda.synchronize()
    assert torch.equal(x, ref_x)

    tp.release(x)
    tp.release(y)


# -- 2. dim-wise ops along the UNSPLIT dim -----------------------------------


@requires_cuda
def test_map_softmax_over_unsplit_dim(grid):
    def body(x):
        # softmax over the hidden (unsplit) dim: every row lives whole
        # inside one die, so this is a valid map despite the reduction
        x.copy_(torch.softmax(x, dim=-1))

    x = tp.localized_empty(SHAPE, torch.float32, grid)
    x.normal_()
    ref = x.clone()

    tp.map(_compiled(body), x)
    body(ref)
    torch.cuda.synchronize()
    assert torch.allclose(x, ref, atol=1e-6)
    tp.release(x)


# -- 3. the boundary: reductions over the SPLIT dim --------------------------


@requires_cuda
def test_split_dim_reduction_is_partials_plus_fold(grid):
    # A global sum reduces OVER the split dim: not a map. The correct
    # construct is per-die partials (each die reduces its own elements
    # into its own slot) followed by a fold of the P partials.
    x = tp.localized_ones(SHAPE, torch.float32, grid)

    # per-die partials over the public per-die views ...
    partials = torch.stack([v.sum() for v in tp.views(x)])
    # ... then the fold of the P partials
    total = partials.sum()
    torch.cuda.synchronize()
    assert total.item() == SHAPE[0] * SHAPE[1]
    assert partials.numel() == len(tp.views(x))  # one partial per grid place
    tp.release(x)


# -- 4. misalignment is rejected eagerly -------------------------------------


@requires_cuda
def test_map_rejects_misaligned_operands(grid):
    x = tp.localized_empty(SHAPE, torch.float32, grid)  # default: blocked dim 0
    y = tp.localized_empty(
        SHAPE, torch.float32, grid, spec=(None, ("cyclic", 0))
    )
    with pytest.raises(ValueError, match="misaligned"):
        tp.map(lambda a, b: a.add_(b), x, y)
    with pytest.raises(ValueError, match="spec="):
        tp.map(lambda a: a.zero_(), torch.zeros(4, device="cuda"))
    tp.release(x)
    tp.release(y)


# -- 5. the whole fork/join is CUDA-graph capturable --------------------------


@requires_cuda
def test_map_captures_into_cuda_graph(grid):
    def body(x):
        x.mul_(3.0).add_(1.0)

    x = tp.localized_ones(SHAPE, torch.float32, grid)
    torch.cuda.synchronize()

    # warm-up outside capture (streams, lazy state)
    tp.map(body, x)
    torch.cuda.synchronize()
    x.fill_(1.0)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # the event-based fork/join is exactly the shape stream capture
        # follows: side streams fork from and join back into the capture
        # stream
        tp.map(body, x)
    x.fill_(1.0)
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    assert torch.all(x == 4.0)
    g.replay()
    torch.cuda.synchronize()
    assert torch.all(x == 13.0)  # replays compose: (1*3+1)*3+1
    tp.release(x)


# -- 6. the motivator: an nn.Module with localized parameters ----------------


@requires_cuda
def test_localized_mlp_module(grid):
    """A pytorch-user-shaped module: weights are localized parameters,
    the forward is ordinary pytorch (placement-transparent tier), the
    in-place activation stage additionally shows the map tier."""

    BATCH, D_IN, D_H = 64, 256, 512

    class TinyMLP(torch.nn.Module):
        # Sizes are kept small for test speed: placement *physics* needs
        # tensors past the 2 MiB block (see the README); the mechanism --
        # placed parameters, localized activations, map for the pointwise
        # stage, plain matmuls for the rest -- is what this shows.
        def __init__(self, grid):
            super().__init__()
            self.w1 = tp.localized_parameter((D_H, D_IN), torch.float32, grid)
            self.w2 = tp.localized_parameter((D_IN, D_H), torch.float32, grid)
            # the activation buffer is placed too (batch-blocked), so the
            # activation stage is a real map over aligned operands
            self.h = tp.localized_empty((BATCH, D_H), torch.float32, grid)
            with torch.no_grad():
                self.w1.normal_(std=0.02)
                self.w2.normal_(std=0.02)

        def forward(self, x):
            # matmuls are NOT maps (contraction over a split dim): they run
            # as ordinary whole-device pytorch -- localized tensors are
            # plain tensors -- writing into the placed buffer via out=
            torch.matmul(x, self.w1.t(), out=self.h)
            # the pointwise stage IS a map: per-die over the batch split
            tp.map(lambda t: t.relu_(), self.h)
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
