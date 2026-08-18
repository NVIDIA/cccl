# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Localized allocation through ``interop.pytorch``: torch tensors backed by
composite VMM data places (structured spec tier + callback escape hatch)."""

import pytest

pytest.importorskip("cuda.stf._experimental._stf_bindings")
torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop import pytorch as tp  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

N_PLACES = 2
# page-aligned outer rows so placement fidelity is exact at 2 MiB VMM pages
SHAPE = (8, 64, 16384)  # row = 64*16384*2B = 2 MiB (bf16/f16)


@pytest.fixture()
def grid():
    stf.machine_init()
    places = [stf.exec_place.device(0)] * N_PLACES
    return stf.exec_place_grid.create(places)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_structured_alloc_roundtrip(grid, dtype):
    t = tp.localized_empty(SHAPE, dtype, grid)
    src = torch.randn(SHAPE, dtype=dtype, device="cuda")
    t.copy_(src)
    torch.cuda.synchronize()
    assert torch.equal(t, src)
    assert t.dtype == dtype and tuple(t.shape) == SHAPE


@requires_cuda
def test_meta_survives_views_and_parameter(grid):
    t = tp.localized_empty(SHAPE, torch.float16, grid)
    meta = tp.get_meta(t)
    assert meta is not None and meta.partition is not None
    # views, reshapes, and Parameter wrapping keep the same storage
    assert tp.get_meta(t.view(-1)) is meta
    assert tp.get_meta(t.reshape(SHAPE[0], -1)) is meta
    assert tp.get_meta(torch.nn.Parameter(t, requires_grad=False)) is meta
    # plain tensors carry no meta
    assert tp.get_meta(torch.zeros(4, device="cuda")) is None


@requires_cuda
def test_registry_pinned_until_release(grid):
    """Allocations are registry-pinned (weights lifetime); release() evicts.
    NB: this test exposed that the consumer prototypes' finalize-on-buffer
    was unreachable (the registry kept the buffer alive) — pinning +
    explicit release are the honest semantics for CAI-imported storage."""
    before = len(tp.live_metas())
    t = tp.localized_empty((4, 64, 16384), torch.float16, grid)
    assert len(tp.live_metas()) == before + 1
    tp.release(t)
    assert len(tp.live_metas()) == before
    with pytest.raises(ValueError):
        tp.release(t)


@requires_cuda
def test_placement_report_and_tier_parity(grid):
    """Structured tier and the callback escape hatch place identically for
    the blocked policy (the localization-lab parity claim, upstreamed)."""
    t_spec = tp.localized_empty(SHAPE, torch.float16, grid)
    row_bytes = SHAPE[1] * SHAPE[2] * 2

    def blocked_rows(data_coords, data_dims, grid_dims):
        n = grid_dims[0]
        rows = data_dims[0] // row_bytes
        r = data_coords[0] // row_bytes
        chunk = -(-rows // n)
        return (min(r // chunk, n - 1),)

    t_map = tp.localized_empty(SHAPE, torch.float16, grid, mapper=blocked_rows)
    s1 = tp.placement_report(t_spec)
    s2 = tp.placement_report(t_map)
    assert list(s1.bytes_per_grid_index) == list(s2.bytes_per_grid_index)
    assert s1.accuracy == s2.accuracy == 1.0


@requires_cuda
def test_parameter_and_spec_mapper_exclusive(grid):
    p = tp.localized_parameter((4, 64, 16384), torch.float16, grid)
    assert isinstance(p, torch.nn.Parameter) and not p.requires_grad
    with pytest.raises(ValueError, match="not both"):
        tp.localized_empty(
            SHAPE,
            torch.float16,
            grid,
            spec=(("blocked", 0), None, None),
            mapper=lambda c, d, g: (0,),
        )


# -- gc lifetime (DLPack tier) -------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gc_structured_roundtrip(grid, dtype):
    """lifetime="gc" allocates through DLPack; the view chain (storage-dtype
    for bf16, then shape) works identically to the pinned tier."""
    t = tp.localized_empty(SHAPE, dtype, grid, lifetime="gc")
    src = torch.randn(SHAPE, dtype=dtype, device="cuda")
    t.copy_(src)
    torch.cuda.synchronize()
    assert torch.equal(t, src)
    meta = tp.get_meta(t)
    assert meta is not None and meta.lifetime == "gc" and meta.partition is not None
    assert not meta._keepalive  # nothing pinned: the storage owns the buffer
    tp.release(t)  # early metadata drop is allowed and harmless


@requires_cuda
def test_gc_callback_roundtrip(grid):
    t = tp.localized_empty(
        (4, 64, 16384),
        torch.float16,
        grid,
        mapper=lambda c, d, g: (0,),
        lifetime="gc",
    )
    t.fill_(3.0)
    torch.cuda.synchronize()
    assert float(t.sum(dtype=torch.float64)) == 3.0 * t.numel()


@requires_cuda
def test_gc_registry_self_evicts(grid):
    """The gc tier's registry finalizer is REACHABLE (nothing pins the
    buffer): when the last tensor view dies, the storage frees the VMM and
    the metadata evicts itself — no release() call, no leak on unload."""
    import gc
    import weakref

    t = tp.localized_empty((4, 64, 16384), torch.float16, grid, lifetime="gc")
    p = torch.nn.Parameter(t, requires_grad=False)
    meta_ref = weakref.ref(tp.get_meta(t))
    assert meta_ref() is not None and tp.get_meta(p) is meta_ref()
    assert meta_ref() in tp.live_metas()
    del t, p
    gc.collect()
    # the registry entry (the meta's only strong holder) is gone
    assert meta_ref() is None


@requires_cuda
def test_gc_and_pinned_place_identically(grid):
    """The lifetime choice is orthogonal to placement: both tiers produce
    the same bytes-per-place decision."""
    t_pin = tp.localized_empty(SHAPE, torch.float16, grid)
    t_gc = tp.localized_empty(SHAPE, torch.float16, grid, lifetime="gc")
    s1, s2 = tp.placement_report(t_pin), tp.placement_report(t_gc)
    assert list(s1.bytes_per_grid_index) == list(s2.bytes_per_grid_index)
    assert s1.accuracy == s2.accuracy == 1.0
    tp.release(t_pin)


@requires_cuda
def test_localized_parameter_defaults_to_gc(grid):
    """A parameter registered on a module is the idiomatic owner: dropping
    the module frees the allocation and the metadata."""
    import gc
    import weakref

    m = torch.nn.Module()
    m.w = tp.localized_parameter((4, 64, 16384), torch.float16, grid)
    meta_ref = weakref.ref(tp.get_meta(m.w))
    assert meta_ref().lifetime == "gc"
    m.w.data.fill_(1.0)
    torch.cuda.synchronize()
    del m
    gc.collect()
    assert meta_ref() is None  # module unload freed everything


@requires_cuda
def test_invalid_lifetime_rejected(grid):
    with pytest.raises(ValueError, match="lifetime"):
        tp.localized_empty(SHAPE, torch.float16, grid, lifetime="forever")


# ---------------------------------------------------------------------------
# Factory family: zeros / ones / full and the *_like variants
# ---------------------------------------------------------------------------


@requires_cuda
def test_localized_factories_values(grid):
    z = tp.localized_zeros((64, 32), torch.float32, grid)
    o = tp.localized_ones((64, 32), torch.float32, grid)
    f = tp.localized_full((64, 32), 3.5, torch.float32, grid)
    torch.cuda.synchronize()
    assert torch.all(z == 0) and torch.all(o == 1) and torch.all(f == 3.5)
    for t in (z, o, f):
        assert tp.get_meta(t) is not None
        tp.release(t)


@requires_cuda
def test_localized_like_reuses_placement(grid):
    t = tp.localized_empty(
        SHAPE, torch.float16, grid, spec=(None, ("blocked", 0), None)
    )
    meta = tp.get_meta(t)

    z = tp.localized_zeros_like(t)
    zmeta = tp.get_meta(z)
    # the partition OBJECT is reused, not rebuilt
    assert zmeta.partition is meta.partition
    assert zmeta.shape == meta.shape and zmeta.dtype == meta.dtype
    torch.cuda.synchronize()
    assert torch.all(z == 0)

    # dtype override keeps the placement (partition is element-indexed)
    h = tp.localized_empty_like(t, dtype=torch.float32)
    assert tp.get_meta(h).partition is meta.partition
    assert h.dtype == torch.float32

    # in-place torch init works on any localized tensor (no first-touch
    # placement semantics: pages are placed at allocation)
    h.normal_()
    torch.cuda.synchronize()

    for x in (t, z, h):
        tp.release(x)


@requires_cuda
def test_localized_like_rejects_non_localized(grid):
    plain = torch.zeros(8, device="cuda")
    with pytest.raises(ValueError, match="localized"):
        tp.localized_empty_like(plain)


@requires_cuda
def test_localized_empty_accepts_prebuilt_partition(grid):
    part = stf.cute_partition.from_spec((256,), (("blocked", 0),), (N_PLACES,))
    t = tp.localized_empty((256,), torch.float32, grid, spec=part)
    assert tp.get_meta(t).partition is part
    with pytest.raises(ValueError, match="true_dims"):
        tp.localized_empty((128,), torch.float32, grid, spec=part)
    tp.release(t)


# ---------------------------------------------------------------------------
# torch.localized convenience namespace (no GPU needed: pure patching)
# ---------------------------------------------------------------------------


def test_install_uninstall_torch_localized():
    ns = tp.install()
    try:
        assert torch.localized is ns
        assert torch.localized.empty is tp.localized_empty
        assert torch.localized.zeros_like is tp.localized_zeros_like
        # import machinery works through the sys.modules entry
        from torch.localized import zeros  # noqa: PLC0415

        assert zeros is tp.localized_zeros
        # idempotent
        assert tp.install() is not None
    finally:
        tp.uninstall()
    assert not hasattr(torch, "localized")


def test_install_refuses_foreign_attribute():
    torch.localized = object()
    try:
        with pytest.raises(RuntimeError, match="does not belong"):
            tp.install()
        with pytest.raises(RuntimeError, match="not removing"):
            tp.uninstall()
    finally:
        del torch.localized


def test_namespace_without_patching():
    ns = tp.namespace()
    assert ns.full is tp.localized_full
    assert not hasattr(torch, "localized")


def test_attribute_chain_and_laziness():
    import sys

    # one import is enough: stf.interop.pytorch resolves lazily
    assert stf.interop.pytorch.install is tp.install
    assert stf.interop.pytorch.localized_zeros is tp.localized_zeros
    # sibling adapters are NOT imported by touching the chain
    assert "cuda.stf._experimental.interop.numba" not in sys.modules


@requires_cuda
def test_spec_and_grid_accessors(grid):
    t = tp.localized_empty(SHAPE, torch.float16, grid)
    assert tp.grid_of(t) is grid
    assert tp.spec_of(t) is tp.get_meta(t).partition
    # resolves through views and Parameter wrapping (storage-keyed)
    assert tp.spec_of(t.view(-1)) is tp.spec_of(t)
    assert tp.grid_of(torch.nn.Parameter(t, requires_grad=False)) is grid
    with pytest.raises(ValueError, match="registered"):
        tp.spec_of(torch.zeros(4, device="cuda"))
    tp.release(t)
