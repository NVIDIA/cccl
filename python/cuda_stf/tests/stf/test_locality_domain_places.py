# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locality-domain places from Python: count, scalar places, the per-device
domain grid, allocation, and task execution. On a device without native
locality-domain support every query degrades to a single whole-device
domain, so these tests adapt to the reported count instead of hardcoding
one."""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def test_count_never_zero():
    n = stf.locality_domain_count(0)
    assert n >= 1
    with pytest.raises(ValueError):
        stf.locality_domain_count(-1)


def test_scalar_places_construct():
    # (place equality is not exposed through the Python bindings; identity
    # semantics are covered by the C++ unittests)
    n = stf.locality_domain_count(0)
    for d in range(n):
        assert stf.exec_place.locality_domain(0, d) is not None
        assert stf.data_place.locality_domain(0, d) is not None


def test_domain_allocation_roundtrip():
    stf.machine_init()
    n = stf.locality_domain_count(0)
    nbytes = 4 << 20
    for d in range(n):
        dp = stf.data_place.locality_domain(0, d)
        ptr = dp.allocate(nbytes)
        assert ptr != 0
        dp.deallocate(ptr, nbytes)


def test_domain_grid_task():
    """A grid task over the device's locality domains, reading a logical
    data at the replicated place: the locality-domain counterpart of the
    replicated grid test."""
    stf.machine_init()
    grid = stf.exec_place_grid.locality_domains(0)
    n = stf.locality_domain_count(0)

    N = 512
    ctx = stf.context()
    X = np.arange(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_domains")

    rep = stf.data_place.replicated(grid)
    with ctx.task(grid, lX.read(rep)):
        pass

    results = []
    ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
    ctx.finalize()
    assert abs(results[0] - float(X.sum())) < 1e-4
    assert n >= 1


def test_domain_exec_place_task():
    """A plain task pinned to each domain in turn."""
    stf.machine_init()
    n = stf.locality_domain_count(0)

    N = 256
    ctx = stf.context()
    X = np.zeros(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_dom_exec")

    for d in range(n):
        with ctx.task(stf.exec_place.locality_domain(0, d), lX.rw()):
            pass

    ctx.finalize()


def test_machine_grid_granularities():
    """machine() covers the current machine at device or locality-domain
    granularity; the domain grid has sum(count(d)) places, device-major."""
    stf.machine_init()
    gdev = stf.exec_place_grid.machine()  # default: device granularity
    gdom = stf.exec_place_grid.machine(granularity="locality_domain")
    assert gdev is not None and gdom is not None

    from cuda.bindings import runtime as rt

    err, ndevs = rt.cudaGetDeviceCount()
    assert int(err) == 0
    expected = sum(stf.locality_domain_count(d) for d in range(ndevs))
    assert expected >= ndevs  # never fewer domains than devices

    with pytest.raises(ValueError, match="granularity"):
        stf.exec_place_grid.machine(granularity="warp")

    # machine() grids with more than one place carry a default BLOCKED
    # affine, so bare dependencies (no explicit data place) resolve
    # naturally -- data blocked along dim 0 over the grid. Single-place
    # machines deliberately get NO composite affine: make_grid degenerates
    # them to the (shared) scalar place, whose own device affine already
    # resolves bare deps -- and mutating the shared place's affine poisons
    # unrelated deactivate paths (found the hard way on GB300).
    N = 128
    ctx = stf.context()
    X = np.arange(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_machine")
    # bare rw resolves at the default blocked affine (multi-place grids)
    # or the scalar place's device affine (single-place machines)
    with ctx.task(gdom, lX.rw()):
        pass
    # explicit places compose with the default affine
    with ctx.task(gdom, lX.read(stf.data_place.replicated(gdom))):
        pass
    results = []
    ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
    ctx.finalize()
    assert abs(results[0] - float(X.sum())) < 1e-4


def test_sm_split_methods_construct():
    """Every SM split method builds usable places for every reported
    domain; unknown methods are rejected."""
    n = stf.locality_domain_count(0)
    for method in ("backfill", "aligned", "fine"):
        for d in range(n):
            p = stf.exec_place.locality_domain(0, d, sm_split=method)
            assert p is not None
        g = stf.exec_place_grid.locality_domains(0, sm_split=method)
        assert g is not None

    # The default is backfill: passing it explicitly is equivalent (both
    # construct; place equality is not exposed through the bindings).
    assert stf.exec_place.locality_domain(0, 0) is not None
    assert stf.exec_place.locality_domain(0, 0, sm_split="backfill") is not None

    with pytest.raises(ValueError, match="sm_split"):
        stf.exec_place.locality_domain(0, 0, sm_split="bogus")
    with pytest.raises(ValueError, match="sm_split"):
        stf.exec_place_grid.locality_domains(0, sm_split="bogus")


def test_sm_split_method_task():
    """A task pinned to each domain, for each SM split method."""
    stf.machine_init()
    n = stf.locality_domain_count(0)

    N = 256
    ctx = stf.context()
    X = np.zeros(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_dom_split")

    for method in ("backfill", "aligned", "fine"):
        for d in range(n):
            with ctx.task(
                stf.exec_place.locality_domain(0, d, sm_split=method), lX.rw()
            ):
                pass

    ctx.finalize()


def test_sm_split_method_grid_task():
    """A grid task over the device's domains built with a strict
    per-domain split method."""
    stf.machine_init()
    grid = stf.exec_place_grid.locality_domains(0, sm_split="fine")

    N = 512
    ctx = stf.context()
    X = np.arange(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_dom_split_grid")

    with ctx.task(grid, lX.read(stf.data_place.replicated(grid))):
        pass

    results = []
    ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
    ctx.finalize()
    assert abs(results[0] - float(X.sum())) < 1e-4
