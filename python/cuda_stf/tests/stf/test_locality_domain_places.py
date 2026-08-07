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

    # machine() grids carry a default BLOCKED affine, so bare
    # dependencies (no explicit data place) resolve naturally -- data
    # blocked along dim 0 over the grid. (A create()-built grid without a
    # mapper still has no affine; deps there need explicit places.)
    #
    # KNOWN C++ ISSUE (found on GB300, 2026-08-08): a replicated read whose
    # valid source instance lives at a COMPOSITE place terminates in
    # exec_place deactivate ("invalid device ordinal", places.cuh device
    # impl) -- the broadcast-copy path derives a device restore ordinal
    # from a composite affine. Reproducer: rw at the machine grid's
    # composite affine, then read(data_place.replicated(grid)). Until
    # fixed, this test keeps the replicated read sourced from the host
    # instance (before any composite write), like test_domain_grid_task.
    N = 128
    ctx = stf.context()
    X = np.arange(N, dtype=np.float32)
    lX = ctx.logical_data(X, name="X_machine")
    # replicated read first: source instance is the host-provided one
    with ctx.task(gdom, lX.read(stf.data_place.replicated(gdom))):
        pass
    # bare rw resolves at the default blocked affine
    with ctx.task(gdom, lX.rw()):
        pass
    results = []
    ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
    ctx.finalize()
    assert abs(results[0] - float(X.sum())) < 1e-4
