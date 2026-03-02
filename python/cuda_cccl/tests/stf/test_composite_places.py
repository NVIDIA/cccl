# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for composite data places: exec_place_grid and data_place.composite with a Python partitioner.
"""

import numpy as np

import cuda.stf as stf


def blocked_mapper_1d(data_coords, data_dims, grid_dims):
    """Blocked partition along first dimension: maps data index to grid position."""
    n = data_dims[0]
    nplaces = grid_dims[0]
    part_size = max((n + nplaces - 1) // nplaces, 1)
    place_x = min(data_coords[0] // part_size, nplaces - 1)
    return (place_x, 0, 0, 0)


def test_composite_from_devices():
    """Composite data place with grid from device IDs (same device repeated for single-GPU)."""
    grid = stf.exec_place_grid.from_devices([0, 0, 0])
    assert grid is not None

    dplace = stf.data_place.composite(grid, blocked_mapper_1d)
    assert dplace.kind == "composite"

    N = 1024
    ctx = stf.context()
    X = np.ones(N, dtype=np.float32)
    for i in range(N):
        X[i] = float(i)

    lX = ctx.logical_data(X, name="X_composite")
    with ctx.task(stf.exec_place.device(0), lX.rw(dplace)) as t:
        pass

    ctx.finalize()

    for i in range(N):
        assert X[i] == float(i)
    del ctx


def test_composite_from_places():
    """Composite data place with grid from list of exec_places."""
    places = [stf.exec_place.device(0), stf.exec_place.device(0)]
    grid = stf.exec_place_grid.create(places)
    assert grid is not None

    dplace = stf.data_place.composite(grid, blocked_mapper_1d)
    assert dplace.kind == "composite"

    N = 512
    ctx = stf.context()
    X = np.arange(N, dtype=np.float32)
    lX = ctx.logical_data(X)
    with ctx.task(stf.exec_place.device(0), lX.rw(dplace)) as t:
        pass

    ctx.finalize()
    del ctx


def test_composite_grid_dims():
    """Composite with explicit grid shape (e.g. 2x2)."""
    places = [stf.exec_place.device(0)] * 4
    grid = stf.exec_place_grid.create(places, grid_dims=(2, 2, 1, 1))
    assert grid is not None

    dplace = stf.data_place.composite(grid, blocked_mapper_1d)
    assert dplace.kind == "composite"

    N = 256
    ctx = stf.context()
    X = np.zeros(N, dtype=np.float32)
    lX = ctx.logical_data(X)
    with ctx.task(stf.exec_place.device(0), lX.rw(dplace)) as t:
        pass

    ctx.finalize()
    del ctx


def test_task_on_exec_place_grid():
    """Task runs on an exec_place_grid; query grid dims and streams by index.
    Uses a composite data place (required when exec place is a grid).
    """
    grid = stf.exec_place_grid.from_devices([0, 0])  # same device, 2 places
    dplace = stf.data_place.composite(grid, blocked_mapper_1d)
    assert dplace.kind == "composite"
    ctx = stf.context()
    X = np.zeros(4, dtype=np.float32)
    lX = ctx.logical_data(X)
    with ctx.task(grid, lX.rw(dplace)) as t:
        dims = t.get_grid_dims()
        assert dims is not None
        assert dims == (2, 1, 1, 1)
        s0 = t.get_stream_at_index(0)
        s1 = t.get_stream_at_index(1)
        assert s0 is not None and s0 != 0
        assert s1 is not None and s1 != 0
    ctx.finalize()
    del ctx


def test_task_on_grid_with_composite_dep():
    """Task on exec_place_grid with a composite data-place dep (required when exec place is a grid)."""
    grid = stf.exec_place_grid.from_devices([0, 0])
    dplace = stf.data_place.composite(grid, blocked_mapper_1d)
    assert dplace.kind == "composite"
    ctx = stf.context()
    X = np.zeros(4, dtype=np.float32)
    lX = ctx.logical_data(X)
    with ctx.task(grid, lX.rw(dplace)) as t:
        dims = t.get_grid_dims()
        assert dims is not None
        assert dims == (2, 1, 1, 1)
    ctx.finalize()
    del ctx


def test_exec_place_from_grid():
    """exec_place.from_grid(grid) produces an exec_place with kind 'grid'."""
    grid = stf.exec_place_grid.from_devices([0])
    ep = stf.exec_place.from_grid(grid)
    assert ep.kind == "grid"
    del grid


if __name__ == "__main__":
    test_composite_from_devices()
    test_composite_from_places()
    test_composite_grid_dims()
    test_task_on_exec_place_grid()
    test_task_on_grid_with_composite_dep()
    test_exec_place_from_grid()
    print("All composite place tests passed.")
