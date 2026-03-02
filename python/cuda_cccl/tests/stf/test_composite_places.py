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


if __name__ == "__main__":
    test_composite_from_devices()
    test_composite_from_places()
    test_composite_grid_dims()
    print("All composite place tests passed.")
