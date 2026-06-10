# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for composite data places: exec_place_grid and data_place.composite
with a Python partitioner.
"""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def blocked_mapper_1d(data_coords, data_dims, grid_dims):
    """Blocked partition along first dimension: maps data index to grid position."""
    n = data_dims[0]
    nplaces = grid_dims[0]
    part_size = max((n + nplaces - 1) // nplaces, 1)
    place_x = min(data_coords[0] // part_size, nplaces - 1)
    return (place_x, 0, 0, 0)


class TestExecPlaceGrid:
    def test_grid_from_devices(self):
        """exec_place_grid.from_devices creates a grid with correct size."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        assert grid.size == 2
        assert grid.dims[0] == 2

    def test_grid_from_devices_single(self):
        grid = stf.exec_place_grid.from_devices([0])
        assert grid.size == 1

    def test_grid_create_from_places(self):
        places = [stf.exec_place.device(0), stf.exec_place.device(0)]
        grid = stf.exec_place_grid.create(places)
        assert grid.size == 2

    def test_grid_create_with_dims(self):
        places = [stf.exec_place.device(0)] * 4
        grid = stf.exec_place_grid.create(places, grid_dims=(2, 2, 1, 1))
        assert grid.size == 4
        assert grid.dims == (2, 2, 1, 1)

    def test_grid_empty_raises(self):
        with pytest.raises(ValueError, match="at least one device"):
            stf.exec_place_grid.from_devices([])

    def test_grid_is_exec_place(self):
        grid = stf.exec_place_grid.from_devices([0])
        assert isinstance(grid, stf.exec_place)

    def test_scalar_exec_place_dims(self):
        ep = stf.exec_place.device(0)
        assert ep.size == 1
        assert ep.dims == (1, 1, 1, 1)


class TestCompositeDataPlace:
    def test_composite_basic(self):
        """data_place.composite creates a composite data place."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        assert dplace is not None

    def test_composite_non_callable_raises(self):
        grid = stf.exec_place_grid.from_devices([0])
        with pytest.raises(TypeError, match="callable"):
            stf.data_place.composite(grid, "not a function")

    def test_current_device_factories(self):
        ep = stf.exec_place.current_device()
        assert ep is not None
        dp = stf.data_place.current_device()
        assert dp is not None


class TestCompositeTask:
    def test_task_with_composite_dep(self):
        """Task uses a composite data place for its dependency."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)

        N = 1024
        ctx = stf.context()
        X = np.ones(N, dtype=np.float32)
        for i in range(N):
            X[i] = float(i)
        lX = ctx.logical_data(X, name="X_composite")

        with ctx.task(stf.exec_place.device(0), lX.rw(dplace)):
            pass

        ctx.finalize()
        for i in range(N):
            assert X[i] == float(i)

    def test_task_with_composite_dep_graph(self):
        """Same test in graph mode."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)

        N = 1024
        ctx = stf.context(use_graph=True)
        X = np.ones(N, dtype=np.float32)
        for i in range(N):
            X[i] = float(i)
        lX = ctx.logical_data(X, name="X_composite_graph")

        with ctx.task(stf.exec_place.device(0), lX.rw(dplace)):
            pass

        ctx.finalize()
        for i in range(N):
            assert X[i] == float(i)

    def test_affine_with_grid(self):
        """Grid with affine data place set; deps use the default (affine) placement."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        grid.set_affine_data_place(dplace)

        N = 512
        ctx = stf.context()
        X = np.arange(N, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(grid, lX.rw()):
            pass

        ctx.finalize()

    def test_grid_create_with_mapper(self):
        """exec_place_grid.create with mapper= sets affine automatically."""
        places = [stf.exec_place.device(0), stf.exec_place.device(0)]
        grid = stf.exec_place_grid.create(places, mapper=blocked_mapper_1d)

        N = 256
        ctx = stf.context()
        X = np.zeros(N, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(grid, lX.rw()):
            pass

        ctx.finalize()

    def test_host_launch_with_composite(self):
        """host_launch can read data placed via a composite data place."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        grid.set_affine_data_place(dplace)

        N = 64
        ctx = stf.context()
        X = np.arange(N, dtype=np.float64)
        lX = ctx.logical_data(X)

        results = []
        ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
        ctx.finalize()
        expected = float(np.arange(N, dtype=np.float64).sum())
        assert len(results) == 1
        assert abs(results[0] - expected) < 1e-6

    def test_task_on_exec_place_grid(self):
        """Task runs on an exec_place_grid; query grid dims and streams by index."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        grid.set_affine_data_place(dplace)

        ctx = stf.context()
        X = np.zeros(4, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(grid, lX.rw()) as t:
            dims = t.get_grid_dims()
            assert dims is not None
            assert dims == (2, 1, 1, 1)
            s0 = t.get_stream_at_index(0)
            s1 = t.get_stream_at_index(1)
            assert s0 is not None and s0 != 0
            assert s1 is not None and s1 != 0

        ctx.finalize()

    def test_task_on_grid_get_stream_ptrs(self):
        """get_stream_ptrs() returns one stream per grid place."""
        grid = stf.exec_place_grid.from_devices([0, 0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        grid.set_affine_data_place(dplace)

        ctx = stf.context()
        X = np.zeros(6, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(grid, lX.rw()) as t:
            ptrs = t.get_stream_ptrs()
            assert len(ptrs) == 3
            for p in ptrs:
                assert p != 0

        ctx.finalize()

    def test_task_get_grid_dims_none_for_scalar(self):
        """get_grid_dims() returns None when exec place is not a grid."""
        ctx = stf.context()
        X = np.zeros(4, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(stf.exec_place.device(0), lX.rw()) as t:
            assert t.get_grid_dims() is None

        ctx.finalize()

    def test_task_get_stream_ptrs_scalar_fallback(self):
        """get_stream_ptrs() returns a single-element list for non-grid tasks."""
        ctx = stf.context()
        X = np.zeros(4, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(stf.exec_place.device(0), lX.rw()) as t:
            ptrs = t.get_stream_ptrs()
            assert len(ptrs) == 1
            assert ptrs[0] != 0

        ctx.finalize()

    def test_task_on_grid_with_composite_dep(self):
        """Task on exec_place_grid; affine set so deps use lX.rw() without explicit dplace."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        dplace = stf.data_place.composite(grid, blocked_mapper_1d)
        grid.set_affine_data_place(dplace)

        ctx = stf.context()
        X = np.zeros(4, dtype=np.float32)
        lX = ctx.logical_data(X)

        with ctx.task(grid, lX.rw()) as t:
            dims = t.get_grid_dims()
            assert dims is not None
            assert dims == (2, 1, 1, 1)

        ctx.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
