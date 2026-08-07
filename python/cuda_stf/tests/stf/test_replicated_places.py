# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for replicated data places: one copy of a logical data per member of
an execution grid, read-only at the place. The no-argument (deferred) form
binds its grid at task acquisition from the task's execution place; a scalar
execution place degenerates to affine.
"""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


class TestReplicatedDataPlace:
    def test_construct_with_grid(self):
        grid = stf.exec_place_grid.from_devices([0, 0])
        dp = stf.data_place.replicated(grid)
        assert dp is not None

    def test_construct_deferred(self):
        dp = stf.data_place.replicated()
        assert dp is not None

    def test_read_dep_on_grid_task(self):
        """A grid task reading at a replicated place sees the payload."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        rep = stf.data_place.replicated(grid)

        N = 512
        ctx = stf.context()
        X = np.arange(N, dtype=np.float32)
        lX = ctx.logical_data(X, name="X_replicated")

        with ctx.task(grid, lX.read(rep)):
            pass

        results = []
        ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
        ctx.finalize()
        assert abs(results[0] - float(X.sum())) < 1e-4

    def test_deferred_read_dep(self):
        """The deferred form binds to the task's execution place."""
        grid = stf.exec_place_grid.from_devices([0, 0])

        N = 256
        ctx = stf.context()
        X = np.arange(N, dtype=np.float32)
        lX = ctx.logical_data(X, name="X_replicated_deferred")

        with ctx.task(grid, lX.read(stf.data_place.replicated())):
            pass

        # scalar degenerate: same dep form on a single-place task
        with ctx.task(stf.exec_place.device(0), lX.read(stf.data_place.replicated())):
            pass

        ctx.finalize()

    def test_mutation_between_replicated_reads(self):
        """Mutate at another place: the next replicated read re-broadcasts.
        Runs on the stream and graph backends (in the latter the re-broadcast
        copies land inside the captured graph)."""
        for use_graph in (False, True):
            grid = stf.exec_place_grid.from_devices([0, 0])
            rep = stf.data_place.replicated(grid)

            N = 256
            ctx = stf.context(use_graph=use_graph)
            X = np.ones(N, dtype=np.float32)
            lX = ctx.logical_data(X, name="X_cycle")

            # generation 1 read at the replicated place
            with ctx.task(grid, lX.read(rep)):
                pass

            # mutate at another (host) place
            def bump(x):
                x[:] = x[:] + 41.0

            ctx.host_launch(lX.rw(), fn=bump)

            # generation 2 read: replicas must be re-broadcast
            with ctx.task(grid, lX.read(rep)):
                pass

            results = []
            ctx.host_launch(lX.read(), fn=lambda x: results.append(float(x.sum())))
            ctx.finalize()
            assert abs(results[0] - 42.0 * N) < 1e-3

    def test_write_rejected(self):
        """Replicated places are read-only: non-read deps are rejected."""
        grid = stf.exec_place_grid.from_devices([0, 0])
        rep = stf.data_place.replicated(grid)

        ctx = stf.context()
        X = np.zeros(64, dtype=np.float32)
        lX = ctx.logical_data(X)

        with pytest.raises(Exception, match="read"):
            with ctx.task(grid, lX.rw(rep)):
                pass
        ctx.finalize()
