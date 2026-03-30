# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for extensible exec_place: opaque handles, duck-typing via _as_stf_exec_place(),
and the dummy custom exec_place (proof of concept for external packages like ugpu-python).
"""

import numpy as np

import cuda.stf as stf


class PurePythonPlace:
    """A pure-Python class that wraps exec_place.device(0) via the duck-typing protocol."""

    def __init__(self, dev_id=0):
        self._dev_id = dev_id

    @property
    def kind(self):
        return "pure_python"

    def _as_stf_exec_place(self):
        return stf.exec_place.device(self._dev_id)


class NotAPlace:
    """An object that does NOT implement the exec_place protocol."""

    pass


def test_exec_place_protocol_check():
    """exec_place and pure-python duck-typed class satisfy ExecPlaceLike."""
    ep = stf.exec_place.device(0)
    assert isinstance(ep, stf.ExecPlaceLike)

    pp = PurePythonPlace(0)
    assert isinstance(pp, stf.ExecPlaceLike)

    bad = NotAPlace()
    assert not isinstance(bad, stf.ExecPlaceLike)


def test_duck_typed_place_in_task():
    """Pure Python class with _as_stf_exec_place() works in ctx.task()."""
    ctx = stf.context()
    arr = np.zeros(64, dtype=np.float32)
    ld = ctx.logical_data(arr)

    pp = PurePythonPlace(0)
    with ctx.task(pp, ld.rw()) as t:
        assert t.stream_ptr() != 0

    ctx.finalize()


def test_duck_typed_place_in_grid():
    """Pure Python duck-typed place works in exec_place_grid.create()."""
    places = [PurePythonPlace(0), stf.exec_place.device(0)]
    grid = stf.exec_place_grid.create(places)
    assert grid.kind == "grid"


def test_duck_typed_place_in_grid_task():
    """Grid containing duck-typed places works in a task (no data deps, just exec place)."""
    ctx = stf.context()

    places = [PurePythonPlace(0), PurePythonPlace(0)]
    grid = stf.exec_place_grid.create(places)

    with ctx.task(grid) as t:
        dims = t.get_grid_dims()
        assert dims is not None
        assert dims[0] == 2

    ctx.finalize()


def test_bad_type_rejected():
    """Passing an object without _as_stf_exec_place() to ctx.task() raises TypeError."""
    ctx = stf.context()
    arr = np.zeros(64, dtype=np.float32)
    ld = ctx.logical_data(arr)

    try:
        with ctx.task(NotAPlace(), ld.rw()) as _t:
            pass
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    ctx.finalize()


def test_bad_type_in_grid_rejected():
    """Passing a non-exec_place to exec_place_grid.create() raises TypeError."""
    try:
        stf.exec_place_grid.create([NotAPlace(), stf.exec_place.device(0)])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_exec_place_as_stf_exec_place():
    """Built-in exec_place._as_stf_exec_place() returns self."""
    ep = stf.exec_place.device(0)
    assert ep._as_stf_exec_place() is ep


def test_data_place_protocol_check():
    """data_place satisfies DataPlaceLike."""
    dp = stf.data_place.device(0)
    assert isinstance(dp, stf.DataPlaceLike)
    assert dp._as_stf_data_place() is dp
