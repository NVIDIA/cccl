# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundles: grouped logical data used as a single dependency.

Mirrors the C++ conformance checklist (cudax/test/stf/interface/bundle.cu):
mode distribution against ceilings, loud error on explicit excess,
unspecified-fields-default-read, one submitted dependency = one slot, fields
remain first-class, mixed bundle + bare-field use in one task.
"""

import numpy as np
import pytest

import cuda.stf._experimental as stf
from cuda.stf._experimental.bundles import bundle_dep, constant

READ = stf.AccessMode.READ
RW = stf.AccessMode.RW
WRITE = stf.AccessMode.WRITE


def _modes(bd: bundle_dep):
    acquired_names = [n for n, got in zip(bd.names, bd.acquired) if got]
    return {name: d.mode for name, d in zip(acquired_names, bd.deps)}


def test_bundle_modes_and_ceilings():
    ctx = stf.context()
    vals = ctx.logical_data(np.zeros(8, dtype=np.float64))
    idx = ctx.logical_data(np.arange(8, dtype=np.int32))
    B = ctx.bundle(vals=vals, idx=constant(idx))

    # fields stay first-class logical data
    assert hasattr(B.vals, "read") and hasattr(B.vals, "rw")
    assert len(B) == 2

    # read: everything read
    assert _modes(B.read()) == {"vals": READ, "idx": READ}
    # rw distributes as the strongest admitted mode: constant clamps to read
    assert _modes(B.rw()) == {"vals": RW, "idx": READ}
    # write clamps constant fields to read (still fetched), mutable to write
    assert _modes(B.write()) == {"vals": WRITE, "idx": READ}
    # per-field spelling: unspecified fields default to read
    assert _modes(B.dep(vals=RW)) == {"vals": RW, "idx": READ}

    # explicit excess over a ceiling raises; unknown fields raise
    with pytest.raises(ValueError):
        B.dep(idx=RW)
    with pytest.raises(KeyError):
        B.dep(nope=READ)

    # NONE excludes a field: no dep, no transfer, view is None
    bd = B.dep(vals=RW, idx=stf.AccessMode.NONE)
    assert _modes(bd) == {"vals": RW}
    with ctx.task(bd) as t:
        g = t.get(0)
        assert g.idx is None and g.vals is not None

    ctx.finalize()


def test_task_slots():
    ctx = stf.context()
    vals = ctx.logical_data(np.full(8, 2.0, dtype=np.float64))
    idx = ctx.logical_data(np.arange(8, dtype=np.int32))
    out = np.zeros(8, dtype=np.float64)

    B = ctx.bundle(vals=vals, idx=constant(idx))
    lo = ctx.logical_data(out)

    # one bundle (2 fields) + one plain dep: bundle counts as ONE slot
    with ctx.task(B.read(), lo.rw()) as t:
        g = t.get(0)
        assert set(g._fields) == {"vals", "idx"}
        v = g.vals.__cuda_array_interface__
        o = t.get(1).__cuda_array_interface__
        assert v["shape"] == (8,) and o["shape"] == (8,)
        assert v["data"][0] != o["data"][0]

    # mixed use: bundle dep + bare dep on one of its own fields
    with ctx.task(B.read(), B.vals.rw()) as t:
        assert t.get(1).__cuda_array_interface__["shape"] == (8,)

    ctx.finalize()


def test_bundle_adopts_handles_only():
    ctx = stf.context()
    lv = ctx.logical_data(np.zeros(4, dtype=np.float32))
    la = ctx.logical_data(np.ones(4, dtype=np.float32))
    B = ctx.bundle(vals=lv, aux=la)
    assert B.vals is lv  # adopted: the same handle, nothing copied
    with ctx.task(B.rw()) as t:
        assert set(t.get(0)._fields) == {"vals", "aux"}
    # bundles group handles, they do not register arrays
    with pytest.raises(TypeError):
        ctx.bundle(vals=np.zeros(4))
    ctx.finalize()


def test_bundle_device_logical_data():
    dplace = stf.data_place.device(0)
    da = stf.DeviceArray(8, np.dtype(np.float32), dplace)
    ctx = stf.context()
    B = ctx.bundle(vals=ctx.logical_data(da, dplace))
    with ctx.task(B.rw()) as t:
        assert t.get(0).vals.__cuda_array_interface__["shape"] == (8,)
    ctx.finalize()


def test_stackable_bundle_tokens():
    """Bundles over stackable contexts: token bundles + launchable replay."""
    ctx = stf.stackable_context()
    B = ctx.bundle(rigid=ctx.token(), soft=ctx.token())
    lx = ctx.logical_data(np.zeros(8, dtype=np.float64))

    # whole-bundle modes distribute over the token fields; one slot each
    with ctx.task(B.rw(), lx.rw()) as t:
        # tokens are ordering-only: the bundle slot exists, its views are CAIs
        # of the token deps' (empty) payloads; the plain dep is slot 1
        assert t.get(1).__cuda_array_interface__["shape"] == (8,)

    # per-field spelling with exclusion works identically to plain contexts
    bd = B.dep(rigid=stf.AccessMode.READ, soft=stf.AccessMode.NONE)
    assert bd.acquired == [True, False]
    with ctx.task(bd, lx.read()):
        pass

    ctx.finalize()
