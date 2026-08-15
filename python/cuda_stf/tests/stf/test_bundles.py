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
    return {name: d.mode for name, d in zip(bd.names, bd.deps)}


def test_bundle_modes_and_ceilings():
    ctx = stf.context()
    vals = np.zeros(8, dtype=np.float64)
    idx = np.arange(8, dtype=np.int32)
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

    ctx.finalize()


def test_task_slots():
    ctx = stf.context()
    vals = np.full(8, 2.0, dtype=np.float64)
    idx = np.arange(8, dtype=np.int32)
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


def test_bundle_adopts_and_registers():
    ctx = stf.context()
    lv = ctx.logical_data(np.zeros(4, dtype=np.float32))
    B = ctx.bundle(vals=lv, aux=np.ones(4, dtype=np.float32))
    assert B.vals is lv  # adopted, not re-registered
    with ctx.task(B.rw()) as t:
        assert set(t.get(0)._fields) == {"vals", "aux"}
    ctx.finalize()


def test_bundle_device_array_inference():
    da = stf.DeviceArray(8, np.dtype(np.float32), stf.data_place.device(0))
    ctx = stf.context()
    # registering a device CUDA-Array-Interface object must infer the device
    # data place (no host-pinning assertion)
    B = ctx.bundle(vals=da)
    with ctx.task(B.rw()) as t:
        assert t.get(0).vals.__cuda_array_interface__["shape"] == (8,)
    ctx.finalize()
