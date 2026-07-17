# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA Array Interface metadata tests for STF task arguments."""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def test_get_arg_cai_preserves_structured_dtype_descr():
    dtype = np.dtype([("x", np.float32), ("y", np.float32)])

    ctx = stf.context()
    dplace = stf.data_place.device(0)
    values = stf.DeviceArray(4, dtype, dplace)
    assert values.__cuda_array_interface__["descr"] == dtype.descr

    ld = ctx.logical_data(values, dplace)
    with ctx.task(ld.rw(dplace)) as t:
        cai = t.get_arg_cai(0).__cuda_array_interface__

    ctx.finalize()

    assert cai["typestr"].startswith("|V")
    assert cai["descr"] == dtype.descr
    assert np.dtype(cai["descr"]) == dtype


class _FakeCAI:
    """Minimal CUDA Array Interface exporter with a configurable stream."""

    def __init__(self, stream):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (4,),
            "typestr": "<f4",
            "data": (1 << 20, False),
            "strides": None,
            "stream": stream,
        }


@pytest.mark.parametrize("stream", [1, 2, 12345])
def test_logical_data_rejects_producer_stream(stream):
    """Importing a CAI object that advertises a producer stream is rejected.

    STF does not yet order imported data behind an external producer stream, so
    rather than silently racing we raise until that plumbing exists.
    """
    ctx = stf.context()
    try:
        with pytest.raises(NotImplementedError, match="producer stream"):
            ctx.logical_data(_FakeCAI(stream), stf.data_place.device(0))
    finally:
        ctx.finalize()
