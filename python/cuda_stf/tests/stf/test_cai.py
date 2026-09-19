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


class _StreamCAIWrapper:
    """Re-export another object's CUDA Array Interface with a producer stream."""

    def __init__(self, source, stream):
        self._source = source  # keep the exporter alive
        cai = dict(source.__cuda_array_interface__)
        cai["version"] = 3
        cai["stream"] = stream
        self.__cuda_array_interface__ = cai


@pytest.mark.parametrize("stream", [1, 2], ids=["legacy-default", "per-thread-default"])
def test_logical_data_synchronizes_producer_stream(stream):
    """A CAI producer stream is synchronized at registration (CAI v3 contract).

    STF does not order imported data behind an external stream asynchronously,
    so registration synchronizes the advertised stream; the buffer is coherent
    from then on.
    """
    dtype = np.dtype(np.float32)
    dplace = stf.data_place.device(0)

    ctx = stf.context()
    values = stf.DeviceArray(4, dtype, dplace)
    ld = ctx.logical_data(_StreamCAIWrapper(values, stream), dplace)
    assert ld.shape == (4,)
    ctx.finalize()


def test_logical_data_rejects_cai_stream_zero():
    """stream=0 is disallowed by the CAI v3 specification."""
    dtype = np.dtype(np.float32)
    dplace = stf.data_place.device(0)

    ctx = stf.context()
    values = stf.DeviceArray(4, dtype, dplace)
    try:
        with pytest.raises(ValueError, match="disallowed"):
            ctx.logical_data(_StreamCAIWrapper(values, 0), dplace)
    finally:
        ctx.finalize()
