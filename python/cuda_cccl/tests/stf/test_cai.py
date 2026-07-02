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
