# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from _utils.device_array import DeviceArray, get_compute_capability

from cuda.compute import (
    OpKind,
    binary_transform,
    gpu_struct,
    make_binary_transform,
    make_unary_transform,
    unary_transform,
)
from cuda.compute._cpp_compile import compile_cpp_op_code
from cuda.compute.op import RawOp

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = [
    pytest.mark.no_numba,
    pytest.mark.skipif(
        USING_V2, reason="fallback diagnostics are specific to the v1 backend"
    ),
]


@pytest.mark.parametrize("mode", ["execute", "build", "compile"])
@pytest.mark.parametrize("binary", [False, True])
def test_struct_builtin_reports_unsupported_type(mode, binary):
    build_only = mode != "execute"
    point = gpu_struct({"x": np.int32, "y": np.int32}, name="Point")
    d_in = DeviceArray.from_numpy(np.zeros(4, dtype=point.dtype))
    d_out = DeviceArray.empty((4,), point.dtype)
    if binary:
        invoke = make_binary_transform if build_only else binary_transform
        args = dict(d_in1=d_in, d_in2=d_in, d_out=d_out, op=OpKind.PLUS)
    else:
        invoke = make_unary_transform if build_only else unary_transform
        args = dict(d_in=d_in, d_out=d_out, op=OpKind.IDENTITY)
    if not build_only:
        args["num_items"] = 4
    if mode == "compile":
        args["compute_capability"] = get_compute_capability()
    with pytest.raises(
        RuntimeError,
        match=r"built-in operations are not supported for storage types .*custom operation implementation",
    ):
        invoke(**args)


def test_struct_raw_operator_still_copies():
    point = gpu_struct({"x": np.int32, "y": np.int32}, name="Point")
    values = np.array([(0, 10), (1, 11), (2, 12), (3, 13)], dtype=point.dtype)
    d_in = DeviceArray.from_numpy(values)
    d_out = DeviceArray.empty(values.shape, values.dtype)
    source = """
struct Point { int x; int y; };
extern "C" __device__ void copy_point(void* input, void* output) {
    *static_cast<Point*>(output) = *static_cast<const Point*>(input);
}
"""
    op = RawOp(name="copy_point", ltoir=compile_cpp_op_code(source))
    unary_transform(d_in=d_in, d_out=d_out, num_items=len(values), op=op)
    np.testing.assert_array_equal(d_out.copy_to_host(), values)


def test_unnamed_custom_operator_reports_missing_name():
    d_in = DeviceArray.from_numpy(np.arange(4, dtype=np.int32))
    d_out = DeviceArray.empty((4,), np.int32)
    with pytest.raises(
        RuntimeError, match="custom operation requires a non-empty function name"
    ):
        make_unary_transform(d_in=d_in, d_out=d_out, op=RawOp(name="", ltoir=b""))
