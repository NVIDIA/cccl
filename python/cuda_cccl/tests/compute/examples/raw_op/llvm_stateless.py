# Copyright (c) 2026 NVIDIA CORPORATION.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ruff: noqa: E402 — the v1-skip block below intentionally precedes the
# example's imports so the imports stay grouped at the start of the example
# body (after `# example-begin`).

# Skip cleanly on v1 — the LLVM-bitcode code-path (DeviceCode(kind="llvm_ir"))
# requires cccl.c.parallel built with CCCL_PYTHON_USE_V2=ON. On a v1 wheel the
# binding silently treats it as LTO-IR and nvJitLink rejects the bitcode bytes
# as malformed LTO-IR.
import sys

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

if not USING_V2:
    print("llvm_stateless requires cccl.c.parallel v2; skipping.")
    sys.exit(0)

# example-begin
"""
Create a custom operator from LLVM bitcode using RawOp.

This example demonstrates how to supply pre-compiled LLVM bitcode to RawOp,
which is the preferred path for cccl.parallel v2 because the bitcode is
linked into the CUB module at the LLVM IR level and the optimizer inlines
the operator through kernel inner loops.
"""

import cupy as cp
import llvmlite.binding as llvm
import numpy as np

import cuda.compute
from cuda.compute._device_code import DeviceCode
from cuda.compute.op import RawOp

# Hand-written LLVM IR for an extern "C" multiply operator with the
# void(void*, void*, void*) ABI RawOp expects. llvmlite parses the text and
# serializes it to LLVM bitcode (the binary form, starting with magic "BC")
# that the v2 backend's LLVM linker accepts.
llvm_ir = """
target triple = "nvptx64-nvidia-cuda"

define void @multiply_op(ptr %a, ptr %b, ptr %result) {
entry:
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %r = mul i32 %x, %y
  store i32 %r, ptr %result, align 4
  ret void
}
"""

mod = llvm.parse_assembly(llvm_ir)
mod.verify()
bitcode = bytes(mod.as_bitcode())

# Wrap the bitcode in DeviceCode so RawOp knows to treat it as LLVM bitcode
# rather than the default LTO-IR. (Raw `bytes` is accepted too and treated
# as LTO-IR — the legacy form.)
multiply_op = RawOp(
    ltoir=DeviceCode(bytes_=bitcode, kind="llvm_ir"),
    name="multiply_op",
)

h_input = np.array([1, 2, 3, 4, 5], dtype=np.int32)
d_input = cp.array(h_input)
d_output = cp.empty(1, dtype=np.int32)
h_init = np.array(1, dtype=np.int32)

cuda.compute.reduce_into(
    d_in=d_input, d_out=d_output, num_items=len(d_input), op=multiply_op, h_init=h_init
)

result = d_output.get()[0]
expected = np.prod(h_input)  # 1 * 2 * 3 * 4 * 5 = 120
assert result == expected, f"Expected {expected}, got {result}"

print(f"Custom multiply reduction result: {result}")
print("RawOp LLVM-IR example completed successfully!")
