# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ruff: noqa: E402 — the v2-skip block below intentionally precedes the
# example's imports so the imports stay grouped at the start of the example
# body (after `# example-begin`).

# Ahead-of-time compilation is only supported on the default (v1) backend; the
# HostJIT (v2) backend raises NotImplementedError. Skip cleanly there so the
# example runner treats it as a pass.
import sys

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

if USING_V2:
    print("ahead-of-time build is unsupported on the HostJIT (v2) backend; skipping.")
    sys.exit(0)

# example-begin
"""
Compile a reduction ahead of time for multiple GPU architectures without a GPU
present, using dtype-only placeholders, and serialize the result for deployment.
"""

import numpy as np

from cuda.compute import OpKind, ProxyArray, ProxyValue, make_reduce_into, serialize

# ProxyArray / ProxyValue describe only the dtype of each argument; they hold no
# GPU memory, so no device is required to build. compute_capability must be given
# explicitly, since there is no device whose architecture we could default to.
reducer = make_reduce_into(
    d_in=ProxyArray(np.int32),
    d_out=ProxyArray(np.int32),
    op=OpKind.PLUS,
    h_init=ProxyValue(np.int32),
    compute_capability=[80, 90],  # build for sm_80 and sm_90
)

# Serialize the multi-architecture build for shipping to deployment targets. On
# a target GPU, cuda.compute.deserialize(blob) reconstructs the object, and the
# build result matching the running architecture is loaded on the first call.
blob = serialize(reducer)

assert len(blob) > 0
print(f"Compiled ahead of time for sm_80 and sm_90; serialized {len(blob)} bytes")
