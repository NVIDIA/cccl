# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
