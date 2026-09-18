# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Reduction example: __call__ (safe-by-default) vs execute() (explicit
invalidation) for repeated calls against the same op/h_init.
"""

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import OpKind

dtype = np.float32
h_init = np.zeros((), dtype=dtype)
d_output = cp.empty((), dtype=dtype)
d_input = cp.arange(100_000, dtype=dtype)

reducer = cuda.compute.make_reduce_into(
    d_in=d_input, d_out=d_output, op=OpKind.PLUS, h_init=h_init
)
temp_storage_size = reducer(
    temp_storage=None, d_in=d_input, d_out=d_output,
    num_items=d_input.size, op=OpKind.PLUS, h_init=h_init,
)
d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)

# Same op/h_init every iteration, only the input batch changes -- the
# realistic "hot loop calling the same reducer" case both paths below cover.
batches = [d_input * (i + 1) for i in range(5)]

# ---------------------------------------------------------------------------
# Before: __call__. Fully implicit, fully safe -- op and h_init are
# re-validated/re-marshaled every call, even though neither changes here.
# ---------------------------------------------------------------------------
for batch in batches:
    reducer(
        temp_storage=d_temp_storage, d_in=batch, d_out=d_output,
        num_items=batch.size, op=OpKind.PLUS, h_init=h_init,
    )
    assert float(d_output.get()) == float(cp.sum(batch).get())

# ---------------------------------------------------------------------------
# After: execute(). op/h_init are pinned once (at construction, from the
# make_reduce_into() call above) and never re-derived inside the loop --
# only d_in/d_out pointer state updates each call. This is the version that
# reaches parity with an equivalent CuPy call.
#
# Caller obligation: if op or h_init actually changes between iterations,
# you must call reducer.set_op(...)/reducer.set_h_init(...) BEFORE the
# execute() that should see the new value -- execute() will not notice on
# its own, and will silently keep using the old state.
# ---------------------------------------------------------------------------
for batch in batches:
    reducer.execute(
        temp_storage=d_temp_storage, d_in=batch, d_out=d_output,
        num_items=batch.size,
    )
    assert float(d_output.get()) == float(cp.sum(batch).get())

# If h_init needs to change partway through (e.g. a running accumulator
# reset), signal it explicitly -- this is the one line execute() needs that
# __call__ didn't:
h_init[...] = 10.0
reducer.set_h_init(h_init)      # <-- required; execute() alone would miss this
reducer.execute(
    temp_storage=d_temp_storage, d_in=d_input, d_out=d_output,
    num_items=d_input.size,
)

expected_result = float(cp.sum(d_input).get()) + 10.0
actual_result = float(d_output.get())
assert abs(actual_result - expected_result) < 1e-3 * abs(expected_result)
print("execute() example completed successfully")
