# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
"""
Given a vector of log-probabilities, compute a vector of logarithms of cumulative density function.
"""

import math

import cupy as cp
import cupyx.scipy.special as cp_special
import numpy as np

import cuda.compute

# Prepare the input data and compute log-probabilities.
# Use log-add-exp binary operation to sidestep flush-to-zero
# and numerical exceptions computing logarithms of that.
n = 500
p = 0.31
m = cp.arange(n + 1, dtype=cp.float64)
nm = n - m
lognorm = (
    cp_special.loggamma(1 + n)
    - cp_special.loggamma(1 + m)
    - cp_special.loggamma(1 + nm)
)
logpdf = lognorm + m * cp.log(p) + nm * cp.log1p(-p)

assert n + 1 == logpdf.size

# Define the binary operations for the scans.


def logaddexp(logp1: cp.float64, logp2: cp.float64):
    m_max = max(logp1, logp2)
    m_min = min(logp1, logp2)
    return m_max + math.log(1.0 + math.exp(m_min - m_max))


def maximum(v1: cp.float64, v2: cp.float64):
    return max(v1, v2)


# Prepare the output arrays and initial value.
logcdf = cp.empty_like(logpdf)
h_init = np.array(-np.inf, dtype=np.float64)

logcdf2 = cp.empty_like(logpdf)

# Perform the first inclusive scan (log-add-exp) using the object API.
scanner_logaddexp = cuda.compute.make_inclusive_scan(logpdf, logcdf, logaddexp, h_init)
temp_storage_bytes = int(
    scanner_logaddexp.get_temp_storage_bytes(
        logpdf,
        logcdf,
        logpdf.size,
        init_value=h_init,
        op=logaddexp,
    )
)
d_temp_storage = None if temp_storage_bytes == 0 else cp.empty(temp_storage_bytes, dtype=np.uint8)
scanner_logaddexp.compute(
    d_temp_storage,
    logpdf,
    logcdf,
    logpdf.size,
    init_value=h_init,
    op=logaddexp,
)

# Perform the second inclusive scan (maximum) using the object API.
scanner_maximum = cuda.compute.make_inclusive_scan(logcdf, logcdf2, maximum, h_init)
temp_storage_bytes_2 = int(
    scanner_maximum.get_temp_storage_bytes(
        logcdf,
        logcdf2,
        logpdf.size,
        init_value=h_init,
        op=maximum,
    )
)
d_temp_storage_2 = None if temp_storage_bytes_2 == 0 else cp.empty(temp_storage_bytes_2, dtype=np.uint8)
scanner_maximum.compute(
    d_temp_storage_2,
    logcdf,
    logcdf2,
    logpdf.size,
    init_value=h_init,
    op=maximum,
)

# Verify the results and compute quantiles.
assert cp.all(logcdf2[:-1] <= logcdf2[1:])

assert float(cp.max(logcdf2)) <= 0.0

q25, q75 = cp.searchsorted(logcdf2, cp.asarray(np.log([0.25, 0.75])))

try:
    from scipy.stats.distributions import binom

    q25_ref, q75_ref = binom(n, p).isf([0.75, 0.25])
    assert q25 == q25_ref
    assert q75 == q75_ref
except ImportError:
    print("scipy not found, skipping assertions")

print(f"Log CDF example completed. q25: {q25}, q75: {q75}")
