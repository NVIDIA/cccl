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

import cuda.cccl.parallel.experimental as parallel

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

# Perform the first inclusive scan (log-add-exp).
parallel.inclusive_scan(logpdf, logcdf, logaddexp, h_init, logpdf.size)

# Perform the second inclusive scan (maximum).
parallel.inclusive_scan(logcdf, logcdf2, maximum, h_init, logpdf.size)

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
