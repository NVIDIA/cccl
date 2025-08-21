# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Application examples using scan algorithm.
"""

import math

import cupy as cp
import cupyx.scipy.special as cp_special
import numpy as np
from scipy.stats.distributions import binom

import cuda.cccl.parallel.experimental as parallel


def inclusive_segmented_sum_example():
    """Implement segmented scan using zip iterator and ordinary scan

    Segmented inclusive sum on array of values and head-flags
    array demarkating locations of start of segments can be implemented
    using ordinary inclusive scan using Schwarz operator acting
    of value-flag pairs. `ZipIterator` can be used to efficiently
    load data from pair of input arrays, instead of copying them
    to array of structs.

    For example, for data = [1, 1, 1, 1, 1, 1, 1, 1] with
    3 segments encoded by head_flags = [0, 0, 1, 0, 0, 1, 1, 0]
    corresponding to segmented data [[1, 1], [1, 1, 1], [1], [1, 1]],
    the expected prefix-sum values are [1, 2, 1, 2, 3, 1, 1, 2]
    """

    print("[Inclusive_segmented_sum example]")
    data = cp.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=cp.int64)
    hflg = cp.asarray([0, 0, 1, 0, 0, 1, 1, 0], dtype=cp.int32)

    @parallel.gpu_struct
    class ValueFlag:
        value: cp.int64
        flag: cp.int32

    def schwartz_sum(op1: ValueFlag, op2: ValueFlag) -> ValueFlag:
        f1: cp.int32 = 1 if op1.flag else 0
        f2: cp.int32 = 1 if op2.flag else 0
        f: cp.int32 = f1 | f2
        v: cp.int64 = op2.value if f2 else op1.value + op2.value
        return ValueFlag(v, f)

    zip_it = parallel.ZipIterator(data, hflg)
    d_output = cp.empty(data.shape, dtype=ValueFlag.dtype)
    h_init = ValueFlag(0, 0)

    parallel.inclusive_scan(zip_it, d_output, schwartz_sum, h_init, data.size)

    expected_prefix = np.asarray([1, 2, 1, 2, 3, 1, 1, 2], dtype=np.int64)
    result = d_output.get()

    assert np.array_equal(result["value"], expected_prefix)

    print(
        f"Inclusive segmented sum: computed result={result['value']}, "
        f"expected result={expected_prefix}"
    )
    return result


def logcdfs_from_logpdfs_example():
    """
    Given a vector of log-probabilities, compute a vector
    of logarithms of cumulative density function.

    Use log-add-exp binary operation to sidestep flush-to-zero
    and numerical exceptions computing logarithms of that.
    """
    print("[Compute logcdfs from logpdfs example]")
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

    def logaddexp(logp1: cp.float64, logp2: cp.float64):
        m_max = max(logp1, logp2)
        m_min = min(logp1, logp2)
        return m_max + math.log(1.0 + math.exp(m_min - m_max))

    logcdf = cp.empty_like(logpdf)
    h_init = np.array(-np.inf, dtype=np.float64)

    logcdf2 = cp.empty_like(logpdf)

    def maximum(v1: cp.float64, v2: cp.float64):
        return max(v1, v2)

    parallel.inclusive_scan(logpdf, logcdf, logaddexp, h_init, logpdf.size)

    # make sequence non-decreasing to resolve fast-math artifacts
    parallel.inclusive_scan(logcdf, logcdf2, maximum, h_init, logpdf.size)

    # check that it is non-increasing
    assert cp.all(logcdf2[:-1] <= logcdf2[1:])

    assert float(cp.max(logcdf2)) <= 0.0

    q25, q75 = cp.searchsorted(logcdf2, cp.asarray(np.log([0.25, 0.75])))

    q25_ref, q75_ref = binom(n, p).isf([0.75, 0.25])
    assert q25 == q25_ref
    assert q75 == q75_ref

    print(f"Quartiles of Binomial({n}, {p}) are {(q25, q75)}")
    print(f"CDF at quartiles: {[math.exp(logcdf[q25]), math.exp(logcdf[q75])]}")


if __name__ == "__main__":
    print("Running scan_applications examples")
    inclusive_segmented_sum_example()
    logcdfs_from_logpdfs_example()
    print("All examples completed successfully!")
