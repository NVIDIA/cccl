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
    print("[Begin inclusive_segmented_sum example]")
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
    print("[End inclusive_segmented_sum example]")
    return result


def logcdfs_from_logpdfs_example():
    """
    Given a vector of log-probabilities, compute a vector
    of logarithms of cumulative density function.

    Use log-add-exp binary operation to sidestep flush-to-zero
    and numerical exceptions computing logarithms of that.
    """
    print("[Begin logCDFs from logPDFs example]")
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
    print("[End logCDFs from logPDFs example]")


def exponential_moving_average_using_scan_example():
    """
    Given input vector u and smoothing parameter 0<alpha<1
    the exponential moving average sequence ma is defined
    by ma[t] = (1 - alpha) * u[t] + alpha * ma[t-1] with
    initial condition ma[0] = u[0].

    This recurrence equation admits a closed form solution:

    ma[t] = alpha ** (t + 1) * u[0] + (1 - alpha) * (alpha ** t) *
        sum(u[s] * alpha ** (-s) for s in range(t))

    Closed form solution could be computed using inclusive_scan,
    except naive implementation suffers from underflow/overflow
    problem for long sequences.

    This implementation solves this problem by representing number
    using (fp_value, int_exponent) to extend representable range.
    """
    print("[Begin exponential moving average example]")
    u = 3.0 + cp.sin(cp.linspace(-3.0, 3.0, num=1024, dtype=cp.double))
    u += cp.random.normal(0.0, 0.1, size=u.size)
    alpha = 0.05

    assert 0.0 < alpha < 1.0

    @parallel.gpu_struct
    class ValueScale:
        value: cp.float64
        scale: cp.int64

    def add_op(v1: ValueScale, v2: ValueScale) -> ValueScale:
        if v1.scale > v2.scale:
            s = v2.scale
            v = v2.value + v1.value * (alpha ** (v1.scale - v2.scale))
        else:
            s = v1.scale
            v = v1.value + v2.value * (alpha ** (v2.scale - v1.scale))
        return ValueScale(v, s)

    def negative_op(i: cp.int64) -> cp.int64:
        return -i

    seq_it = parallel.CountingIterator(cp.int64(0))
    negative_exponents_it = parallel.TransformIterator(seq_it, negative_op)
    d_inp = parallel.ZipIterator(u, negative_exponents_it)

    d_cumsum = cp.empty(u.shape, dtype=ValueScale.dtype)
    h_init = ValueScale(0.0, 0)

    parallel.inclusive_scan(d_inp, d_cumsum, add_op, h_init, u.size)

    it_seq = parallel.CountingIterator(cp.int64(0))
    d_ema = cp.empty_like(u)

    def combine_op(v: ValueScale, t: cp.int64) -> cp.float64:
        return (1 - alpha) * v.value * alpha ** (t + v.scale)

    parallel.binary_transform(d_cumsum, it_seq, d_ema, combine_op, u.size)

    d_ema += (alpha ** cp.arange(1, u.size + 1)) * u[0]

    def ema_ref(u: np.ndarray, alpha: float):
        "Sequential reference implementation of EMA"
        a = np.empty_like(u)
        a[0] = u[0]
        for t in range(1, u.size):
            a[t] = (1 - alpha) * u[t] + alpha * a[t - 1]

        return a

    h_u = u.get()
    h_ema = ema_ref(h_u, alpha)

    assert np.allclose(h_ema, d_ema.get())

    print(f"Start of input signal: {h_u[:4]}")
    print(f"Smoothing parameter: {alpha}")
    print(f"EMA: {d_ema[:4]}")
    print(f"EMA_ref: {h_ema[:4]}")
    print("[End exponential moving average example]")


if __name__ == "__main__":
    print("Running scan_applications examples")
    inclusive_segmented_sum_example()
    logcdfs_from_logpdfs_example()
    exponential_moving_average_using_scan_example()
    print("All examples completed successfully!")
