# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel

# Given input vector u and smoothing parameter 0<alpha<1
# the exponential moving average sequence ma is defined
# by ma[t] = (1 - alpha) * u[t] + alpha * ma[t-1] with
# initial condition ma[0] = u[0].
#
# This recurrence equation admits a closed form solution:
#
# ma[t] = alpha ** (t + 1) * u[0] + (1 - alpha) * (alpha ** t) *
#     sum(u[s] * alpha ** (-s) for s in range(t))
#
# Closed form solution could be computed using inclusive_scan,
# except naive implementation suffers from underflow/overflow
# problem for long sequences.
#
# This implementation solves this problem by representing number
# using (fp_value, int_exponent) to extend representable range.
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
print("Exponential moving average example completed successfully")
