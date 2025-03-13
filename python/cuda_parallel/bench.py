import time

import cupy as cp
import numpy as np

import cuda.parallel.experimental.algorithms._cy_reduce as impl_new
import cuda.parallel.experimental.algorithms.reduce as impl_base


def time1(reps):
    global pr
    d = cp.ones(10, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = impl_new.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, d.size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, d, res, d.size, h_init)
    t1 = time.perf_counter_ns()

    return t1 - t0


def time2(reps):
    global pr
    d = cp.ones(10, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = impl_base.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, d.size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, d, res, d.size, h_init)
    t1 = time.perf_counter_ns()

    return t1 - t0


if __name__ == "__main__":
    t_base = time2(100000) / 100000
    t_new = time1(100000) / 100000
    print(f"Base: {t_base:<5.2f} ns per submission")
    print(f"New:  {t_new:<5.2f} ns per submission")
    print(f"'Base - New' diff: {t_base - t_new:<5.2f} ns")
