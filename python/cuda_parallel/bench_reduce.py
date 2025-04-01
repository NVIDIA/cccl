import cProfile
import io
import pstats
import time

import cupy as cp
import line_profiler
import numpy as np

import cuda.parallel.experimental.algorithms._cy_reduce as impl_new
import cuda.parallel.experimental.algorithms.reduce as impl_base
import cuda.parallel.experimental.cy_iterators as iter_new
import cuda.parallel.experimental.iterators as iter_base


def time_reduce_pointer(reps, mod):
    n = 10
    d = cp.ones(n, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = mod.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, n, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, d, res, n, h_init)
    t1 = time.perf_counter_ns()

    return t1 - t0


def time_reduce_iterator(reps, alg_mod, iter_mod):
    n = 10
    dt = cp.int32
    d = iter_mod.CountingIterator(np.int32(0))
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = alg_mod.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, n, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, d, res, n, h_init)
    t1 = time.perf_counter_ns()

    return t1 - t0


def cprofile_reduce(reps, mod, pr):
    d = cp.ones(10, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = mod.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, d.size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    pr.enable()
    for i in range(reps):
        alg(scratch, d, res, d.size, h_init)
    pr.disable()


@line_profiler.profile
def lineprofile_reduce(reps, mod):
    d = cp.ones(10, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = mod.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, d.size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    for i in range(reps):
        alg(scratch, d, res, d.size, h_init)


def run_specific(mod):
    time_reduce_pointer(10000, mod)


def run_timer_pointer():
    fn = time_reduce_pointer
    # warm-up
    fn(100, impl_new)
    fn(100, impl_base)
    # time
    t_new = fn(100000, impl_new) / 100000
    t_base = fn(100000, impl_base) / 100000
    print(f"Base: {t_base:<5.2f} ns per submission")
    print(f"New:  {t_new:<5.2f} ns per submission")
    print(f"'Base - New' diff: {t_base - t_new:<5.2f} ns")


def run_timer():
    run_timer_pointer()


def run_timer_iterator():
    fn = time_reduce_iterator
    # warm-up
    fn(100, impl_new, iter_new)
    fn(100, impl_base, iter_base)
    # time
    t_new = fn(100000, impl_new, iter_new) / 100000
    t_base = fn(100000, impl_base, iter_base) / 100000
    print(f"Base: {t_base:<5.2f} ns per submission")
    print(f"New:  {t_new:<5.2f} ns per submission")
    print(f"'Base - New' diff: {t_base - t_new:<5.2f} ns")


def run_cprofile():
    sortby = pstats.SortKey.CUMULATIVE
    pr_base = cProfile.Profile()
    cprofile_reduce(10000, impl_base, pr_base)

    s_base = io.StringIO()
    ps_base = pstats.Stats(pr_base, stream=s_base).sort_stats(sortby)
    ps_base.print_stats()
    print(s_base.getvalue())

    pr_new = cProfile.Profile()
    cprofile_reduce(10000, impl_new, pr_new)

    s_new = io.StringIO()
    ps_new = pstats.Stats(pr_new, stream=s_new).sort_stats(sortby)
    ps_new.print_stats()
    print(s_new.getvalue())


def run_line_profiler():
    lineprofile_reduce(100000, impl_base)
    lineprofile_reduce(100000, impl_new)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "bench_bindings",
        description="Script to benchmark performance of cuda.parallel bindings",
    )
    parser.add_argument(
        "--time", action="store_true", help="Time calls using both bindings"
    )
    parser.add_argument(
        "--time-pointer",
        action="store_true",
        dest="timer_pointer",
        help="Time calls using both bindings for pointer-based example",
    )
    parser.add_argument(
        "--time-iterator",
        action="store_true",
        dest="timer_iterator",
        help="Time calls using both bindings for iterator-based example",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Collect cProfile traces for both bindings",
    )
    parser.add_argument(
        "--line-profile",
        action="store_true",
        dest="line_profiler",
        help="Run line_profiler collection for both bindings",
    )
    parser.add_argument(
        "--run-new",
        action="store_true",
        dest="run_new",
        help="Run algorithm using new bindings",
    )
    parser.add_argument(
        "--run-base",
        action="store_true",
        dest="run_base",
        help="Run algorithm using ctypes bindings",
    )

    args = parser.parse_args()

    n_args_given = sum(
        getattr(args, attr)
        for attr in [
            "time",
            "cprofile",
            "line_profiler",
            "run_new",
            "run_base",
            "timer_pointer",
            "timer_iterator",
        ]
    )

    if 1 != n_args_given:
        raise ValueError("One of the options is required")

    if args.time:
        run_timer()
    elif args.timer_pointer:
        run_timer_pointer()
    elif args.timer_iterator:
        run_timer_iterator()
    elif args.cprofile:
        run_cprofile()
    elif args.line_profiler:
        run_line_profiler()
    elif args.run_new:
        run_specific(impl_new)
    elif args.run_base:
        run_specific(impl_base)
    else:
        raise ValueError("Argument not supported")
