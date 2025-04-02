import cProfile
import io
import pstats
import time

import cupy as cp
import line_profiler
import numpy as np
import palanteer as ps

import cuda.parallel.experimental.algorithms as impl_new
import cuda.parallel.experimental.algorithms.legacy as impl_base
import cuda.parallel.experimental.iterators as iter_new
import cuda.parallel.experimental.iterators_legacy as iter_base


def time_merge_sort_pointer(reps, mod):
    n = 10
    keys = cp.arange(n, dtype="i4")
    vals = cp.arange(n, dtype="i8")
    res_keys = cp.empty_like(keys)
    res_vals = cp.empty_like(vals)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = mod.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, n)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, keys, vals, res_keys, res_vals, n)
    t1 = time.perf_counter_ns()

    cp.cuda.runtime.deviceSynchronize()

    return t1 - t0


def time_merge_sort_iterator(reps, alg_mod, iter_mod):
    n = 10
    keys_dt = cp.int32
    vals_dt = cp.int64
    keys = iter_mod.CountingIterator(np.int32(0))
    vals = iter_mod.CountingIterator(np.int64(0))
    res_keys = cp.empty(n, dtype=keys_dt)
    res_vals = cp.empty(n, dtype=vals_dt)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = alg_mod.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, n)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    t0 = time.perf_counter_ns()
    for i in range(reps):
        alg(scratch, keys, vals, res_keys, res_vals, n)
    t1 = time.perf_counter_ns()

    cp.cuda.runtime.deviceSynchronize()

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
    cp.cuda.runtime.deviceSynchronize()


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
    time_merge_sort_pointer(10000, mod)


def run_timer_pointer():
    fn = time_merge_sort_pointer
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
    run_timer_iterator()


def run_timer_iterator():
    fn = time_merge_sort_iterator
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
    cp.cuda.runtime.deviceSynchronize()
    lineprofile_reduce(100000, impl_new)
    cp.cuda.runtime.deviceSynchronize()


def run_palanteer_pointer_impl(reps, mod, tag: str):
    n = 10
    keys = cp.arange(n, dtype="i4")
    vals = cp.arange(n, dtype="i8")
    res_keys = cp.empty_like(keys)
    res_vals = cp.empty_like(vals)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = mod.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, n)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    with ps.plScope(tag):
        for i in range(reps):
            alg(scratch, keys, vals, res_keys, res_vals, n)

    return


def run_palanteer_pointer():
    run_palanteer_pointer_impl(100000, impl_new, "merge_sort_pointer_new")
    run_palanteer_pointer_impl(100000, impl_base, "merge_sort_pointer_base")


def run_palanteer_iterator_impl(reps, alg_mod, iter_mod, tag: str):
    n = 10
    keys_dt = cp.int32
    vals_dt = cp.int64
    keys = iter_mod.CountingIterator(np.int32(0))
    vals = iter_mod.CountingIterator(np.int64(0))
    res_keys = cp.empty(n, dtype=keys_dt)
    res_vals = cp.empty(n, dtype=vals_dt)

    def my_cmp(a: np.int32, b: np.int32) -> np.int32:
        return np.int32(a < b)

    alg = alg_mod.merge_sort(keys, vals, res_keys, res_vals, my_cmp)
    temp_bytes = alg(None, keys, vals, res_keys, res_vals, n)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)

    with ps.plScope(tag):
        for i in range(reps):
            alg(scratch, keys, vals, res_keys, res_vals, n)

    return


def run_palanteer_iterator():
    run_palanteer_iterator_impl(100000, impl_new, iter_new, "merge_sort_iterator_new")
    run_palanteer_iterator_impl(
        100000, impl_base, iter_base, "merge_sort_iterator_base"
    )


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
    parser.add_argument(
        "--palanteer-pointer",
        action="store_true",
        dest="palanteer_pointer",
        help="Instrument pointer-base example using Palanteer",
    )
    parser.add_argument(
        "--palanteer-iterator",
        action="store_true",
        dest="palanteer_iterator",
        help="Instrument iterator-base example using Palanteer",
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
            "palanteer_iterator",
            "palanteer_pointer",
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
    elif args.palanteer_iterator:
        run_palanteer_iterator()
    elif args.palanteer_pointer:
        run_palanteer_pointer()
    else:
        raise ValueError("Argument not supported")
