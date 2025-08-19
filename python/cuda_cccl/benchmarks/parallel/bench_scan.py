import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def scan_pointer_exclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    alg = parallel.make_exclusive_scan(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_pointer_inclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    alg = parallel.make_inclusive_scan(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_pointer_well_known_exclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    # Use the well-known PLUS operation from OpKind
    alg = parallel.make_exclusive_scan(input_array, res, parallel.OpKind.PLUS, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_pointer_well_known_inclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    # Use the well-known PLUS operation from OpKind
    alg = parallel.make_inclusive_scan(input_array, res, parallel.OpKind.PLUS, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct_exclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    alg = parallel.make_exclusive_scan(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct_inclusive(input_array, build_only):
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    alg = parallel.make_inclusive_scan(input_array, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, input_array, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, input_array, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator_exclusive(inp, size, build_only):
    dt = cp.int32
    res = cp.empty(size, dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = parallel.make_exclusive_scan(inp, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, inp, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, inp, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator_inclusive(inp, size, build_only):
    dt = cp.int32
    res = cp.empty(size, dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = parallel.make_inclusive_scan(inp, res, my_add, h_init)
    if not build_only:
        temp_storage_bytes = alg(None, inp, res, size, h_init)
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
        alg(temp_storage, inp, res, size, h_init)

    cp.cuda.runtime.deviceSynchronize()


@parallel.gpu_struct
class MyStruct:
    x: np.int32
    y: np.int32


def bench_compile_scan_pointer_exclusive(compile_benchmark):
    input_array = cp.random.randint(0, 10, 10)

    def run():
        scan_pointer_exclusive(input_array, build_only=True)

    compile_benchmark(parallel.make_exclusive_scan, run)


def bench_compile_scan_pointer_inclusive(compile_benchmark):
    input_array = cp.random.randint(0, 10, 10)

    def run():
        scan_pointer_inclusive(input_array, build_only=True)

    compile_benchmark(parallel.make_inclusive_scan, run)


def bench_compile_scan_iterator_exclusive(compile_benchmark):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        scan_iterator_exclusive(inp, 10, build_only=True)

    compile_benchmark(parallel.make_exclusive_scan, run)


def bench_compile_scan_iterator_inclusive(compile_benchmark):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        scan_iterator_inclusive(inp, 10, build_only=True)

    compile_benchmark(parallel.make_inclusive_scan, run)


def bench_scan_pointer_exclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        scan_pointer_exclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_pointer_inclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        scan_pointer_inclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_pointer_well_known_exclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        scan_pointer_well_known_exclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_pointer_well_known_inclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    def run():
        scan_pointer_well_known_inclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_iterator_exclusive(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        scan_iterator_exclusive(inp, size, build_only=False)

    benchmark(run)


def bench_scan_iterator_inclusive(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    def run():
        scan_iterator_inclusive(inp, size, build_only=False)

    benchmark(run)


def bench_scan_struct_exclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    def run():
        scan_struct_exclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_struct_inclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    def run():
        scan_struct_inclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_pointer_single_phase_exclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    # warm up run
    scan_pointer_single_phase_exclusive(input_array, build_only=False)

    # benchmark run
    def run():
        scan_pointer_single_phase_exclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_pointer_single_phase_inclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, size)

    # warm up run
    scan_pointer_single_phase_inclusive(input_array, build_only=False)

    # benchmark run
    def run():
        scan_pointer_single_phase_inclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_iterator_single_phase_exclusive(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    # warm up run
    scan_iterator_single_phase_exclusive(inp, size, build_only=False)

    # benchmark run
    def run():
        scan_iterator_single_phase_exclusive(inp, size, build_only=False)

    benchmark(run)


def bench_scan_iterator_single_phase_inclusive(benchmark, size):
    inp = parallel.CountingIterator(np.int32(0))

    # warm up run
    scan_iterator_single_phase_inclusive(inp, size, build_only=False)

    # benchmark run
    def run():
        scan_iterator_single_phase_inclusive(inp, size, build_only=False)

    benchmark(run)


def bench_scan_struct_single_phase_exclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    # warm up run
    scan_struct_single_phase_exclusive(input_array, build_only=False)

    # benchmark run
    def run():
        scan_struct_single_phase_exclusive(input_array, build_only=False)

    benchmark(run)


def bench_scan_struct_single_phase_inclusive(benchmark, size):
    input_array = cp.random.randint(0, 10, (size, 2), dtype="int32").view(MyStruct)

    # warm up run
    scan_struct_single_phase_inclusive(input_array, build_only=False)

    # benchmark run
    def run():
        scan_struct_single_phase_inclusive(input_array, build_only=False)

    benchmark(run)


def scan_pointer_single_phase_exclusive(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for exclusive scan."""
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    parallel.exclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_pointer_single_phase_inclusive(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for inclusive scan."""
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    def my_add(a, b):
        return a + b

    parallel.inclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct_single_phase_exclusive(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for structs exclusive scan."""
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    parallel.exclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_struct_single_phase_inclusive(input_array, build_only):
    """Single-phase API that automatically manages temporary storage for structs inclusive scan."""
    size = len(input_array)
    res = cp.empty_like(input_array)
    h_init = MyStruct(0, 0)

    def my_add(a, b):
        return MyStruct(a.x + b.x, a.y + b.y)

    parallel.inclusive_scan(input_array, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator_single_phase_exclusive(inp, size, build_only):
    """Single-phase API that automatically manages temporary storage for iterators exclusive scan."""
    dt = cp.int32
    res = cp.empty(size, dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    parallel.exclusive_scan(inp, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()


def scan_iterator_single_phase_inclusive(inp, size, build_only):
    """Single-phase API that automatically manages temporary storage for iterators inclusive scan."""
    dt = cp.int32
    res = cp.empty(size, dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    parallel.inclusive_scan(inp, res, my_add, h_init, size)

    cp.cuda.runtime.deviceSynchronize()
