# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise caller arrays reserved alongside cooperative scratch."""

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
@pytest.mark.parametrize(
    "dtype, c_type, value, alignment",
    [
        pytest.param(types.int32, "int", 17, 32, id="int32"),
        pytest.param(types.float64, "double", 1.0 + 2**-30, 64, id="float64"),
    ],
)
def test_reservations_survive_scratch_reuse_and_accept_foreign_consumer(
    items_per_thread, sharing, dtype, c_type, value, alignment
):
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()
    consume = cuda.declare_device(
        "consume_reserved_array",
        types.int32(types.CPointer(dtype)),
        link=cuda.CUSource(
            f"""
            extern "C" __device__ int consume_reserved_array({c_type}* data) {{
                data[0] += 5;
                return ((unsigned long long)data) % {alignment};
            }}
            """
        ),
        abi="c",
    )
    count = 32 * items_per_thread

    @cuda.jit
    def kernel(source, scanned, preserved, prefix, facts, items_per_thread):
        storage = coop.TempStorage(sharing=sharing)
        persistent = coop.TempStorage(sharing="exclusive")
        small = persistent.reserve(3, types.uint8)
        values = persistent.reserve(count, dtype, alignment=alignment)
        thread = cuda.threadIdx.x
        if thread < 3:
            small[thread] = 201 + thread
        for i in range(thread, count, cuda.blockDim.x):
            values[i] = value + i
        cuda.syncthreads()

        block = coop.this_block()
        items = coop.ThreadData(items_per_thread)
        coop.load(
            block,
            source,
            items,
            algorithm="transpose",
            temp_storage=storage,
        )
        cuda.syncthreads()
        result = coop.exclusive_sum(block, items, temp_storage=storage)
        cuda.syncthreads()
        coop.store(
            block,
            scanned,
            result,
            algorithm="transpose",
            temp_storage=storage,
        )
        cuda.syncthreads()

        if thread == 0:
            facts[0] = consume(ffi.from_buffer(values))
            facts[1] = values.shape[0]
        cuda.syncthreads()
        for i in range(thread, count, cuda.blockDim.x):
            preserved[i] = values[(i + 1) % count]
        if thread < 3:
            prefix[thread] = small[thread]

    source = np.arange(count, dtype=np.int32)
    scanned = np.full_like(source, -1)
    preserved = np.empty(count, dtype=str(dtype))
    prefix = np.empty(3, dtype=np.uint8)
    facts = np.full(2, -1, dtype=np.int64)
    kernel[1, 32](source, scanned, preserved, prefix, facts, items_per_thread)
    expected = value + np.arange(count, dtype=preserved.dtype)
    expected[0] += 5
    np.testing.assert_array_equal(preserved, np.roll(expected, -1))
    np.testing.assert_array_equal(scanned, np.cumsum(source) - source)
    np.testing.assert_array_equal(prefix, [201, 202, 203])
    np.testing.assert_array_equal(facts, [0, count])


def test_reserve_only_alias_passed_to_inlined_helper():
    @cuda.jit(device=True, inline="always")
    def allocate(storage):
        alias = storage
        return alias.reserve(33, types.float64, alignment=1)

    @cuda.jit
    def kernel(output, facts):
        storage = numba_coop.TempStorage(sharing="exclusive")
        prefix = storage.reserve(3, types.uint8)
        alias = storage
        values = allocate(alias)
        thread = cuda.threadIdx.x
        if thread < 3:
            prefix[thread] = 200 + thread
        for i in range(thread, 33, cuda.blockDim.x):
            values[i] = i + 0.5
        cuda.syncthreads()
        for i in range(thread, 33, cuda.blockDim.x):
            output[i] = values[32 - i]
        if thread == 0:
            facts[0] = values.shape[0]
            facts[1] = prefix[2]

    output = np.empty(33, dtype=np.float64)
    facts = np.empty(2, dtype=np.int64)
    kernel[1, 32](output, facts)
    np.testing.assert_array_equal(output, np.arange(33)[::-1] + 0.5)
    np.testing.assert_array_equal(facts, [33, 202])


def test_other_descriptor_can_automatically_synchronize_primitive_scratch():
    @cuda.jit
    def kernel(source, output, sums):
        persistent = coop.TempStorage()
        values = persistent.reserve(32, types.int32)
        scratch = coop.TempStorage(auto_sync=True)
        thread = cuda.threadIdx.x
        values[thread] = source[thread] + 1000
        cuda.syncthreads()
        first = coop.sum(
            coop.this_block(), source[thread], temp_storage=scratch
        )
        second = coop.sum(
            coop.this_block(), source[thread] * 2, temp_storage=scratch
        )
        cuda.syncthreads()
        output[thread] = values[31 - thread]
        if thread == 0:
            sums[0] = first
            sums[1] = second

    source = np.arange(32, dtype=np.int32)
    output = np.empty_like(source)
    sums = np.empty(2, dtype=np.int32)
    kernel[1, 32](source, output, sums)
    np.testing.assert_array_equal(output, source[::-1] + 1000)
    np.testing.assert_array_equal(sums, [source.sum(), 2 * source.sum()])


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_shared_reservations_alias_with_manual_phases(items_per_thread):
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()
    address = cuda.declare_device(
        "reserved_array_address",
        types.uint64(types.CPointer(types.int32)),
        link=cuda.CUSource(
            """
            extern "C" __device__ unsigned long long
            reserved_array_address(int* data) {
                return (unsigned long long)data;
            }
            """
        ),
        abi="c",
    )
    count = 32 * items_per_thread
    capacity = count * 4

    @cuda.jit
    def kernel(source, output, facts):
        # Capacity covers the largest reservation, not their sum. Primitive
        # scratch must share this region too for compilation to succeed.
        storage = coop.TempStorage(capacity)
        wide = storage.reserve(count, types.int32, alignment=64)
        separate = coop.TempStorage()
        live = separate.reserve(32, types.int32)
        thread = cuda.threadIdx.x
        live[thread] = 1000 + thread

        for phase in range(2):
            # Executing this reservation again returns the same region.
            reserve = storage.reserve
            narrow = reserve(32, types.int32)
            for i in range(thread, count, cuda.blockDim.x):
                wide[i] = source[i] + 100 * phase
            cuda.syncthreads()
            output[phase * 32 + thread] = narrow[31 - thread]
            value = narrow[thread]
            # Every thread must load its value before primitive scratch can
            # overwrite the same shared array.
            cuda.syncthreads()

            total = coop.sum(coop.this_block(), value, temp_storage=storage)
            if thread == 0:
                output[64 + phase] = total
                facts[phase] = address(ffi.from_buffer(narrow))
            # Complete this phase before reusing the aliased region.
            cuda.syncthreads()

        output[66 + thread] = live[31 - thread]
        if thread == 0:
            facts[2] = address(ffi.from_buffer(wide))
            facts[3] = address(ffi.from_buffer(live))
            facts[4] = wide.size
            facts[5] = narrow.size

    source = np.arange(count, dtype=np.int32)
    output = np.empty(98, dtype=np.int32)
    facts = np.empty(6, dtype=np.uint64)
    kernel[1, 32](source, output, facts)
    np.testing.assert_array_equal(output[:32], source[:32][::-1])
    np.testing.assert_array_equal(output[32:64], source[:32][::-1] + 100)
    np.testing.assert_array_equal(
        output[64:66], [source[:32].sum(), source[:32].sum() + 3200]
    )
    np.testing.assert_array_equal(output[66:], np.arange(1000, 1032)[::-1])
    assert facts[0] == facts[1] == facts[2]
    assert facts[2] != facts[3]
    assert facts[2] % 64 == 0
    np.testing.assert_array_equal(facts[4:], [count, 32])


def test_large_reservations_survive_cooperative_scratch_reuse():
    count = 13 * 1024

    @cuda.jit
    def kernel(output):
        storage = coop.TempStorage(alignment=16, sharing="exclusive")
        own = storage.reserve(32, types.int32)
        reserved = storage.reserve(count, types.int32)
        thread = cuda.threadIdx.x
        own[thread] = 1000 + thread
        for i in range(thread, count, cuda.blockDim.x):
            reserved[i] = i
        cuda.syncthreads()
        total = coop.sum(coop.this_block(), own[thread], temp_storage=storage)
        cuda.syncthreads()
        released = storage.release(types.int32)
        # Primitive scratch precedes the two exclusive reservation slices.
        own_start = released.size - count - 32
        output[thread] = (
            released[own_start + 31 - thread]
            + released[own_start + 32 + count - 1 - thread]
        )
        if thread == 0:
            output[32] = total
            output[33] = own.size
            output[34] = reserved.size
            output[35] = released.size

    output = np.empty(36, dtype=np.int32)
    kernel[1, 32](output)
    expected = 1000 + np.arange(32)[::-1] + count - 1 - np.arange(32)
    np.testing.assert_array_equal(output[:32], expected)
    assert output[32] == np.arange(1000, 1032).sum()
    np.testing.assert_array_equal(output[33:35], [32, count])
    compiled = next(iter(kernel._launch_config_overloads.values()))
    required_bytes = compiled.metadata["required_dynamic_shared_memory"]
    assert required_bytes >= count * 4
    released_bytes = int(output[35]) * np.dtype(np.int32).itemsize
    assert released_bytes > (count + 32) * np.dtype(np.int32).itemsize
    # The backing allocation also rounds the region up to its alignment.
    assert (released_bytes + 15) // 16 * 16 == required_bytes


@pytest.mark.parametrize(
    "dtype, c_type, released_count",
    [
        pytest.param(None, "unsigned char", 424, id="default-bytes"),
        pytest.param(types.float64, "double", 53, id="float64"),
    ],
)
def test_release_hands_off_complete_exclusive_reservation_region(
    dtype, c_type, released_count
):
    cffi = pytest.importorskip("cffi")
    ffi = cffi.FFI()
    record_addresses = cuda.declare_device(
        "record_reservation_addresses",
        types.void(
            types.CPointer(types.int32),
            types.CPointer(types.float64),
            types.CPointer(types.int32),
            types.CPointer(types.uint64),
        ),
        link=cuda.CUSource(
            """
            extern "C" __device__ void record_reservation_addresses(
                int* prefix, double* values, int* independent,
                unsigned long long* facts) {
                facts[0] = (unsigned long long)prefix;
                facts[1] = (unsigned long long)values;
                facts[2] = (unsigned long long)independent;
            }
            """
        ),
        abi="c",
    )
    inspect_released = cuda.declare_device(
        "inspect_released_reservations",
        types.uint64(
            types.CPointer(types.uint8 if dtype is None else dtype),
            types.CPointer(types.uint8),
        ),
        link=cuda.CUSource(
            f"""
            extern "C" __device__ unsigned long long
            inspect_released_reservations({c_type}* data,
                                          unsigned char* output) {{
                unsigned char* bytes = (unsigned char*)data;
                for (int i = threadIdx.x; i < 424; i += blockDim.x) {{
                    // Initialize padding before reading the whole region.
                    if (i >= 132 && i < 160) bytes[i] = 0xa5;
                    output[i] = bytes[i];
                }}
                return (unsigned long long)data;
            }}
            """
        ),
        abi="c",
    )

    @cuda.jit
    def kernel(output, independent_output, facts):
        storage = coop.TempStorage(sharing="exclusive")
        prefix = storage.reserve(33, types.int32)
        values = storage.reserve(33, types.float64, alignment=32)
        independent = coop.TempStorage()
        live = independent.reserve(32, types.int32)
        thread = cuda.threadIdx.x
        for i in range(thread, 33, cuda.blockDim.x):
            prefix[i] = 1000 + i
            values[i] = 100.25 + i
        live[thread] = 500 + thread
        cuda.syncthreads()
        if thread == 0:
            record_addresses(
                ffi.from_buffer(prefix),
                ffi.from_buffer(values),
                ffi.from_buffer(live),
                ffi.from_buffer(facts),
            )

        if dtype is None:
            released = storage.release()
        else:
            released = storage.release(dtype)
        address = inspect_released(
            ffi.from_buffer(released), ffi.from_buffer(output)
        )
        if thread == 0:
            facts[3] = address
            facts[4] = released.size

        # Releasing one descriptor does not consume another descriptor.
        still_live = independent.reserve(32, types.int32)
        independent_output[thread] = still_live[31 - thread]

    output = np.empty(424, dtype=np.uint8)
    independent_output = np.empty(32, dtype=np.int32)
    facts = np.zeros(5, dtype=np.uint64)
    kernel[1, 32](output, independent_output, facts)

    expected = np.empty(424, dtype=np.uint8)
    expected[:132] = (1000 + np.arange(33, dtype=np.int32)).view(np.uint8)
    expected[132:160] = 0xA5
    expected[160:] = (100.25 + np.arange(33, dtype=np.float64)).view(np.uint8)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(independent_output, np.arange(500, 532)[::-1])
    assert facts[1] - facts[0] == 160
    assert facts[3] == facts[0]
    assert facts[3] % 32 == 0
    assert not facts[0] <= facts[2] < facts[0] + 424
    assert facts[4] == released_count
