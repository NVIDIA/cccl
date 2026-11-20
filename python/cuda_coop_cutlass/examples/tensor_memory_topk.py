# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Interleave Tensor Core matrix products and cooperative TopK selection.

Each block owns a matrix batch. Every window reads a real tcgen05 accumulator
from TMEM, selects keys with coop, writes the result back to TMEM, and uses it
as the next MMA's accumulator. The undefined TopK tail is explicitly zeroed.
Later products are uniform so an independent multiset reference can account
for TopK's unspecified order without imposing a register-to-matrix mapping.
"""

import cutlass
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import numpy as np
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
from cuda.bindings import driver

_M, _N, _K = 128, 8, 16
_ITEMS = _M * _N


def _check(result):
    """Raise on Driver errors and unwrap a single returned handle or value."""

    if int(result[0]):
        raise RuntimeError(f"CUDA Driver call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


# docs: start cutlass-tensor-memory-topk
@cute.struct
class _SharedStorage:
    barrier: cutlass.Int64
    tmem_address: cutlass.Int32


@cute.kernel
def _windowed_topk_kernel(
    output: cute.Pointer,
    windows: cutlass.Int32,
    keep: cutlass.Int32,
    mma: cute.TiledMma,
    a_layout: cute.ComposedLayout,
    b_layout: cute.ComposedLayout,
):
    thread, _, _ = cute.arch.thread_idx()
    batch, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    # docs: start cutlass-tmem-storage
    # Native operand buffers stay live alongside coop's separate scratch.
    # CuTe places allocations from both libraries in one shared-memory layout.
    smem = cutlass.memory.SmemAllocator()
    storage = smem.allocate(_SharedStorage)
    a = smem.allocate_tensor(
        cutlass.Float16, a_layout.outer, 128, a_layout.inner
    )
    b = smem.allocate_tensor(
        cutlass.Float16, b_layout.outer, 128, b_layout.inner
    )
    a_coords = mma.get_slice(0).partition_A(cute.make_identity_tensor((_M, _K)))
    b_coords = mma.get_slice(0).partition_B(cute.make_identity_tensor((_N, _K)))
    a_fragment = mma.make_fragment_A(a)
    b_fragment = mma.make_fragment_B(b)
    accumulator = mma.make_fragment_C(mma.partition_shape_C((_M, _N)))
    tmem = cutlass.memory.TmemAllocator(
        storage.tmem_address.ptr,
        barrier_for_retrieve=pipeline.NamedBarrier(1, _M),
    )
    # TMEM allocations have a 32-column minimum; this tile uses eight.
    tmem.allocate(32)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(storage.barrier.ptr, 1)
    cute.arch.mbarrier_init_fence()
    tmem.wait_for_alloc()
    pointer = tmem.retrieve_ptr(cutlass.Float32)
    accumulator = cute.make_tensor(pointer, accumulator.layout)
    tmem.relinquish_alloc_permit()
    # docs: end cutlass-tmem-storage

    # docs: start cutlass-tmem-copy-layout
    # Use the same matrix slice for the copy and both thread partitions.
    # The destination partition determines the eight registers per thread.
    load_copy = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x8), cutlass.Float32
        ),
        accumulator[(None, None), 0, 0],
    )
    store_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x8), cutlass.Float32
    )
    source = load_copy.get_slice(thread).partition_S(
        accumulator[(None, None), 0, 0]
    )
    coordinates = mma.get_slice(0).partition_C(
        cute.make_identity_tensor((_M, _N))
    )
    register_shape = (
        load_copy.get_slice(thread)
        .partition_D(coordinates[(None, None), 0, 0])
        .shape
    )
    assert cute.size(register_shape) == _N
    registers = cute.make_rmem_tensor(register_shape, cutlass.Float32)
    # docs: end cutlass-tmem-copy-layout
    # Record the MMA output, the selection before storing, and the TMEM reload.
    result = cute.make_tensor(
        output, cute.make_layout((3, _ITEMS, windows, cute.arch.grid_dim()[0]))
    )
    group = coop.this_block()
    scratch = coop.TempStorage()
    phase = cutlass.Int32(0)

    for window in range(windows):
        # docs: start cutlass-tmem-mma
        # The first product varies by matrix coordinate. Later products are
        # uniform, so their host reference is independent of TopK ordering.
        # SIMT writes must become visible to the asynchronous MMA proxy.
        for index in cutlass.range_constexpr(_M * _K // _M):
            offset = thread + index * _M
            row, k = a_coords[offset]
            value = cutlass.Float16(batch + window + 1)
            if window == 0:
                value = cutlass.Float16((row % 17 - 8) + (k % 3))
            a[offset] = value
        for index in cutlass.range_constexpr(_N * _K // _M):
            offset = thread + index * _M
            column, k = b_coords[offset]
            value = cutlass.Float16(1)
            if window == 0:
                value = cutlass.Float16(column % 7 - 3 + batch)
            b[offset] = value
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        # One warp issues MMA; all threads wait for its completion barrier.
        # After the first window, C is the selection stored by the last window.
        if warp == 0:
            mma.set(tcgen05.Field.ACCUMULATE, window > 0)
            cute.gemm(
                mma,
                accumulator,
                a_fragment[None, None, None, 0],
                b_fragment[None, None, None, 0],
                accumulator,
            )
            with cute.arch.elect_one():
                tcgen05.commit(storage.barrier.ptr)
        cute.arch.mbarrier_wait(storage.barrier.ptr, phase)
        phase ^= 1
        # docs: end cutlass-tmem-mma

        # docs: start cutlass-tmem-simt
        # Native CuTe moves TMEM values into registers. ThreadData then exposes
        # those per-thread values to coop without claiming ownership of TMEM.
        cute.copy(load_copy, source, registers)
        values = coop.ThreadData.from_register_tensor(registers)
        for item in cutlass.range_constexpr(_N):
            result[0, thread * _N + item, window, batch] = values[item]
        selected = coop.topk_max_keys(
            group, values, k=keep, temp_storage=scratch
        )
        # Only the selected prefix is defined. Zero the remaining positions
        # before the next MMA consumes the entire accumulator.
        for item in cutlass.range_constexpr(_N):
            value = cutlass.Float32(0)
            if thread * _N + item < keep:
                value = selected[item]
            values[item] = value
            result[1, thread * _N + item, window, batch] = value
        # docs: end cutlass-tmem-simt

        # docs: start cutlass-tmem-store
        # Reconstruct the native register layout and apply the inverse copy.
        registers.store(values.to_tensor_ssa(like=registers))
        cute.copy(store_atom, registers, source)
        # Complete all stores before another warp or the next MMA uses TMEM.
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

        # Read back the stored selection, including the explicitly zero tail.
        cute.copy(load_copy, source, registers)
        roundtrip = coop.ThreadData.from_register_tensor(registers)
        for item in cutlass.range_constexpr(_N):
            result[2, thread * _N + item, window, batch] = roundtrip[item]
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_threads()
        # docs: end cutlass-tmem-store

    tmem.free(pointer)


@cute.jit
def _launch(
    output: cute.Pointer,
    windows: cutlass.Int32,
    keep: cutlass.Int32,
    batches: cutlass.Constexpr,
):
    mma = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.Float16,
            cutlass.Float32,
            (_M, _N, _K),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
    )
    a_layout = sm100_utils.make_smem_layout_a(
        mma, (_M, _N, _K), cutlass.Float16, 1
    )
    b_layout = sm100_utils.make_smem_layout_b(
        mma, (_M, _N, _K), cutlass.Float16, 1
    )
    _windowed_topk_kernel(
        output, windows, keep, mma, a_layout, b_layout
    ).launch(grid=batches, block=_M)


# docs: end cutlass-tensor-memory-topk


def is_supported_device():
    """Check whether the active device supports this SM100/SM103 example."""

    cutlass.cuda.initialize_cuda_context()
    device = _check(driver.cuCtxGetDevice())
    attribute = driver.CUdevice_attribute
    major = _check(
        driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
        )
    )
    minor = _check(
        driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
        )
    )
    return (major, minor) in {(10, 0), (10, 3)}


# docs: start cutlass-tmem-validation
def validate_result(output, keep):
    """Check exact roundtrips and each window's independent matrix reference.

    TopK is unsorted, so compare multisets for the matrix product and selected
    keys. Exact before/after comparisons separately check the native TMEM copy
    mapping. Small integer inputs make these float32 comparisons exact.
    """

    batches, windows = output.shape[:2]
    np.testing.assert_array_equal(output[..., 1], output[..., 2])
    np.testing.assert_array_equal(output[:, :, keep:, 1], 0)

    rows = np.arange(_M)[:, None]
    columns = np.arange(_N)[:, None]
    k = np.arange(_K)[None, :]
    a = (rows % 17 - 8 + k % 3).astype(np.float32)
    for batch in range(batches):
        b = np.broadcast_to(columns % 7 - 3 + batch, (_N, _K))
        expected = np.sort((a @ b.T).ravel())
        for window in range(windows):
            if window:
                expected = np.sort(expected + _K * (batch + window + 1))
            np.testing.assert_array_equal(
                np.sort(output[batch, window, :, 0]), expected
            )
            expected = np.sort(
                np.concatenate(
                    (
                        expected[-keep:],
                        np.zeros(_ITEMS - keep, dtype=np.float32),
                    )
                )
            )
            np.testing.assert_array_equal(
                np.sort(output[batch, window, :, 1]), expected
            )


# docs: end cutlass-tmem-validation


def run_example(batches=2, windows=4, keep=129):
    """Run independent batches of dependent MMA/TopK windows and check them.

    The returned array has shape (batches, windows, 1024, 3). Its channels hold
    each MMA output, the selection with a zeroed tail, and the TMEM reload.
    """

    if batches < 1 or windows < 1 or not 1 <= keep <= _ITEMS:
        raise ValueError(
            "Require positive batches/windows and 1 <= keep <= 1024"
        )
    if not is_supported_device():
        raise RuntimeError("This tcgen05 TMEM example requires SM100 or SM103")

    output = np.full((batches, windows, _ITEMS, 3), np.nan, dtype=np.float32)
    allocation = _check(driver.cuMemAlloc(output.nbytes))
    try:
        _check(
            driver.cuMemcpyHtoD(allocation, output.ctypes.data, output.nbytes)
        )
        pointer = make_ptr(
            cutlass.Float32,
            int(allocation),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        _launch(pointer, cutlass.Int32(windows), cutlass.Int32(keep), batches)
        _check(driver.cuCtxSynchronize())
        _check(
            driver.cuMemcpyDtoH(output.ctypes.data, allocation, output.nbytes)
        )
    finally:
        _check(driver.cuMemFree(allocation))
    validate_result(output, keep)
    return output


if __name__ == "__main__":
    output = run_example()
    batches, windows = output.shape[:2]
    print(f"Validated {batches} batches with {windows} MMA/TopK windows each.")
