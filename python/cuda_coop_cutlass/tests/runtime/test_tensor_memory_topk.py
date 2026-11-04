# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise repeated Tensor Core -> TopK -> Tensor Core dependencies.

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
import pytest
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05

import cuda.coop.cutlass as coop
from cuda.bindings import driver
from tests.runtime_support import check_cuda, device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_M, _N, _K = 128, 8, 16
_ITEMS = _M * _N


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
    tmem.allocate(32)
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(storage.barrier.ptr, 1)
    cute.arch.mbarrier_init_fence()
    tmem.wait_for_alloc()
    pointer = tmem.retrieve_ptr(cutlass.Float32)
    accumulator = cute.make_tensor(pointer, accumulator.layout)
    tmem.relinquish_alloc_permit()

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
    result = cute.make_tensor(
        output, cute.make_layout((3, _ITEMS, windows, cute.arch.grid_dim()[0]))
    )
    group = coop.this_block()
    scratch = coop.TempStorage()
    phase = cutlass.Int32(0)

    for window in range(windows):
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

        cute.copy(load_copy, source, registers)
        values = coop.ThreadData.from_register_tensor(registers)
        for item in cutlass.range_constexpr(_N):
            result[0, thread * _N + item, window, batch] = values[item]
        selected = coop.topk_max_keys(
            group, values, k=keep, temp_storage=scratch
        )
        for item in cutlass.range_constexpr(_N):
            value = cutlass.Float32(0)
            if thread * _N + item < keep:
                value = selected[item]
            values[item] = value
            result[1, thread * _N + item, window, batch] = value
        registers.store(values.to_tensor_ssa(like=registers))
        cute.copy(store_atom, registers, source)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_threads()

        # Read back the stored selection, including the explicitly zero tail.
        cute.copy(load_copy, source, registers)
        roundtrip = coop.ThreadData.from_register_tensor(registers)
        for item in cutlass.range_constexpr(_N):
            result[2, thread * _N + item, window, batch] = roundtrip[item]
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_threads()

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


@pytest.mark.parametrize("batches,windows,keep", [(1, 1, 1), (2, 4, 129)])
def test_tensor_core_windows_with_topk(batches, windows, keep):
    cutlass.cuda.initialize_cuda_context()
    device = check_cuda(driver.cuCtxGetDevice())
    attribute = driver.CUdevice_attribute
    major = check_cuda(
        driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
        )
    )
    minor = check_cuda(
        driver.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
        )
    )
    if (major, minor) not in {(10, 0), (10, 3)}:
        pytest.skip("tcgen05 TMEM test requires SM100 or SM103")

    output = np.full((batches, windows, _ITEMS, 3), np.nan, dtype=np.float32)
    with device_array(output) as pointer:
        _launch(pointer, cutlass.Int32(windows), cutlass.Int32(keep), batches)

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
