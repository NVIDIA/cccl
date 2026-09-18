# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Compiler-owned launch facts and integral results for group methods."""

import pytest

cutlass = pytest.importorskip("cutlass")
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer():
    return make_ptr(cutlass.Uint64, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_hierarchy_compile(api):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        output = cute.make_tensor(memory, cute.make_layout(32))
        thread = api.this_thread()
        warp = api.this_warp()
        block = api.this_block()
        grid = api.this_grid()
        lanes = warp.group_by(3, exhaustive=False)
        warps = block.group_by(2, exhaustive=False)
        assert isinstance(block.rank(), cutlass.Uint32)
        assert isinstance(block.count("grid"), cutlass.Uint64)
        assert isinstance(grid.count(), cutlass.Uint64)
        assert isinstance(lanes.is_member(), cutlass.Uint8)
        output[0] = thread.rank("block")
        output[1] = warp.count("block")
        output[2] = grid.count()
        output[3] = lanes.count()
        output[4] = warps.is_member()
        output[5] = warps.count("block")
        thread.sync()
        warp.sync()
        block.sync_aligned()

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=(2, 3, 1), block=(8, 4, 3))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("method", ("rank", "count", "is_member", "sync"))
def test_missing_exact_block(method):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = getattr(cutlass_coop.this_block(), method)()
        if cutlass.const_expr(method != "sync"):
            cute.make_tensor(memory, cute.make_layout(1))[0] = value

    @cute.jit
    def launch(memory: cute.Pointer, size: cutlass.Int32):
        kernel(memory).launch(grid=1, block=(size, 1, 1))

    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), 64)


@pytest.mark.parametrize("group_kind", ("grid", "mapped"))
@pytest.mark.parametrize("method", ("sync", "sync_aligned"))
def test_unsupported_synchronization(group_kind, method):
    @cute.kernel
    def kernel():
        if cutlass.const_expr(group_kind == "grid"):
            group = cutlass_coop.this_grid()
        else:
            group = cutlass_coop.this_block().group_by(2)
        getattr(group, method)()

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match="synchronization"):
        cute.compile[(GPUArch("sm_80"),)](launch)


@pytest.mark.parametrize("dtype", (float, bool, cutlass.Float32, cutlass.Float64))
def test_query_rejects_noninteger(dtype):
    @cute.kernel
    def kernel():
        cutlass_coop.this_block().rank_as(dtype)

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match="query dtype"):
        cute.compile[(GPUArch("sm_80"),)](launch)


def test_higher_mapped_query_rejected():
    @cute.kernel
    def kernel():
        cutlass_coop.this_warp().group_by(8).rank("block")

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match="immediate parent"):
        cute.compile[(GPUArch("sm_80"),)](launch)
