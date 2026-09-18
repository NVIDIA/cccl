# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Physical Warp launch, algorithm, storage, and tile contracts in real traces."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core import this_warp
from cuda.coop.cutlass._compiler import _bundle, _rendering
from cuda.coop.cutlass._lowering._load_store import _CubLoadStoreRequest
from tests.support.group_planning import _load_store, _plan

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer():
    return make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)


def test_exact_per_group_scratch():
    operation = _load_store(dtype=cutlass.Int32, algorithm="transpose")
    requests = [
        _CubLoadStoreRequest(_plan(this_warp(), operation, launch=block), cutlass.Int32)
        for block in ((32, 1, 1), (8, 4, 2))
    ]
    probes = _rendering.bundle_scratch_layout_probes(requests)
    compilation = _bundle.compile_bundle_source_with_layouts(
        _rendering.render_bundle_source(requests),
        arch="compute_80",
        required_headers=tuple(_rendering.registered_bundle_headers().values()),
        layout_probes=tuple(probes.values()),
    )
    one, two = [
        compilation.layouts[request.scratch_requirement_key] for request in requests
    ]
    assert one.size_in_bytes > 0
    assert two.size_in_bytes == 2 * one.size_in_bytes
    assert one.alignment == two.alignment


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", ("direct", "striped", "vectorize", "transpose"))
def test_physical_warp_compile(api, algorithm):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        payload = api.ThreadData(2, dtype=cutlass.Int32)
        api.load(
            api.this_warp(),
            source,
            payload,
            algorithm=algorithm,
            valid_items=47,
            oob_default=-13,
            offset=5,
        )
        api.store(
            api.this_warp(),
            destination,
            payload,
            algorithm=algorithm,
            valid_items=47,
            offset=9,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=(8, 4, 2))

    result = cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), _pointer())
    assert result is not None


@pytest.mark.parametrize("block", (1, 31, 33, 48))
def test_incomplete_warps_fail(block):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(cutlass_coop.this_warp(), memory, cutlass_coop.ThreadData(2))

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=block)

    with pytest.raises(Exception, match="complete"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


def test_dynamic_dimensions_fail():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(cutlass_coop.this_warp(), memory, cutlass_coop.ThreadData(2))

    @cute.jit
    def launch(memory: cute.Pointer, block_size: cutlass.Int32):
        kernel(memory).launch(grid=1, block=(block_size, 1, 1))

    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), 64)


@pytest.mark.parametrize("algorithm", ("warp_transpose", "warp_transpose_timesliced"))
def test_block_algorithms_fail(algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_warp(),
            memory,
            cutlass_coop.ThreadData(2),
            algorithm=algorithm,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="algorithm"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", ("direct", "striped", "vectorize", "transpose"))
def test_warp_storage_descriptor_fails(api, algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        api.load(
            api.this_warp(),
            memory,
            api.ThreadData(2),
            algorithm=algorithm,
            temp_storage=api.TempStorage(1024),
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="(?i)temp_?storage"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


@pytest.mark.parametrize("valid", (-1, 65, 128))
def test_count_excludes_other_warps(valid):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_warp(),
            memory,
            cutlass_coop.ThreadData(2),
            valid_items=valid,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="valid_items"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


def test_extent_includes_second_warp():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        inputs = cute.make_tensor(memory, cute.make_layout(128))
        cutlass_coop.load(
            cutlass_coop.this_warp(), inputs, cutlass_coop.ThreadData(2), offset=1
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="(?i)(extent|elements|size)"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
