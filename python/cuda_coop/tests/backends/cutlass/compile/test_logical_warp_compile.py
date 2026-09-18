# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Compile logical-warp primitives against exact launch and scratch facts."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core import ArgumentBinding, GroupLoadStoreKind, LaunchFacts, this_warp
from cuda.coop.cutlass._compiler import _bundle, _rendering
from cuda.coop.cutlass._compiler._types import ScratchLayoutProbe
from cuda.coop.cutlass._lowering import _load_store

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]

_WIDTHS = (1, 2, 4, 8, 16, 32)
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")


def _pointer():
    return make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("width", _WIDTHS)
def test_exact_subgroup_scratch(width):
    requests = [
        _load_store._make_request(
            group=this_warp().group_by(width),
            launch=LaunchFacts(exact_block_dim=block),
            kind=GroupLoadStoreKind(kind),
            value_type=cutlass.Int32,
            items_per_thread=2,
            algorithm="transpose",
            valid_items_binding=ArgumentBinding.omitted(),
            oob_default_binding=ArgumentBinding.omitted(),
            offset_binding=ArgumentBinding.omitted(),
        )
        for kind in ("load", "store")
        for block in ((32, 1, 1), (8, 4, 2))
    ]
    probes = list(_rendering.bundle_scratch_layout_probes(requests).values())
    for request in requests[::2]:
        probes.append(
            ScratchLayoutProbe(
                requirement_key=("single_group", request.operation.kind),
                size_expression=f"sizeof(typename {request.cpp_type}::TempStorage)",
                alignment_expression=f"alignof(typename {request.cpp_type}::TempStorage)",
            )
        )
    compilation = _bundle.compile_bundle_source_with_layouts(
        _rendering.render_bundle_source(requests),
        arch="compute_80",
        required_headers=tuple(_rendering.registered_bundle_headers().values()),
        layout_probes=tuple(probes),
    )
    for index, request in enumerate(requests):
        single = compilation.layouts[("single_group", request.operation.kind)]
        block_threads = 32 if index % 2 == 0 else 64
        actual = compilation.layouts[request.scratch_requirement_key]
        assert single.size_in_bytes > 0
        assert actual.size_in_bytes == single.size_in_bytes * (block_threads // width)
        assert actual.alignment == single.alignment


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_logical_warp_compile(width, algorithm, api):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        group = api.this_warp().group_by(width)
        payload = api.ThreadData(2, dtype=cutlass.Int32)
        api.load(
            group,
            source,
            payload,
            algorithm=algorithm,
            valid_items=2 * width - 1,
            oob_default=-13,
            offset=5,
        )
        api.store(
            group,
            destination,
            payload,
            algorithm=algorithm,
            valid_items=2 * width - 1,
            offset=9,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), _pointer()) is not None


@pytest.mark.parametrize("width", _WIDTHS)
def test_nonexhaustive_divisor_compile(width):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_warp().group_by(width, exhaustive=False),
            memory,
            cutlass_coop.ThreadData(2),
            algorithm="transpose",
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_explicit_storage_rejected(api, algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        api.load(
            api.this_warp().group_by(8),
            memory,
            api.ThreadData(2),
            algorithm=algorithm,
            temp_storage=api.TempStorage(1024),
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(
        Exception,
        match="temp_storage is not supported|explicit TempStorage is supported only",
    ):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


@pytest.mark.parametrize("case", ("width_three", "nested", "mapped_warps"))
@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_unsupported_mapping(case, api):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        if cutlass.const_expr(case == "width_three"):
            group = api.this_warp().group_by(3, exhaustive=False)
        elif cutlass.const_expr(case == "nested"):
            group = api.this_warp().group_by(8).group_by(4)
        else:
            group = api.this_block().group_by(1)
        api.load(group, memory, api.ThreadData(2))

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    expected = {
        "width_three": "power-of-two group width",
        "nested": "nested ThreadGroup.group_by is not supported",
        "mapped_warps": "does not support group kind|requires a block, physical warp, or logical warp group",
    }[case]
    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


def test_partial_physical_warp_fails():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_warp().group_by(8), memory, cutlass_coop.ThreadData(2)
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 3, 1))

    with pytest.raises(Exception, match="complete"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())


def test_dynamic_dimensions_fail():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(
            cutlass_coop.this_warp().group_by(8), memory, cutlass_coop.ThreadData(2)
        )

    @cute.jit
    def launch(memory: cute.Pointer, block_size: cutlass.Int32):
        kernel(memory).launch(grid=1, block=(block_size, 1, 1))

    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), 64)


def test_extent_includes_last_group():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        inputs = cute.make_tensor(memory, cute.make_layout(128))
        cutlass_coop.load(
            cutlass_coop.this_warp().group_by(8),
            inputs,
            cutlass_coop.ThreadData(2),
            offset=1,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    with pytest.raises(Exception, match="(?i)(extent|elements|size)"):
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
