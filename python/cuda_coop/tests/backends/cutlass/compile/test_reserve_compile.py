# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check reservation diagnostics and native typed views without a launch."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(launch, *args):
    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer, *args)


@pytest.mark.parametrize(
    "dtype",
    (*ALL_PROVIDER_TYPES, cutlass.Float16, cutlass.BFloat16, np.float16),
)
def test_supported_scalar_views(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        values = coop.TempStorage().reserve(64, dtype, alignment=32)
        values[0] = values.element_type(7)
        output = cute.make_tensor(memory, cute.make_layout(1))
        output[0] = cutlass.Int32(values[0])

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    assert _compile(launch) is not None


@pytest.mark.parametrize(
    "count,dtype,alignment,diagnostic",
    [
        (True, cutlass.Int32, None, "compile-time integer"),
        (0, cutlass.Int32, None, "must be positive"),
        (-1, cutlass.Int32, None, "must be positive"),
        (1.5, cutlass.Int32, None, "compile-time integer"),
        (64, object, None, "supported fixed-width"),
        (64, np.complex64, None, "supported fixed-width"),
        (64, 3, None, "compile-time scalar type"),
        (64, cutlass.Int32, 3, "power of 2"),
        (64, cutlass.Int32, 0, "positive integer"),
    ],
)
def test_invalid_reservation_arguments(count, dtype, alignment, diagnostic):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        coop.TempStorage().reserve(count, dtype, alignment=alignment)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match=diagnostic):
        _compile(launch)


def test_runtime_extent_is_rejected():
    @cute.kernel
    def kernel(memory: cute.Pointer, count: cutlass.Int32):
        coop.TempStorage().reserve(count, cutlass.Int32)

    @cute.jit
    def launch(memory: cute.Pointer, count: cutlass.Int32):
        kernel(memory, count).launch(grid=1, block=64)

    with pytest.raises(Exception, match="compile-time integer"):
        _compile(launch, cutlass.Int32(64))


@pytest.mark.parametrize(
    "sharing,capacity", (("shared", 128), ("exclusive", 256))
)
def test_capacity_accounts_for_every_reservation(sharing, capacity):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        storage = coop.TempStorage(capacity, sharing=sharing)
        storage.reserve(64, cutlass.Int32)
        storage.reserve(64, cutlass.Int32)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="capacity is smaller"):
        _compile(launch)


@pytest.mark.parametrize(
    "order", ("reserve_only", "primitive_first", "reserve_first")
)
def test_auto_sync_rejected_through_aliases_and_policy_changes(order):
    """Audit prior and later primitive uses, including a mutated descriptor."""

    @cute.jit
    def reserve(storage):
        alias = storage
        return alias.reserve(64, cutlass.Int32)

    @cute.kernel
    def kernel(memory: cute.Pointer):
        storage = coop.TempStorage(auto_sync=order != "reserve_first")
        alias = storage
        if cutlass.const_expr(order == "primitive_first"):
            coop.sum(coop.this_block(), cutlass.Int32(1), temp_storage=alias)
            storage.auto_sync = False
        reserve(alias)
        if cutlass.const_expr(order == "reserve_first"):
            alias.auto_sync = True
            coop.sum(coop.this_block(), cutlass.Int32(1), temp_storage=storage)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    with pytest.raises(Exception, match="reserve requires auto_sync=False"):
        _compile(launch)
