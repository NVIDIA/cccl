# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What inference and call resolution leave in the typed Numba IR.

The dump is taken after the last rewrite, so it shows the ``ThreadData``
types with their evidence and the ``cuda.shared.array`` backing the storage
pass inserted.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_thread_data_takes_its_dtype_from_the_load():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir k1
        CHECK:         block = call {{.*}} :: coop.group[block]
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK-NOT:     $coop_backing
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_explicit_dtype_is_the_strongest_evidence():
    @cuda.jit
    def explicit(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4, dtype=output.dtype)
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        explicit,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir explicit
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, explicit)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_store_destination_does_not_override_typed_writes():
    @cuda.jit
    def fill(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        for i in range(4):
            items[i] = 1
        coop.store(block, output, items)

    with pytest.raises(
        TypingError,
        match="payload dtype int64 conflicts with destination dtype float32",
    ):
        testing.compile_kernel(fill, f32_in_out(), block=BLOCK, chip=CHIP)


def test_a_load_through_a_tuple_alias_types_the_payload():
    @cuda.jit
    def packed(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        alias = (items,)[0]
        coop.load(block, source, alias)
        coop.store(block, output, items)

    testing.check(
        packed,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir packed
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         = build_tuple(items=[Var(items,
        CHECK-SAME:      :: UniTuple(coop.ThreadData(float32, items=4,
        CHECK-SAME:      producer) x 1)
        CHECK:         alias = static_getitem(
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         _coop_store_wrapper_{{.*}}_value = items
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_a_load_through_a_phi_types_both_constructors():
    @cuda.jit
    def joined(source, output, first):
        block = coop.this_block()
        a = coop.ThreadData(4)
        b = coop.ThreadData(4)
        if first[0]:
            items = a
        else:
            items = b
        coop.load(block, source, items)
        coop.store(block, output, items)

    testing.check(
        joined,
        types.void(*f32_in_out().args, types.int32[::1]),
        """
        CHECK-LABEL: typed-ir joined
        CHECK:         a = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         b = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         [[JOIN:items[.][0-9]+]] = items
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        CHECK:         [[OTHER:items[.][0-9]+]] = b
        CHECK-NEXT:    [[JOIN]] = [[OTHER]]
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, producer)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )
