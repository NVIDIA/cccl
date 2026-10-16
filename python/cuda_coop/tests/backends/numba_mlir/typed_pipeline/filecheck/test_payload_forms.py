# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A ThreadData's ``items_per_thread``, and local-array payloads.

``items_per_thread`` types as the payload's extent, an integer literal,
whether or not its dtype is known yet. A ``cuda.coop.numba_mlir`` call takes
a ``cuda.local.array`` payload: its type is a plain array, and the site's
extent comes from the array's constant shape. A shape that is a kernel
argument is requested as a literal.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_items_per_thread_is_the_extent_as_a_literal():
    @cuda.jit
    def extent(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        for i in range(items.items_per_thread):
            items[i] = source[i]
        coop.store(block, output, items)

    testing.check(
        extent,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir extent
        CHECK:         items = call
        CHECK-SAME:      :: coop.ThreadData(float32, items=4, store-dst)
        CHECK:         = getattr(value=items, attr=items_per_thread)
        CHECK-SAME:      :: Literal[int](4)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )


def test_a_local_array_sized_by_a_kernel_argument_requests_a_literal():
    # A launch retries the kernel with the argument as a literal, which
    # compiles as in the next test.
    def kernel(source, output, items_per_thread):
        block = qualified_coop.this_block()
        items = cuda.local.array(items_per_thread, types.float32)
        qualified_coop.load(block, source, items)
        qualified_coop.store(block, output, items)

    signature = types.void(*f32_in_out().args, types.int64)
    with pytest.raises(ForceLiteralArg) as raised:
        testing.compile_kernel(kernel, signature, block=BLOCK, chip=CHIP)
    assert raised.value.requested_args == {2}


def test_a_local_array_payload_sized_by_a_literal_argument():
    @cuda.jit
    def local(source, output, items_per_thread):
        block = qualified_coop.this_block()
        items = cuda.local.array(items_per_thread, dtype=types.float32)
        qualified_coop.load(block, source, items, algorithm="transpose")
        qualified_coop.store(block, output, items)

    signature = types.void(*f32_in_out().args, types.IntegerLiteral(4))
    testing.check(
        local,
        signature,
        """
        CHECK-LABEL: typed-ir local
        CHECK:         items_per_thread = arg(2, name=items_per_thread)
        CHECK-SAME:      :: Literal[int](4)
        CHECK:         items = call
        CHECK-SAME:      :: array(float32, 1d, C)
        """,
        dump="typed_ir",
        block=BLOCK,
        chip=CHIP,
    )
    testing.check(
        local,
        signature,
        """
        CHECK-LABEL: coop-plan local
        CHECK:         site #0 load @L{{[0-9]+}} algorithm=transpose
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1
        CHECK-SAME:      region=implicit.0{{$}}
        CHECK:         site #1 store @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1{{$}}
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_a_local_array_payload_of_constant_shape():
    @cuda.jit
    def fixed(source, output):
        block = qualified_coop.this_block()
        items = cuda.local.array(shape=2, dtype=types.float32)
        qualified_coop.load(block, source, items)
        qualified_coop.store(block, output, items, algorithm="striped")

    testing.check(
        fixed,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan fixed
        CHECK:         site #0 load @L{{[0-9]+}} algorithm=direct
        CHECK-SAME:      dtype=float32 items_per_thread=2 block=128x1x1{{$}}
        CHECK-NEXT:    site #1 store @L{{[0-9]+}} algorithm=striped
        CHECK-SAME:      dtype=float32 items_per_thread=2 block=128x1x1{{$}}
        """,
        block=BLOCK,
        chip=CHIP,
    )
