# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What call resolution decides, read from its plan dump.

Each test compiles one kernel for a stand-in launch of ``BLOCK`` threads and
checks the plan with FileCheck: the sites in program order and the request
each completed to, the scratch region and its backing, and the barriers.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import (
    BLOCK,
    CHIP,
    STORAGE_FREE,
    TRANSPOSING,
    f32_in_out,
    requires_typed_pipeline,
)

pytestmark = requires_typed_pipeline


def test_inferred_payload_takes_the_load_source_dtype():
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
        CHECK-LABEL: coop-plan k1
        CHECK-NEXT:    block (128, 1, 1) from launch
        CHECK-NEXT:    bundle sites=2 specializations=2
        CHECK-NEXT:    site #0 load @L{{[0-9]+}} algorithm=direct dtype=float32
        CHECK-SAME:      items_per_thread=4 block=128x1x1{{$}}
        CHECK-NEXT:    site #1 store @L{{[0-9]+}} algorithm=direct dtype=float32
        CHECK-SAME:      items_per_thread=4 block=128x1x1{{$}}
        CHECK-NOT:     barrier
        """,
        block=BLOCK,
        chip=CHIP,
    )


@pytest.mark.parametrize("algorithm", STORAGE_FREE)
def test_storage_free_algorithms_get_no_scratch_and_no_barrier(algorithm):
    @cuda.jit
    def round_trip(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm)
        coop.store(block, output, items, algorithm=algorithm)

    testing.check(
        round_trip,
        f32_in_out(),
        f"""
        CHECK-LABEL: coop-plan round_trip
        CHECK-NOT:     backing
        CHECK-NOT:     region
        CHECK:         site #0 load {{{{.*}}}} algorithm={algorithm}
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1{{{{$}}}}
        CHECK-NEXT:    site #1 store {{{{.*}}}} algorithm={algorithm}
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1{{{{$}}}}
        CHECK-NOT:     barrier
        """,
        block=BLOCK,
        chip=CHIP,
    )


@pytest.mark.parametrize("algorithm", TRANSPOSING)
def test_transposing_algorithms_share_one_region_with_barriers(algorithm):
    @cuda.jit
    def round_trip(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm=algorithm)
        coop.store(block, output, items, algorithm=algorithm)

    testing.check(
        round_trip,
        f32_in_out(),
        f"""
        CHECK-LABEL: coop-plan round_trip
        CHECK:         backing static size=[[SIZE:[0-9]+]] align=16
        CHECK-NEXT:    region implicit.0 offset=0 size=[[SIZE]] align=16
        CHECK-NEXT:    site #0 load {{{{.*}}}} algorithm={algorithm}
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1
        CHECK-SAME:      region=implicit.0
        CHECK-NEXT:    barrier after #0
        CHECK-NEXT:    site #1 store {{{{.*}}}} algorithm={algorithm}
        CHECK-SAME:      dtype=float32 items_per_thread=4 block=128x1x1
        CHECK-SAME:      region=implicit.0
        CHECK-NEXT:    barrier after #1
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_a_kernel_without_coop_sites_asks_for_no_launch():
    @cuda.jit
    def local_only(source, output):
        items = coop.ThreadData(4, dtype=source.dtype)
        base = cuda.threadIdx.x * 4
        for i in range(4):
            items[i] = source[base + i]
        for i in range(4):
            output[base + i] = items[i]

    testing.check(
        local_only,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan local_only
        CHECK-NEXT:    block not requested
        CHECK-NEXT:    no coop sites
        """,
        chip=CHIP,
    )


def test_all_sites_share_one_bundle():
    @cuda.jit
    def three_sites(source, output):
        block = coop.this_block()
        a = coop.ThreadData(4)
        b = coop.ThreadData(4)
        coop.load(block, source, a)
        coop.load(block, source, b)
        coop.store(block, output, a, algorithm="transpose")

    testing.check(
        three_sites,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan three_sites
        CHECK:         bundle sites=3 specializations=2
        CHECK:         site #0 load {{.*}} algorithm=direct
        CHECK-NEXT:    site #1 load {{.*}} algorithm=direct
        CHECK-NEXT:    site #2 store {{.*}} algorithm=transpose
        CHECK-SAME:      region=implicit.0
        """,
        block=BLOCK,
        chip=CHIP,
    )
