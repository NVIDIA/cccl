# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


def test_topk_keys_and_pairs_materialize_linkable_ltoir() -> None:
    types = pytest.importorskip("numba_cuda_mlir.types")

    from cuda.coop.numba_mlir._lowering import _topk as _topk_lowering

    invocables = (
        _topk_lowering._common_topk_max_keys(
            dtype=types.int32,
            threads_per_block=64,
            items_per_thread=2,
            num_valid=True,
            begin_bit=True,
            end_bit=True,
        ),
        _topk_lowering._qualified_group_topk_min_pairs(
            keys=types.float32,
            values=types.int16,
            threads_per_block=(64, 1, 1),
            items_per_thread=2,
            begin_bit=True,
        ),
    )

    for invocable in invocables:
        assert invocable.temp_storage_bytes > 0
        assert invocable.temp_storage_alignment > 0
        assert len(invocable.files) == 1
        assert invocable.files[0].endswith(".ltoir")
