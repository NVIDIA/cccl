# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_group_planner_requests_configured_launch(optional_backend):
    coop = optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import compiler, types

    def group_first_reduce(value):
        coop.reduce(coop.this_block(), value)

    expected = (
        "whole-function planner requires launch metadata; "
        "compile through a configured kernel launch"
    )
    with pytest.raises(RuntimeError) as exc_info:
        compiler.compile(
            group_first_reduce,
            types.void(types.int32),
            device=False,
            abi="numba",
            cc=(8, 0),
        )

    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value) == expected
    assert exc_info.value.__cause__ is None
