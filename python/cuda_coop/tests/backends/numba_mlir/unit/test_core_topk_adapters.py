# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_topk_adapter_lowers_core_signature_without_codegen():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import ArgumentBinding, make_block_topk_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    core_spec = make_block_topk_spec(
        key_dtype=types.int32,
        value_dtype=types.float32,
        block_dim=(64, 1, 1),
        items_per_thread=2,
        selection="min",
        num_valid=ArgumentBinding.runtime(),
        begin_bit=ArgumentBinding.static(0),
        end_bit=ArgumentBinding.static(32),
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert specialization.method_name == "min_pairs_partial"
    assert specialization.struct_name.startswith("BlockTopKCoop<")
    assert [type(parameter).__name__ for parameter in specialization.parameters[0]] == [
        "Pointer",
        "Array",
        "Array",
        "Value",
        "Value",
        "CxxFunction",
        "CxxFunction",
    ]
