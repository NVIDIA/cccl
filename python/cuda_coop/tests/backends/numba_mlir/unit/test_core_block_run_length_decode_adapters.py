# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_fused_adapter_preserves_driver_variants_and_abi():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import (
        BlockRunLengthDecodeStage,
        make_block_run_length_decode_spec,
    )
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    core_spec = make_block_run_length_decode_spec(
        item_dtype=types.int32,
        run_length_dtype=types.uint32,
        decoded_offset_dtype=types.uint64,
        total_decoded_size_dtype=types.uint64,
        relative_offset_dtype=types.uint32,
        block_dim=(32, 1, 1),
        runs_per_thread=2,
        decoded_items_per_thread=4,
        stage=BlockRunLengthDecodeStage.FUSED,
        with_relative_offsets=True,
        with_decoded_window_offset=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert specialization.method_name == "DecodeWithOffsetsAt"
    assert [type(parameter).__name__ for parameter in specialization.parameters[0]] == [
        "Pointer",
        "Array",
        "Array",
        "PointerReference",
        "Array",
        "Array",
        "Reference",
    ]
    assert len(specialization.type_definitions) == 1


def test_numba_mlir_factory_rejects_legacy_integer_coercions():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block._block_run_length_decode import (
        _run_length_decode,
    )

    with pytest.raises(
        ValueError,
        match="decoded_items_per_thread must be a positive integer",
    ):
        _run_length_decode(
            types.int32,
            types.uint32,
            types.uint32,
            types.uint32,
            32,
            2,
            1.5,
        )
