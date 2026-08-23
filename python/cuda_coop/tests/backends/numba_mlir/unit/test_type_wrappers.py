# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_opaque_extension_storage_fails_closed_without_exact_abi_layout():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.datamodel import default_manager
    from numba_cuda_mlir.numba_cuda.datamodel.models import StructModel
    from numba_cuda_mlir.numba_cuda.models import cuda_data_manager

    from cuda.coop.numba_mlir._types import numba_type_to_wrapper

    from ..support.padded_extension import padded_opaque_type

    cuda_model = cuda_data_manager.chain(default_manager).lookup(padded_opaque_type)
    assert isinstance(cuda_model, StructModel)
    assert tuple(
        cuda_model.get_type(index) for index in range(cuda_model.field_count)
    ) == (
        types.int32,
        types.int32,
    )

    with pytest.raises(
        TypeError,
        match=(
            r"cannot safely materialize CUB storage for dtype PaddedOpaque: "
            r"exact ABI size and alignment are unavailable"
        ),
    ):
        numba_type_to_wrapper(padded_opaque_type)


def test_registered_cuda_struct_model_retains_inspectable_layout():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from cuda.coop.numba_mlir._types import numba_type_to_wrapper

    from ..support.runtime import numba_mlir_keypair_type

    wrapper = numba_type_to_wrapper(numba_mlir_keypair_type)

    assert "struct __align__(4) storage_t" in wrapper.code
    assert "char data[8]" in wrapper.code


def test_mismatched_cuda_and_mlir_struct_models_fail_closed():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from cuda.coop.numba_mlir._types import numba_type_to_wrapper

    from ..support.padded_extension import mismatched_struct_type

    with pytest.raises(
        TypeError,
        match=(
            r"cannot safely materialize CUB storage for dtype MismatchedStruct: "
            r"exact ABI size and alignment are unavailable"
        ),
    ):
        numba_type_to_wrapper(mismatched_struct_type)


def test_padded_aggregate_type_retains_structural_layout():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from numba_cuda_mlir import types
    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    from cuda.coop.numba_mlir._types import numba_type_to_wrapper

    dtype = AggregateType(
        "CudaCoopPaddedAggregateTypeUnit",
        [("tag", types.uint8), ("payload", types.int64)],
    )
    wrapper = numba_type_to_wrapper(dtype)

    assert "struct __align__(8) storage_t" in wrapper.code
    assert "char data[16]" in wrapper.code
