# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_radix_rank_adapter_substitutes_digit_extractor_type():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_radix_rank_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    core_spec = make_block_radix_rank_spec(
        key_dtype=types.uint32,
        key_bit_width=32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        begin_bit=4,
        end_bit=9,
        with_exclusive_digit_prefix=True,
    )
    adapter = NumbaMlirCoreAdapter()
    specialization = adapter.materialize(core_spec.specialization)

    assert [type(parameter).__name__ for parameter in specialization.parameters[0]] == [
        "Pointer",
        "Array",
        "Array",
        "CxxFunction",
        "Array",
    ]
    extractor = specialization.parameters[0][3]
    assert extractor.cpp == (
        f"::cub::BFEDigitExtractor<{adapter.cpp_type(types.uint32)}>(4, 5)"
    )
    assert not specialization.parameters[0][2].is_output
    assert not specialization.parameters[0][4].is_output


@pytest.mark.parametrize(
    ("source_dtype_name", "target_dtype_name", "sign_mask"),
    [
        ("int32", "uint32", "0x80000000u"),
        ("int64", "uint64", "0x8000000000000000ull"),
    ],
)
def test_numba_mlir_radix_rank_adapter_twiddles_signed_keys_before_cub(
    source_dtype_name,
    target_dtype_name,
    sign_mask,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_radix_rank_spec
    from cuda.coop.numba_mlir import _types as backend_types
    from cuda.coop.numba_mlir._core_adapter import (
        NumbaMlirArrayInputTransform,
        NumbaMlirCoreAdapter,
    )

    source_dtype = getattr(types, source_dtype_name)
    target_dtype = getattr(types, target_dtype_name)
    expression = f"(static_cast<unsigned long long>({{value}}) ^ {sign_mask})"
    if source_dtype_name == "int32":
        expression = f"(static_cast<unsigned int>({{value}}) ^ {sign_mask})"
    core_spec = make_block_radix_rank_spec(
        key_dtype=target_dtype,
        key_bit_width=source_dtype.bitwidth,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        begin_bit=source_dtype.bitwidth - 4,
        end_bit=source_dtype.bitwidth,
    )
    specialization = NumbaMlirCoreAdapter(
        input_transforms={
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=source_dtype,
                cpp_expression=expression,
            )
        }
    ).materialize(core_spec.specialization)

    keys = specialization.parameters[0][1]
    assert isinstance(keys, backend_types.TransformedArray)
    assert keys.value_dtype == source_dtype
    assert keys.target_dtype == target_dtype
    assert keys.size == 2

    source, *_ = specialization._source_code()
    assert expression.format(value="param_0[0]") in source
    assert expression.format(value="param_0[1]") in source
    assert (
        f"BFEDigitExtractor<{NumbaMlirCoreAdapter().cpp_type(target_dtype)}>" in source
    )


@pytest.mark.parametrize(
    ("parameter_name", "message"),
    [
        ("missing", "unknown Numba-CUDA-MLIR input transform"),
        ("ranks", "input-only array parameters"),
    ],
)
def test_numba_mlir_input_transform_rejects_invalid_targets(
    parameter_name,
    message,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_radix_rank_spec
    from cuda.coop.numba_mlir._core_adapter import (
        NumbaMlirArrayInputTransform,
        NumbaMlirCoreAdapter,
    )

    core_spec = make_block_radix_rank_spec(
        key_dtype=types.uint32,
        key_bit_width=32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        begin_bit=28,
        end_bit=32,
    )
    adapter = NumbaMlirCoreAdapter(
        input_transforms={
            parameter_name: NumbaMlirArrayInputTransform(
                source_dtype=types.int32,
                cpp_expression="static_cast<unsigned>({value})",
            )
        }
    )

    with pytest.raises(ValueError, match=message):
        adapter.materialize(core_spec.specialization)


def test_signed_transform_and_native_unsigned_rank_do_not_coalesce():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_radix_rank_spec
    from cuda.coop.numba_mlir._core_adapter import (
        NumbaMlirArrayInputTransform,
        NumbaMlirCoreAdapter,
    )
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    core_spec = make_block_radix_rank_spec(
        key_dtype=types.uint32,
        key_bit_width=32,
        block_dim=(64, 1, 1),
        items_per_thread=2,
        begin_bit=28,
        end_bit=32,
    )
    signed = NumbaMlirCoreAdapter(
        input_transforms={
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=types.int32,
                cpp_expression=("(static_cast<unsigned int>({value}) ^ 0x80000000u)"),
            )
        }
    ).materialize(core_spec.specialization)
    unsigned = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert algo_coalesce_key(signed) != algo_coalesce_key(unsigned)
    assert signed.parameters[0][1].mangled_name() != (
        unsigned.parameters[0][1].mangled_name()
    )
    signed_source, *_ = signed._source_code()
    unsigned_source, *_ = unsigned._source_code()
    assert "^ 0x80000000u" in signed_source
    assert "^ 0x80000000u" not in unsigned_source
