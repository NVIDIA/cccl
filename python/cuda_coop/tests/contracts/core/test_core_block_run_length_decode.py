# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import Array, Dependency, Pointer, Reference, TempStorageParameter
from cuda.coop._core.block import (
    BLOCK_RUN_LENGTH_DECODE_DRIVER,
    BlockRunLengthDecodeOutput,
    BlockRunLengthDecodeStage,
    BlockRunLengthDecodeWindow,
    make_block_run_length_decode_semantics,
    make_block_run_length_decode_spec,
)


def test_constructor_spec_owns_exact_public_cub_specialization_and_abi():
    spec = make_block_run_length_decode_spec(
        item_dtype="i32",
        run_length_dtype="u32",
        decoded_offset_dtype="u64",
        total_decoded_size_dtype="u64",
        block_dim=(16, 2, 1),
        runs_per_thread=3,
        decoded_items_per_thread=4,
        stage=BlockRunLengthDecodeStage.CONSTRUCTOR,
    )

    assert spec.stage is BlockRunLengthDecodeStage.CONSTRUCTOR
    assert spec.method_name == "BlockRunLengthDecode"
    assert spec.specialization.template_parameter_names == (
        "ItemT",
        "BLOCK_DIM_X",
        "RUNS_PER_THREAD",
        "DECODED_ITEMS_PER_THREAD",
        "DecodedOffsetT",
        "BLOCK_DIM_Y",
        "BLOCK_DIM_Z",
    )
    assert spec.specialization.template_arguments == {
        "ItemT": "i32",
        "BLOCK_DIM_X": 16,
        "RUNS_PER_THREAD": 3,
        "DECODED_ITEMS_PER_THREAD": 4,
        "DecodedOffsetT": "u64",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
        # These constructor-template arguments are deliberately auxiliary;
        # they are deduced by CUB and are not class-template arguments.
        "RunLengthT": "u32",
        "TotalDecodedSizeT": "u64",
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("ItemT"),
                Dependency("RUNS_PER_THREAD"),
                name="run_values",
            ),
            Array(
                Dependency("RunLengthT"),
                Dependency("RUNS_PER_THREAD"),
                name="run_lengths",
            ),
            Pointer(
                Dependency("TotalDecodedSizeT"),
                name="total_decoded_size",
                is_array_pointer=True,
                deref_on_call=True,
            ),
        ),
    )


def test_storage_probe_retains_class_shape_without_constructor_dtypes():
    spec = make_block_run_length_decode_spec(
        item_dtype="i32",
        decoded_offset_dtype="u32",
        block_dim=(32, 1, 1),
        runs_per_thread=2,
        decoded_items_per_thread=4,
        stage="constructor",
    )

    assert "RunLengthT" not in spec.specialization.template_arguments
    assert "TotalDecodedSizeT" not in spec.specialization.template_arguments
    assert not spec.call.returns_total_decoded_size
    assert spec.runs_per_thread == 2
    assert spec.decoded_items_per_thread == 4


@pytest.mark.parametrize(
    ("with_offsets", "with_window", "output", "window", "driver_method"),
    [
        (False, False, "items", "default", "Decode"),
        (False, True, "items", "explicit", "DecodeAt"),
        (True, False, "items_and_offsets", "default", "DecodeWithOffsets"),
        (
            True,
            True,
            "items_and_offsets",
            "explicit",
            "DecodeWithOffsetsAt",
        ),
    ],
)
def test_decode_semantics_own_output_and_window_variants(
    with_offsets,
    with_window,
    output,
    window,
    driver_method,
):
    call = make_block_run_length_decode_semantics(
        item_dtype="i32",
        run_length_dtype="u32",
        decoded_offset_dtype="u64",
        total_decoded_size_dtype="u64",
        relative_offset_dtype="u32" if with_offsets else None,
        runs_per_thread=2,
        decoded_items_per_thread=4,
        with_relative_offsets=with_offsets,
        with_decoded_window_offset=with_window,
    )

    assert call.output is BlockRunLengthDecodeOutput(output)
    assert call.window is BlockRunLengthDecodeWindow(window)
    assert call.driver_method_name == driver_method
    names = [parameter.name for parameter in call.decode_parameters]
    assert names[0] == "decoded_items"
    assert ("relative_offsets" in names) is with_offsets
    assert ("from_decoded_offset" in names) is with_window


def test_native_decode_spec_uses_cub_member_without_temp_storage_operand():
    spec = make_block_run_length_decode_spec(
        item_dtype="i32",
        decoded_offset_dtype="u32",
        relative_offset_dtype="u32",
        block_dim=(32, 1, 1),
        runs_per_thread=2,
        decoded_items_per_thread=4,
        stage="decode",
        with_relative_offsets=True,
        with_decoded_window_offset=True,
    )

    assert spec.method_name == "RunLengthDecode"
    assert spec.specialization.struct_name == "BlockRunLengthDecode"
    assert [type(parameter) for parameter in spec.specialization.parameters[0]] == [
        Array,
        Array,
        Reference,
    ]
    assert not any(
        isinstance(parameter, TempStorageParameter)
        for parameter in spec.specialization.parameters[0]
    )
    assert "RelativeOffsetT" not in spec.specialization.template_parameter_names
    assert spec.specialization.template_arguments["RelativeOffsetT"] == "u32"


def test_fused_driver_owns_combined_mlir_call_without_replacing_public_cub_api():
    spec = make_block_run_length_decode_spec(
        item_dtype="i32",
        run_length_dtype="u32",
        decoded_offset_dtype="u64",
        total_decoded_size_dtype="u64",
        relative_offset_dtype="u32",
        block_dim=(32, 1, 1),
        runs_per_thread=2,
        decoded_items_per_thread=4,
        stage="fused",
        with_relative_offsets=True,
        with_decoded_window_offset=True,
    )

    assert spec.method_name == "DecodeWithOffsetsAt"
    assert spec.specialization.struct_name == "BlockRunLengthDecodeDriver"
    assert len(spec.specialization.template_parameter_names) == 10
    assert spec.specialization.type_definitions == (BLOCK_RUN_LENGTH_DECODE_DRIVER,)
    assert "::cub::BlockRunLengthDecode<" in BLOCK_RUN_LENGTH_DECODE_DRIVER.code
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "run_values",
        "run_lengths",
        "total_decoded_size",
        "decoded_items",
        "relative_offsets",
        "from_decoded_offset",
    ]


def test_semantic_identity_tracks_shapes_and_variants_not_runtime_payloads():
    def make(**kwargs):
        options = {
            "item_dtype": "i32",
            "run_length_dtype": "u32",
            "decoded_offset_dtype": "u32",
            "total_decoded_size_dtype": "u32",
            "runs_per_thread": 2,
            "decoded_items_per_thread": 4,
        }
        options.update(kwargs)
        return make_block_run_length_decode_semantics(**options)

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(runs_per_thread=3).semantic_key
    assert make().semantic_key != make(with_decoded_window_offset=True).semantic_key
    assert (
        make().semantic_key
        != make(
            with_relative_offsets=True,
            relative_offset_dtype="u32",
        ).semantic_key
    )
    assert make().semantic_key != make(returns_total_decoded_size=False).semantic_key


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "two"])
@pytest.mark.parametrize(
    "name",
    ["runs_per_thread", "decoded_items_per_thread"],
)
def test_invalid_item_counts_are_rejected_without_integer_coercion(name, value):
    options = {
        "item_dtype": "i32",
        "decoded_offset_dtype": "u32",
        "runs_per_thread": 2,
        "decoded_items_per_thread": 4,
    }
    options[name] = value
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        make_block_run_length_decode_semantics(**options)


def test_invalid_dtype_and_variant_combinations_are_rejected():
    base = {
        "item_dtype": "i32",
        "decoded_offset_dtype": "u32",
        "runs_per_thread": 2,
        "decoded_items_per_thread": 4,
    }
    with pytest.raises(ValueError, match="must be provided together"):
        make_block_run_length_decode_semantics(
            **base,
            run_length_dtype="u32",
        )
    with pytest.raises(ValueError, match="relative_offset_dtype must be provided"):
        make_block_run_length_decode_semantics(
            **base,
            with_relative_offsets=True,
        )
    with pytest.raises(ValueError, match="requires with_relative_offsets"):
        make_block_run_length_decode_semantics(
            **base,
            relative_offset_dtype="u32",
        )
    with pytest.raises(ValueError, match="requires total_decoded_size_dtype"):
        make_block_run_length_decode_semantics(
            **base,
            returns_total_decoded_size=True,
        )
    with pytest.raises(ValueError, match="with_relative_offsets must be a boolean"):
        make_block_run_length_decode_semantics(
            **base,
            with_relative_offsets=1,
        )
    with pytest.raises(ValueError, match="fused run-length decode requires"):
        make_block_run_length_decode_spec(
            **base,
            block_dim=(32, 1, 1),
            stage="fused",
        )
    with pytest.raises(ValueError, match="constructor stage cannot select"):
        make_block_run_length_decode_spec(
            **base,
            block_dim=(32, 1, 1),
            stage="constructor",
            with_decoded_window_offset=True,
        )
    with pytest.raises(ValueError, match="fused stage always returns"):
        make_block_run_length_decode_spec(
            **base,
            run_length_dtype="u32",
            total_decoded_size_dtype="u32",
            block_dim=(32, 1, 1),
            stage="fused",
            returns_total_decoded_size=False,
        )
