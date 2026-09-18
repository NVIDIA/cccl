# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    INT64,
    ArgumentBinding,
    ArgumentKind,
    Array,
    Dependency,
    Pointer,
    PointerOffset,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.warp import (
    WarpLoadStoreAlgorithm,
    WarpLoadStoreKind,
    make_warp_load_spec,
    make_warp_load_store_semantics,
    make_warp_store_spec,
)


def test_warp_load_full_tile_semantics_are_physical_warp_scoped():
    spec = make_warp_load_spec(
        dtype="i32",
        items_per_thread=3,
        algorithm="striped",
    )

    assert spec.kind is WarpLoadStoreKind.LOAD
    assert spec.algorithm is WarpLoadStoreAlgorithm.STRIPED
    assert spec.algorithm_cpp == "::cub::WARP_LOAD_STRIPED"
    assert spec.threads_in_warp == 32
    assert spec.items_per_thread == 3
    assert spec.has_full_tile
    assert not spec.has_valid_items
    assert spec.specialization.template_arguments == {
        "T": "i32",
        "ITEMS_PER_THREAD": 3,
        "ALGORITHM": "::cub::WARP_LOAD_STRIPED",
        "LOGICAL_WARP_THREADS": 32,
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Pointer(
                Dependency("T"),
                name="src",
                is_array_pointer=True,
                restrict=True,
            ),
            Array(
                Dependency("T"),
                Dependency("ITEMS_PER_THREAD"),
                name="dst",
                is_output=True,
                is_return=False,
            ),
        ),
    )


@pytest.mark.parametrize("threads_in_warp", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("make_spec", [make_warp_load_spec, make_warp_store_spec])
def test_warp_load_store_supports_cub_logical_warp_widths(
    make_spec,
    threads_in_warp,
):
    spec = make_spec(
        dtype="i32",
        items_per_thread=2,
        threads_in_warp=threads_in_warp,
        algorithm="direct",
    )

    assert spec.threads_in_warp == threads_in_warp
    assert (
        spec.specialization.template_arguments["LOGICAL_WARP_THREADS"]
        == threads_in_warp
    )
    assert spec.specialization.metadata["effective_offset_stride"] == (
        2 * threads_in_warp
    )


def test_warp_store_partial_runtime_effective_offset_abi():
    spec = make_warp_store_spec(
        dtype="i64",
        items_per_thread=2,
        algorithm="vectorize",
        valid_items=ArgumentBinding.runtime(),
        include_pointer_offset=ArgumentBinding.runtime(),
    )

    assert spec.kind is WarpLoadStoreKind.STORE
    assert spec.requires_runtime_effective_offset
    assert not spec.has_full_tile
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Pointer(
                Dependency("T"),
                name="dst",
                is_output=True,
                is_return=False,
                is_array_pointer=True,
                restrict=True,
            ),
            Array(Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="src"),
            Value(INT32, name="num_valid_items"),
            PointerOffset(INT64, name="offset", pointer_arg_index=0),
        ),
    )
    assert spec.specialization.metadata["requires_runtime_effective_offset"]
    assert spec.specialization.metadata["effective_offset_origin"] == ("group_instance")
    assert spec.specialization.metadata["effective_offset_stride"] == 64


def test_partial_transpose_load_preserves_invalid_payload_items():
    preserving = make_warp_load_spec(
        dtype="i32",
        items_per_thread=2,
        algorithm="transpose",
        threads_in_warp=8,
        valid_items=True,
    )
    defaulting = make_warp_load_spec(
        dtype="i32",
        items_per_thread=2,
        algorithm="transpose",
        valid_items=True,
        oob_default=True,
    )
    direct = make_warp_load_spec(
        dtype="i32",
        items_per_thread=2,
        algorithm="direct",
        valid_items=True,
    )

    assert preserving.specialization.struct_name == (
        "CudaCoopWarpLoadPreservingInvalid"
    )
    assert preserving.specialization.metadata["preserves_invalid_items"]
    assert len(preserving.specialization.type_definitions) == 1
    wrapper = preserving.specialization.type_definitions[0].code
    assert "original[item] = items[item]" in wrapper
    assert "::cuda::ptx::get_sreg_laneid()" in wrapper
    assert "get_sreg_laneid()) %" in wrapper
    assert "LogicalWarpThreads;" in wrapper
    assert "lane * ItemsPerThread + item >= valid_items" in wrapper
    assert "(LogicalWarpThreads & (LogicalWarpThreads - 1)) == 0" in wrapper
    assert defaulting.specialization.struct_name == "WarpLoad"
    assert not defaulting.specialization.metadata["preserves_invalid_items"]
    assert defaulting.specialization.type_definitions == ()
    assert direct.specialization.struct_name == "WarpLoad"
    assert not direct.specialization.metadata["preserves_invalid_items"]


def test_warp_load_store_support_exactly_four_algorithms():
    assert {algorithm.value for algorithm in WarpLoadStoreAlgorithm} == {
        "direct",
        "striped",
        "vectorize",
        "transpose",
    }
    specs = [
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=2,
            algorithm=algorithm,
            threads_in_warp=8,
        )
        for algorithm in WarpLoadStoreAlgorithm
    ]
    assert len({spec.semantic_key for spec in specs}) == 4


def test_warp_load_store_width_is_part_of_specialization_identity():
    specs = [
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=2,
            algorithm="direct",
            threads_in_warp=threads_in_warp,
        )
        for threads_in_warp in (8, 16, 32)
    ]

    assert len({spec.call.semantic_key for spec in specs}) == len(specs)
    assert len({spec.semantic_key for spec in specs}) == len(specs)


@pytest.mark.parametrize(
    "threads_in_warp",
    [True, 0, -1, 3, 6, 12, 24, 31, 33, 64, 8.0],
)
def test_warp_load_store_rejects_unsupported_widths(threads_in_warp):
    with pytest.raises(ValueError, match="threads_in_warp in"):
        make_warp_load_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=threads_in_warp,
            algorithm="direct",
        )


@pytest.mark.parametrize("make_spec", [make_warp_load_spec, make_warp_store_spec])
@pytest.mark.parametrize("valid_items", [-1, 17])
def test_warp_load_store_rejects_static_valid_items_outside_tile(
    make_spec,
    valid_items,
):
    with pytest.raises(ValueError, match="warp tile size"):
        make_spec(
            dtype="i32",
            items_per_thread=2,
            threads_in_warp=8,
            algorithm="direct",
            valid_items=ArgumentBinding.static(valid_items),
        )


def test_warp_load_store_normalizes_static_controls_in_semantic_identity():
    kwargs = {
        "kind": "load",
        "dtype": "i32",
        "items_per_thread": 2,
        "algorithm": "direct",
    }
    plain = make_warp_load_store_semantics(
        **kwargs,
        valid_items=ArgumentBinding.static(5),
        include_pointer_offset=ArgumentBinding.static(7),
    )
    numpy = make_warp_load_store_semantics(
        **kwargs,
        valid_items=ArgumentBinding.static(np.int32(5)),
        include_pointer_offset=ArgumentBinding.static(np.int64(7)),
    )

    assert plain.semantic_key == numpy.semantic_key
    assert numpy.valid_items == ArgumentBinding.static(5)
    assert numpy.pointer_offset == ArgumentBinding.static(7)
    assert not numpy.requires_runtime_effective_offset


def test_warp_load_store_rejects_invalid_options():
    with pytest.raises(ValueError, match="unsupported WarpLoad algorithm"):
        make_warp_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=1,
            algorithm="warp_transpose",
        )
    with pytest.raises(ValueError, match="only valid for WarpLoad"):
        make_warp_load_store_semantics(
            kind="store",
            dtype="i32",
            items_per_thread=1,
            algorithm="direct",
            valid_items=True,
            oob_default=True,
        )
    with pytest.raises(ValueError, match="requires a valid_items"):
        make_warp_load_store_semantics(
            kind="load",
            dtype="i32",
            items_per_thread=1,
            algorithm="direct",
            oob_default=True,
        )


def test_warp_provider_parameter_classification_is_stable():
    spec = make_warp_store_spec(
        dtype="i32",
        items_per_thread=1,
        algorithm="direct",
    )

    assert [entry.kind for entry in spec.specialization.classify_method()] == [
        ArgumentKind.RUNTIME,
        ArgumentKind.RUNTIME,
        ArgumentKind.RUNTIME,
    ]
