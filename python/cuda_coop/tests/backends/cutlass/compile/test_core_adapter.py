# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import dataclasses
import os

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


def _artifact_with_templated_definition():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts, TypeDefinition
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_adjacent_difference_provider  # noqa: F401
    from cuda.coop.cutlass._dsl._core_adapter import (
        CutlassCoreAdapter,
        CutlassRuntimeIntRange,
    )
    from cuda.coop.cutlass._group_adjacent_difference import (
        _make_group_adjacent_difference_plan,
    )

    plan = _make_group_adjacent_difference_plan(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        dtype=Int32,
        items_per_thread=3,
        direction="left",
        valid_items=31,
        tile_predecessor_item=0,
        source="adapter_test",
    ).require_supported()
    specialization = plan.implementation
    algorithm = dataclasses.replace(
        specialization.algorithm,
        type_definitions=(
            TypeDefinition(
                "cuda_coop_cutlass_test_identity",
                "template <typename T>\n"
                "struct cuda_coop_cutlass_test_identity { using type = T; };",
            ),
        ),
    )
    specialization = algorithm.specialize(
        dict(specialization.template_arguments),
        metadata=dict(specialization.metadata),
    )
    plan = dataclasses.replace(plan, implementation=specialization)
    artifact = CutlassCoreAdapter().materialize(
        specialization,
        plan=plan,
        kind="cuda_coop_cutlass_cub_block_adjacent_difference",
        symbol_name="cuda_coop_cutlass_test_adjacent_difference",
        runtime_int_ranges=(CutlassRuntimeIntRange("valid_items", 0, 192),),
    )
    return artifact


def test_cutlass_core_adapter_renders_typed_cub_wrapper_outside_c_linkage():
    artifact = _artifact_with_templated_definition()

    from cuda.coop.cutlass._dsl import _provider as provider_support
    from cuda.coop.cutlass._dsl._core_adapter import (
        CutlassCoreAdapter,
        render_cutlass_core_artifact,
    )

    assert [parameter.cpp_name for parameter in artifact.abi_parameters] == [
        "input_items_0",
        "input_items_1",
        "input_items_2",
        "output_items_result",
        "valid_items",
        "tile_predecessor_item",
    ]
    assert artifact.bind_ffi_arguments(
        {
            "input_items": (1, 2, 3),
            "valid_items": 31,
            "tile_predecessor_item": 0,
        },
        {"output_items": "result-pointer"},
    ) == (1, 2, 3, "result-pointer", 31, 0)

    renderer = provider_support.bundle_renderer_for(artifact)
    assert renderer is not None
    assert renderer.render is render_cutlass_core_artifact
    assert provider_support.bundle_include_lines([artifact]) == [
        "#include <cub/block/block_adjacent_difference.cuh>",
        "#include <cuda/std/functional>",
    ]
    lines = renderer.render(artifact)
    source = "\n".join(lines)
    assert lines[0] == "}"
    assert "template <typename T>" in source
    assert source.index("template <typename T>") < source.index('extern "C" {')
    assert source.index('extern "C" {') < source.index(
        "void cuda_coop_cutlass_test_adjacent_difference"
    )
    assert source.count("implementation_type(storage).SubtractLeftPartialTile(") == 1
    assert "::cub::BlockAdjacentDifference<int, 8, 4, 2>" in source
    assert "::cuda::std::minus<int>{}" in source
    assert "if (valid_items < 0 || valid_items > 192)" in source
    assert source.count('asm volatile("trap;");') == 1
    assert "cuda_coop_cutlass_block_sync();" in source
    assert "for (" not in source
    assert "shfl" not in source
    assert "atomic" not in source

    doubled_algorithm = dataclasses.replace(
        artifact.specialization.algorithm,
        parameters=(
            artifact.specialization.parameters[0],
            artifact.specialization.parameters[0],
        ),
    )
    doubled_spec = doubled_algorithm.specialize(
        dict(artifact.specialization.template_arguments),
        metadata=dict(artifact.specialization.metadata),
    )
    doubled_plan = dataclasses.replace(artifact.plan, implementation=doubled_spec)
    adapter = CutlassCoreAdapter()
    first = adapter.materialize(
        doubled_spec,
        plan=doubled_plan,
        kind=artifact.kind,
        symbol_name="cuda_coop_cutlass_test_overload",
        method_index=0,
    )
    second = adapter.materialize(
        doubled_spec,
        plan=doubled_plan,
        kind=artifact.kind,
        symbol_name="cuda_coop_cutlass_test_overload",
        method_index=1,
    )
    assert first.semantic_key != second.semantic_key
    assert first.symbol_name != second.symbol_name
    assert second.symbol_name.endswith("_m1")

    generated_second = adapter.materialize(
        doubled_spec,
        plan=doubled_plan,
        kind=artifact.kind,
        method_index=1,
    )
    assert not generated_second.symbol_name.endswith("_m1")


def test_adjacent_difference_symbols_include_exact_multidimensional_shape():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockAdjacentDifferenceDirection
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_adjacent_difference_provider as provider

    def make_request(block_dim):
        return provider._make_request(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=block_dim),
            value_type=Int32,
            items_per_thread=2,
            direction=BlockAdjacentDifferenceDirection.LEFT,
            valid_items=None,
            tile_predecessor_item=None,
            tile_successor_item=None,
            source="shape_test",
        )

    linear = make_request((64, 1, 1))
    multidimensional = make_request((8, 4, 2))

    assert linear.semantic_key != multidimensional.semantic_key
    assert linear.symbol_name != multidimensional.symbol_name
    assert "_b64_" in linear.symbol_name
    assert "_b8x4x2_" in multidimensional.symbol_name


def _merge_sort_artifacts():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Float64, Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import this_block, this_warp
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    block = provider._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        key_type=Int32,
        value_type=Float64,
        items_per_thread=2,
        descending=True,
        valid_items=31,
        oob_default=Int32(-999),
        source="block_adapter_test",
    )
    logical_warp = provider._make_request(
        group=this_warp().group_by(16),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=Float64,
        items_per_thread=2,
        descending=False,
        valid_items=17,
        oob_default=Int32(999),
        source="warp_adapter_test",
    )
    return block, logical_warp


def test_cutlass_core_adapter_renders_one_merge_sort_call_and_exact_storage():
    block, logical_warp = _merge_sort_artifacts()

    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    block_source = "\n".join(render_cutlass_core_artifact(block))
    warp_source = "\n".join(render_cutlass_core_artifact(logical_warp))

    assert block_source.count(".Sort(") == 1
    assert "::cub::BlockMergeSort<int, 8, 2, double, 4, 2>" in block_source
    assert "::cuda::std::greater<int>{}" in block_source
    assert "int* keys_result" in block_source
    assert "double* values_result" in block_source
    assert "int valid_items" in block_source
    assert "int oob_default" in block_source
    assert "if (valid_items < 0 || valid_items > 128)" in block_source
    assert block_source.count('asm volatile("trap;");') == 1
    assert "cuda_coop_cutlass_block_sync();" in block_source
    assert "for (" not in block_source
    assert "if (descending)" not in block_source

    assert warp_source.count(".Sort(") == 1
    assert "::cub::WarpMergeSort<int, 2, 16, double>" in warp_source
    assert "typename implementation_type::TempStorage storage[4];" in warp_source
    assert "cuda_coop_cutlass_linear_tid() / 16u" in warp_source
    assert "::cuda::std::less<int>{}" in warp_source
    assert "int valid_items" in warp_source
    assert "int oob_default" in warp_source
    assert "if (valid_items < 0 || valid_items > 32)" in warp_source
    assert warp_source.count('asm volatile("trap;");') == 1
    assert "cuda_coop_cutlass_warp_sync();" in warp_source
    assert "for (" not in warp_source
    assert "if (descending)" not in warp_source
    assert [
        (guard.logical_name, guard.minimum, guard.maximum)
        for guard in block.runtime_int_ranges
    ] == [("valid_items", 0, 128)]
    assert [
        (guard.logical_name, guard.minimum, guard.maximum)
        for guard in logical_warp.runtime_int_ranges
    ] == [("valid_items", 0, 32)]


def test_merge_sort_artifact_identity_excludes_runtime_partial_values():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import this_block, this_warp
    from cuda.coop.cutlass._dsl import _cub_merge_sort_provider as provider

    def make(valid_items, oob_default, source):
        return provider._make_request(
            group=this_block(),
            launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
            key_type=Int32,
            value_type=None,
            items_per_thread=3,
            descending=False,
            valid_items=valid_items,
            oob_default=Int32(oob_default),
            source=source,
        )

    root = make(17, -999, "cutlass_root")
    scoped = make(31, 999, "scoped_block")

    assert root.semantic_key == scoped.semantic_key
    assert root.symbol_name == scoped.symbol_name
    assert root.symbol_name == (
        "cuda_coop_cutlass_cub_merge_sort_block_b64_keys_ascending_ki32_x3_partial"
    )

    warp_root = provider._make_request(
        group=this_warp(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=3,
        descending=True,
        valid_items=17,
        oob_default=Int32(-999),
        source="cutlass_root",
    )
    warp_scoped = provider._make_request(
        group=this_warp(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=3,
        descending=True,
        valid_items=31,
        oob_default=Int32(999),
        source="scoped_warp",
    )
    assert warp_root == warp_scoped
    assert warp_root.symbol_name == warp_scoped.symbol_name
    assert warp_root.symbol_name == (
        "cuda_coop_cutlass_cub_merge_sort_warp_b64_w32_keys_descending_ki32_x3_partial"
    )


def test_cutlass_core_adapter_templated_definition_compiles_with_nvrtc(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("cuda.bindings.nvrtc")
    artifact = _artifact_with_templated_definition()

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT",
        "ltoir",
    )
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    source = block_provider._render_bundle_source([artifact])
    bundle_path = provider_bundle.compile_bundle_source(
        source,
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_80",
        resolve_nvrtc_arch=lambda: "compute_80",
    )

    assert bundle_path.endswith(".ltoir")
    assert os.path.getsize(bundle_path) > 0


def _discontinuity_and_shuffle_artifacts(block_dim=(8, 4, 2)):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockDiscontinuityMode, BlockShuffleMode
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_discontinuity_provider as discontinuity
    from cuda.coop.cutlass._dsl import _cub_shuffle_provider as shuffle

    launch = LaunchFacts(exact_block_dim=block_dim)
    group = this_block()
    dual = discontinuity._make_request(
        group=group,
        launch=launch,
        value_type=Int32,
        items_per_thread=3,
        mode=BlockDiscontinuityMode.HEADS_AND_TAILS,
        tile_predecessor_item=0,
        tile_successor_item=9,
        source="adapter_test",
    )
    up = shuffle._make_request(
        group=group,
        launch=launch,
        value_type=Int32,
        items_per_thread=3,
        mode=BlockShuffleMode.UP,
        block_prefix=False,
        block_suffix=True,
        source="adapter_test",
    )
    offset = shuffle._make_request(
        group=group,
        launch=launch,
        value_type=Int32,
        items_per_thread=None,
        mode=BlockShuffleMode.OFFSET,
        block_prefix=False,
        block_suffix=False,
        source="adapter_test",
    )
    return dual, up, offset


def test_discontinuity_and_shuffle_render_one_public_cub_call_with_adapters():
    dual, up, offset = _discontinuity_and_shuffle_artifacts()

    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    dual_source = "\n".join(render_cutlass_core_artifact(dual))
    assert dual_source.count(".FlagHeadsAndTails(") == 1
    assert ".FlagHeads(" not in dual_source
    assert ".FlagTails(" not in dual_source
    assert "::cub::BlockDiscontinuity<int, 8, 4, 2>" in dual_source
    assert "head_flags, tile_predecessor_item, tail_flags" in dual_source
    assert "tail_flags, tile_successor_item, input_items" in dual_source
    assert [parameter.cpp_name for parameter in dual.abi_parameters] == [
        "head_flags_result",
        "tile_predecessor_item",
        "tail_flags_result",
        "tile_successor_item",
        "input_items_0",
        "input_items_1",
        "input_items_2",
    ]

    up_source = "\n".join(render_cutlass_core_artifact(up))
    assert up_source.count(".Up(") == 1
    assert "output_items[0] = input_items[0];" in up_source
    assert "output_items[2] = input_items[2];" in up_source
    assert "*block_suffix" in up_source
    assert (
        next(
            parameter
            for parameter in up.abi_parameters
            if parameter.logical_name == "block_suffix"
        ).source
        == "output"
    )

    offset_source = "\n".join(render_cutlass_core_artifact(offset))
    assert offset_source.count(".Offset(") == 1
    assert "output_item = input_item;" in offset_source
    assert "for (" not in dual_source + up_source + offset_source
    assert "shfl" not in dual_source + up_source + offset_source
    assert "atomic" not in dual_source + up_source + offset_source


def test_shuffle_rotate_wrapper_rejects_negative_then_normalizes_distance():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockShuffleMode
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_shuffle_provider as shuffle
    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    rotate = shuffle._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        value_type=Int32,
        items_per_thread=None,
        mode=BlockShuffleMode.ROTATE,
        block_prefix=False,
        block_suffix=False,
        source="adapter_test",
    )
    source = "\n".join(render_cutlass_core_artifact(rotate))

    assert [
        (guard.logical_name, guard.minimum, guard.maximum, guard.modulus)
        for guard in rotate.runtime_int_ranges
    ] == [("distance", 0, 63, 64)]
    assert "if (distance < 0)" in source
    assert "distance %= 64;" in source
    assert source.count('asm volatile("trap;");') == 1
    assert source.count(".Rotate(") == 1


def test_shuffle_offset_wrapper_clamps_extreme_runtime_distance():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockShuffleMode
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_shuffle_provider as shuffle
    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    offset = shuffle._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        value_type=Int32,
        items_per_thread=None,
        mode=BlockShuffleMode.OFFSET,
        block_prefix=False,
        block_suffix=False,
        source="adapter_test",
    )
    source = "\n".join(render_cutlass_core_artifact(offset))

    assert [
        (guard.logical_name, guard.minimum, guard.maximum, guard.clamp)
        for guard in offset.runtime_int_ranges
    ] == [("distance", -64, 64, True)]
    assert "if (distance < -64)" in source
    assert "distance = -64;" in source
    assert "else if (distance > 64)" in source
    assert "distance = 64;" in source
    assert 'asm volatile("trap;");' not in source
    assert source.count(".Offset(") == 1


@pytest.mark.parametrize(
    "mode,distance,expected",
    [
        ("rotate", (1 << 100) + 3, 3),
        ("offset", 1 << 100, 64),
        ("offset", -(1 << 100), -64),
    ],
)
def test_shuffle_normalizes_static_python_distance_before_int32_narrowing(
    mode,
    distance,
    expected,
):
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop._core import LaunchFacts
    from cuda.coop._core.block import BlockShuffleMode
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_shuffle_provider as shuffle

    request = shuffle._make_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        value_type=Int32,
        items_per_thread=None,
        mode=BlockShuffleMode(mode),
        block_prefix=False,
        block_suffix=False,
        source="adapter_test",
    )

    assert (
        shuffle._normalize_static_distance(
            distance,
            request.runtime_int_ranges[0],
        )
        == expected
    )


def test_discontinuity_and_shuffle_symbols_include_exact_shape():
    linear = _discontinuity_and_shuffle_artifacts((64, 1, 1))
    multidimensional = _discontinuity_and_shuffle_artifacts((8, 4, 2))

    for linear_artifact, multidimensional_artifact in zip(
        linear,
        multidimensional,
    ):
        assert linear_artifact.semantic_key != multidimensional_artifact.semantic_key
        assert linear_artifact.symbol_name != multidimensional_artifact.symbol_name
        assert "_b64_" in linear_artifact.symbol_name
        assert "_b8x4x2_" in multidimensional_artifact.symbol_name


def test_discontinuity_and_shuffle_wrappers_compile_with_nvrtc(monkeypatch, tmp_path):
    pytest.importorskip("cuda.bindings.nvrtc")
    artifacts = _discontinuity_and_shuffle_artifacts()

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    source = block_provider._render_bundle_source(list(artifacts))
    bundle_path = provider_bundle.compile_bundle_source(
        source,
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_80",
        resolve_nvrtc_arch=lambda: "compute_80",
    )

    assert bundle_path.endswith(".ltoir")
    assert os.path.getsize(bundle_path) > 0


def _radix_artifacts():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Float64, Int32

    from cuda.coop._core import GroupOperandKind, LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider

    sort_keys = provider._make_sort_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        key_type=Int32,
        value_type=None,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=False,
        source="adapter_test",
    )
    sort_pairs = provider._make_sort_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        key_type=Int32,
        value_type=Float64,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=True,
        source="adapter_test",
    )
    rank = provider._make_rank_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=(64, 1, 1)),
        input_type=Int32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        begin_bit=0,
        end_bit=8,
        descending=False,
        prefix_items=4,
        source="adapter_test",
    )
    return sort_keys, sort_pairs, rank


def test_cutlass_radix_artifacts_render_one_public_cub_call_and_typed_outputs():
    sort_keys, sort_pairs, rank = _radix_artifacts()

    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    keys_source = "\n".join(render_cutlass_core_artifact(sort_keys))
    pairs_source = "\n".join(render_cutlass_core_artifact(sort_pairs))
    rank_source = "\n".join(render_cutlass_core_artifact(rank))

    assert keys_source.count("implementation_type(storage).Sort(") == 1
    assert "::cub::BlockRadixSort<int, 64, 2" in keys_source
    assert "int* keys_result" in keys_source
    assert pairs_source.count("implementation_type(storage).SortDescending(") == 1
    assert "::cub::BlockRadixSort<int, 8, 2, double" in pairs_source
    assert ", 4, 2>;" in pairs_source
    assert "double* values_result" in pairs_source
    assert rank_source.count("implementation_type(storage).RankKeys(") == 1
    assert "::cub::BlockRadixRank<64, 8, false" in rank_source
    assert "cudaSharedMemBankSizeEightByte" in rank_source
    assert "unsigned int keys[2]" in rank_source
    assert "^ 0x80000000u" in rank_source
    assert "exclusive_digit_prefix[4] = {-1, -1, -1, -1};" in rank_source
    assert "int* ranks_result" in rank_source
    for source in (keys_source, pairs_source, rank_source):
        assert source.count("cuda_coop_cutlass_block_sync();") == 1
        assert "for (" not in source
        assert "atomic" not in source
        assert "shfl" not in source


def test_cutlass_radix_artifact_identity_tracks_shape_types_and_static_rank_bits():
    sort_keys, sort_pairs, rank = _radix_artifacts()
    from cutlass.base_dsl.typing import Int32, Int64, Uint32

    from cuda.coop._core import GroupOperandKind, LaunchFacts
    from cuda.coop.cutlass import this_block
    from cuda.coop.cutlass._dsl import _cub_radix_provider as provider
    from cuda.coop.cutlass._dsl._core_adapter import render_cutlass_core_artifact

    rank_unsigned = provider._make_rank_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        input_type=Uint32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        begin_bit=0,
        end_bit=8,
        descending=False,
        prefix_items=4,
        source="adapter_test",
    )
    rank_other_bits = provider._make_rank_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        input_type=Int32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        begin_bit=4,
        end_bit=8,
        descending=False,
        prefix_items=1,
        source="adapter_test",
    )
    rank_i64 = provider._make_rank_request(
        group=this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        input_type=Int64,
        items_per_thread=1,
        operand_kind=GroupOperandKind.SCALAR,
        begin_bit=0,
        end_bit=8,
        descending=False,
        prefix_items=None,
        source="adapter_test",
    )

    assert sort_keys.semantic_key != sort_pairs.semantic_key
    assert rank.semantic_key != rank_unsigned.semantic_key
    assert rank.semantic_key != rank_other_bits.semantic_key
    assert "_b8x4x2_" in sort_pairs.symbol_name
    rank_i64_source = "\n".join(render_cutlass_core_artifact(rank_i64))
    assert "unsigned long long keys[1]" in rank_i64_source
    assert "^ 0x8000000000000000ull" in rank_i64_source


def test_cutlass_radix_artifacts_compile_with_nvrtc(monkeypatch, tmp_path):
    pytest.importorskip("cuda.bindings.nvrtc")
    artifacts = _radix_artifacts()

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    source = block_provider._render_bundle_source(list(artifacts))
    bundle_path = provider_bundle.compile_bundle_source(
        source,
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_120",
        resolve_nvrtc_arch=lambda: "compute_120",
    )

    assert bundle_path.endswith(".ltoir")
    assert os.path.getsize(bundle_path) > 0


def test_monolithic_block_provider_has_no_handwritten_radix_fallback():
    provider_source = (
        SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_dsl" / "block" / "_provider.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        'request.kind == "radix_rank"',
        'request.kind == "radix_rank_digit_prefix"',
        'request.kind == "radix_sort_keys"',
        'request.kind == "radix_sort_values"',
        "def provider_radix_rank(",
        "def provider_radix_sort_keys(",
        "def provider_radix_sort_pairs(",
        "def _provider_thread_data_radix_sort_",
    ):
        assert forbidden not in provider_source


@pytest.mark.parametrize(
    ("sm_arch", "compute_arch"),
    [("sm_75", "compute_75"), ("sm_120", "compute_120")],
)
def test_cutlass_merge_sort_wrappers_compile_with_nvrtc(
    monkeypatch,
    tmp_path,
    sm_arch,
    compute_arch,
):
    pytest.importorskip("cuda.bindings.nvrtc")
    block, logical_warp = _merge_sort_artifacts()

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR",
        str(tmp_path / "provider-cache"),
    )
    source = block_provider._render_bundle_source([block, logical_warp])
    bundle_path = provider_bundle.compile_bundle_source(
        source,
        scope=block_provider._SCOPE,
        provider_dir=os.path.dirname(block_provider.__file__),
        registered_headers=block_provider._registered_cccl_headers,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: sm_arch,
        resolve_nvrtc_arch=lambda: compute_arch,
    )

    assert source.count(".Sort(") == 2
    assert bundle_path.endswith(".ltoir")
    assert os.path.getsize(bundle_path) > 0
