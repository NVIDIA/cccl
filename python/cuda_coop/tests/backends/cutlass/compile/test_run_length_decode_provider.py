# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cuda.coop._core import (  # noqa: E402
    GroupLoweringTarget,
    GroupRunLengthDecodeSemantics,
    LaunchFacts,
    PreconditionEnforcement,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)
from cuda.coop._core.block import (  # noqa: E402
    BLOCK_RUN_LENGTH_DECODE_DRIVER,
    make_block_run_length_decode_semantics,
)


def _operation(*, with_relative_offsets: bool = True):
    return GroupRunLengthDecodeSemantics(
        make_block_run_length_decode_semantics(
            item_dtype="i32",
            run_length_dtype="u32",
            decoded_offset_dtype="u32",
            total_decoded_size_dtype="u32",
            relative_offset_dtype=("u32" if with_relative_offsets else None),
            runs_per_thread=2,
            decoded_items_per_thread=3,
            with_relative_offsets=with_relative_offsets,
            with_decoded_window_offset=True,
            returns_total_decoded_size=True,
        )
    )


def test_group_run_length_decode_plan_records_fused_public_cub_contract():
    call = make_group_primitive_call(this_block(), _operation(), source="root")
    plan = plan_group_primitive(
        call,
        LaunchFacts(exact_block_dim=(8, 4, 2)),
    ).require_supported()

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockRunLengthDecodeDriver"
    assert plan.implementation.method_name == "DecodeWithOffsetsAt"
    assert plan.implementation.type_definitions == (BLOCK_RUN_LENGTH_DECODE_DRIVER,)
    assert plan.implementation.template_arguments == {
        "ItemT": "i32",
        "BLOCK_DIM_X": 8,
        "RUNS_PER_THREAD": 2,
        "DECODED_ITEMS_PER_THREAD": 3,
        "DecodedOffsetT": "u32",
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
        "RunLengthT": "u32",
        "TotalDecodedSizeT": "u32",
        "RelativeOffsetT": "u32",
    }
    assert plan.provenance.header == "cub/block/block_run_length_decode.cuh"
    assert plan.provenance.cpp_class == "cub::BlockRunLengthDecode"
    assert [result.name for result in plan.result.values] == [
        "decoded_items",
        "relative_offsets",
        "total_decoded_size",
    ]
    assert plan.result.values[0].items_per_member == 3
    assert plan.result.values[0].visibility is ResultVisibility.PER_MEMBER
    assert plan.result.values[1].dtype == "u32"
    assert plan.result.values[2].visibility is ResultVisibility.ALL_MEMBERS
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.participation.uniform_arguments == ("decoded_window_offset",)
    assert [item.name for item in plan.participation.argument_preconditions] == [
        "run_lengths",
        "sum(run_lengths)",
        "decoded_window_offset",
    ]
    assert all(
        item.enforcement is PreconditionEnforcement.CALLER
        for item in plan.participation.argument_preconditions
    )
    assert [item.name for item in call.argument_classifications] == [
        "run_values",
        "run_lengths",
        "decoded_items_per_thread",
        "decoded_window_offset",
        "relative_offsets",
        "total_decoded_size",
    ]


def test_group_run_length_decode_requires_exact_block_group():
    warp = plan_group_primitive(
        make_group_primitive_call(this_warp(), _operation()),
        LaunchFacts(exact_block_dim=64),
    )
    missing = plan_group_primitive(
        make_group_primitive_call(this_block(), _operation()),
        LaunchFacts(max_block_dim=64),
    )
    assert warp.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM


def test_cutlass_run_length_decode_rejects_invalid_static_window_offsets():
    from cuda.coop.cutlass._run_length_controls import (
        validate_decoded_window_offset,
    )

    assert validate_decoded_window_offset(9, scope="cuda.coop.cutlass") == 9
    with pytest.raises(ValueError, match="must be nonnegative"):
        validate_decoded_window_offset(-7, scope="cuda.coop.cutlass")
    for value in (False, 1.5, "7", object()):
        with pytest.raises(TypeError, match="must be an integer"):
            validate_decoded_window_offset(value, scope="cuda.coop.cutlass")


def test_cutlass_run_length_decode_rejects_static_offset_overflow():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32, Uint32, Uint64

    from cuda.coop.cutlass._dsl._cub_run_length_decode_provider import (
        _as_decoded_window_offset,
    )

    _as_decoded_window_offset((1 << 31) - 1, length_type=Int32)
    _as_decoded_window_offset((1 << 32) - 1, length_type=Uint32)
    _as_decoded_window_offset(1 << 32, length_type=Uint64)
    with pytest.raises(ValueError, match="does not fit Int32"):
        _as_decoded_window_offset(1 << 31, length_type=Int32)
    with pytest.raises(ValueError, match="does not fit Uint32"):
        _as_decoded_window_offset(1 << 32, length_type=Uint32)


def _requests():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32, Uint32

    from cuda.coop.cutlass import this_block as cutlass_this_block
    from cuda.coop.cutlass._dsl import _cub_run_length_decode_provider as provider

    common = {
        "value_type": Int32,
        "length_type": Uint32,
        "runs_per_thread": 2,
        "decoded_items_per_thread": 3,
        "with_relative_offsets": True,
        "source": "provider_test",
    }
    linear = provider._make_request(
        group=cutlass_this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        **common,
    )
    multidimensional = provider._make_request(
        group=cutlass_this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        **common,
    )
    return linear, multidimensional


def test_run_length_renderer_uses_one_fused_driver_call_and_local_postmask():
    linear, multidimensional = _requests()

    from cuda.coop.cutlass._dsl import _cub_run_length_decode_provider as provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    renderer = provider_support.bundle_renderer_for(multidimensional)
    assert renderer is not None
    assert renderer.render is provider.render_run_length_decode_artifact
    assert "#include <cuda/std/type_traits>" in renderer.include_lines
    assert (
        "#include <cuda/std/type_traits>",
        "cuda/std/type_traits",
    ) in renderer.cccl_headers
    source = "\n".join(renderer.render(multidimensional))
    assert linear.semantic_key != multidimensional.semantic_key
    assert "_b64_" in linear.symbol_name
    assert "_b8x4x2_" in multidimensional.symbol_name
    assert "namespace cub" in source
    assert (
        "::cub::BlockRunLengthDecodeDriver<int, 8, 2, 3, unsigned int, "
        "4, 2, unsigned int, unsigned int, unsigned int>" in source
    )
    assert source.count("implementation_type(storage).DecodeWithOffsetsAt(") == 1
    assert (
        "using decoder_offset_t = "
        "::cuda::std::make_unsigned_t<DecodedOffsetT>;" in source
    )
    assert "const decoder_offset_t safe_from_decoded_offset" in source
    assert "decoded_size == 0" in source
    assert "cuda_coop_cutlass_block_sync();" in source
    assert "decoded_items_result[0] = valid_0" in source
    assert "relative_offsets_result[0] = valid_0" in source
    assert "static_cast<unsigned int>(~0ull)" in source
    assert "*total_decoded_size_result = total_decoded_size;" in source
    assert "unsigned int decoded_window_offset" in source
    assert "decoded_window_offset < 0" not in source
    assert "decoded_offset < decoded_total" in source
    assert "decoded_total - decoded_offset : 0ull" in source
    assert "local_target_0 < decoded_remaining" in source
    assert "first_target" not in source
    wrapper = source[source.index(f"void {multidimensional.symbol_name}") :]
    assert "for (" not in wrapper
    assert "shfl" not in wrapper
    assert "atom." not in wrapper


def test_run_length_renderer_uses_negative_signed_relative_oob_sentinel():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int32, Int64

    from cuda.coop.cutlass import this_block as cutlass_this_block
    from cuda.coop.cutlass._dsl import _cub_run_length_decode_provider as provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    artifact = provider._make_request(
        group=cutlass_this_block(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        value_type=Int32,
        length_type=Int64,
        runs_per_thread=2,
        decoded_items_per_thread=3,
        with_relative_offsets=True,
        source="signed_relative_offset_test",
    )
    renderer = provider_support.bundle_renderer_for(artifact)
    assert renderer is not None
    source = "\n".join(renderer.render(artifact))
    assert "long long* relative_offsets_result" in source
    assert "static_cast<long long>(-1)" in source
    assert "static_cast<long long>(~0ull)" not in source


def test_root_and_scoped_run_length_requests_share_one_artifact():
    linear, _ = _requests()

    from cutlass.base_dsl.typing import Int32, Uint32

    from cuda.coop.cutlass import this_block as cutlass_this_block
    from cuda.coop.cutlass._dsl import _cub_run_length_decode_provider as provider

    scoped = provider._make_request(
        group=cutlass_this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        value_type=Int32,
        length_type=Uint32,
        runs_per_thread=2,
        decoded_items_per_thread=3,
        with_relative_offsets=True,
        source="scoped_block",
    )
    assert linear == scoped
    assert linear.symbol_name == scoped.symbol_name


def test_run_length_renderer_compiles_with_nvrtc(monkeypatch, tmp_path):
    pytest.importorskip("cuda.bindings.nvrtc")
    _, artifact = _requests()

    from cuda.coop.cutlass._dsl import _provider_bundle as provider_bundle
    from cuda.coop.cutlass._dsl.block import _provider as block_provider

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT", "ltoir")
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
