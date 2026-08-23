# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cuda.coop._core import (  # noqa: E402
    GroupHistogramSemantics,
    GroupLoweringTarget,
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
from cuda.coop._core.block import make_block_histogram_semantics  # noqa: E402


def _operation(*, bins_per_thread: int = 2) -> GroupHistogramSemantics:
    return GroupHistogramSemantics(
        make_block_histogram_semantics(
            item_dtype="u8",
            counter_dtype="i64",
            items_per_thread=3,
            bins=32,
            algorithm="atomic",
        ),
        bins_per_thread=bins_per_thread,
    )


def test_group_histogram_plan_records_static_cub_and_projection_contracts():
    call = make_group_primitive_call(this_block(), _operation(), source="root")
    plan = plan_group_primitive(
        call,
        LaunchFacts(exact_block_dim=(8, 4, 2)),
    ).require_supported()

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockHistogram"
    assert plan.implementation.method_name == "Histogram"
    assert plan.implementation.template_arguments == {
        "T": "u8",
        "BLOCK_DIM_X": 8,
        "ITEMS_PER_THREAD": 3,
        "BINS": 32,
        "ALGORITHM": "::cub::BLOCK_HISTO_ATOMIC",
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
    }
    assert plan.provenance.header == "cub/block/block_histogram.cuh"
    assert plan.provenance.cpp_class == "cub::BlockHistogram"
    assert [item.name for item in call.argument_classifications] == [
        "samples",
        "bins",
        "bins_per_thread",
        "algorithm",
    ]
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.primary.dtype == "i64"
    assert plan.result.result_items_per_thread == 2
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.participation.argument_preconditions[0].name == "samples"
    assert plan.participation.argument_preconditions[0].minimum == 0
    assert plan.participation.argument_preconditions[0].maximum == 31
    assert (
        plan.participation.argument_preconditions[0].enforcement
        is PreconditionEnforcement.CALLER
    )


def test_group_histogram_requires_static_bins_and_exact_block_group():
    with pytest.raises(ValueError, match="static bin count"):
        GroupHistogramSemantics(
            make_block_histogram_semantics(
                item_dtype="u8",
                counter_dtype="i32",
                items_per_thread=1,
                bins=None,
            ),
            bins_per_thread=1,
        )

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


def _requests():
    pytest.importorskip("cutlass")
    from cutlass.base_dsl.typing import Int64, Uint8

    from cuda.coop.cutlass import this_block as cutlass_this_block
    from cuda.coop.cutlass._dsl import _cub_histogram_provider as provider

    common = {
        "sample_type": Uint8,
        "counter_type": Int64,
        "items_per_thread": 2,
        "bins": 32,
        "bins_per_thread": 2,
        "algorithm": "atomic",
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


def test_histogram_renderer_uses_one_public_cub_call_and_shared_projection():
    linear, multidimensional = _requests()

    from cuda.coop.cutlass._dsl import _cub_histogram_provider as provider
    from cuda.coop.cutlass._dsl import _provider as provider_support

    renderer = provider_support.bundle_renderer_for(multidimensional)
    assert renderer is not None
    assert renderer.render is provider.render_histogram_artifact
    source = "\n".join(renderer.render(multidimensional))
    assert linear.semantic_key != multidimensional.semantic_key
    assert "_b64_" in linear.symbol_name
    assert "_b8x4x2_" in multidimensional.symbol_name
    assert (
        "::cub::BlockHistogram<unsigned char, 8, 2, 32, "
        "::cub::BLOCK_HISTO_ATOMIC, 4, 2>" in source
    )
    assert "__shared__ unsigned long long histogram[32];" in source
    assert source.count("implementation_type(storage).Histogram(") == 1
    assert "cuda_coop_cutlass_block_sync();" in source
    assert "histogram_result[0]" in source
    assert "histogram_result[1]" in source
    assert "for (" not in source
    assert "shfl" not in source
    assert "cuda_coop_cutlass_atomic_add_shared" not in source


def test_root_and_scoped_histogram_requests_share_one_artifact():
    linear, _ = _requests()

    from cutlass.base_dsl.typing import Int64, Uint8

    from cuda.coop.cutlass import this_block as cutlass_this_block
    from cuda.coop.cutlass._dsl import _cub_histogram_provider as provider

    scoped = provider._make_request(
        group=cutlass_this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        sample_type=Uint8,
        counter_type=Int64,
        items_per_thread=2,
        bins=32,
        bins_per_thread=2,
        algorithm="atomic",
        source="scoped_block",
    )
    assert linear == scoped
    assert linear.semantic_key == scoped.semantic_key
    assert linear.symbol_name == scoped.symbol_name


def test_histogram_renderer_compiles_with_nvrtc(monkeypatch, tmp_path):
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
