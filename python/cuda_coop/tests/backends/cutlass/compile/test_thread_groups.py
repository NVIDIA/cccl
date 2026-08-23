# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pytest

import cuda.coop

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT
_COOP_SOURCE_PATH = str(SOURCE_ROOT / "cuda" / "coop")
cuda.coop.__path__ = [
    _COOP_SOURCE_PATH,
    *(entry for entry in cuda.coop.__path__ if entry != _COOP_SOURCE_PATH),
]


def _verified_grid_facts(*, exact_block_dim=64):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=exact_block_dim,
        exact_grid_dim=(8, 2, 1),
        exact_cluster_dim=(2, 1, 1),
        cooperative_launch=True,
        cluster_launch=True,
        provenance=(
            LaunchFactOrigin(
                "cooperative_launch",
                "test_launch_config",
                verified=True,
            ),
            LaunchFactOrigin(
                "cluster_launch",
                "test_launch_config",
                verified=True,
            ),
        ),
    )


def test_generic_finalizer_uses_registered_group_renderer():
    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _finalize as _provider_finalizer
    from cuda.coop.cutlass._compiler import _state as provider_state
    from cuda.coop.cutlass._lowering import _thread_group as _thread_group_provider

    request = _thread_group_provider._CudaxGroupRequest(
        group=coop.this_block(),
        op="sync",
    )
    source = _provider_finalizer._render_bundle_source([request])

    assert f"void {request.symbol_name}()" in source
    assert (
        "::cuda::experimental::this_block "
        "group{::cuda::experimental::implicit_hierarchy()};"
    ) in source
    assert provider_state._ensure_bundle_finalizer() is (
        _provider_finalizer._trace_finalize_hook
    )


def test_group_query_rendering_covers_static_hierarchy_and_mappings():
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import resolve_thread_group
    from cuda.coop.cutlass._lowering import _thread_group as provider

    facts = _verified_grid_facts()
    thread = resolve_thread_group(
        coop.this_thread(),
        facts,
        through_level="grid",
    ).require_supported()
    grid = resolve_thread_group(
        coop.this_grid(),
        facts,
    ).require_supported()
    mapped = resolve_thread_group(
        coop.this_warp().group_by(12, exhaustive=False),
        facts,
        through_level="warp",
    ).require_supported()

    thread_source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(
                group=thread,
                op="rank",
                level="grid",
                result_type=Int32,
            )
        )
    )
    grid_thread_source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(
                group=grid,
                op="count",
                level="thread",
                result_type=Int32,
            )
        )
    )
    grid_self_source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(
                group=grid,
                op="count",
                level="grid",
                result_type=Int32,
            )
        )
    )
    mapped_source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(
                group=mapped,
                op="rank",
                level="warp",
                result_type=Int32,
            )
        )
    )
    mapped_count_source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(
                group=mapped,
                op="count",
                level="warp",
                result_type=Int32,
            )
        )
    )

    assert "group.rank(::cuda::grid)" in thread_source
    assert "::cuda::gpu_thread.count(group)" in grid_thread_source
    assert "return static_cast<int>(1);" in grid_self_source
    assert "group.rank(group_parent)" in mapped_source
    assert "group.count(group_parent)" in mapped_count_source
    assert "group_by<12, false>" in mapped_source


@pytest.mark.parametrize("kind", ("thread", "warp", "block", "cluster", "grid"))
def test_current_physical_group_rendering_uses_implicit_hierarchy(kind):
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._lowering import _thread_group as provider

    group = getattr(coop, f"this_{kind}")()
    source = "\n".join(
        provider._render_cudax_group(
            provider._CudaxGroupRequest(group=group, op="is_member")
        )
    )

    assert (
        f"::cuda::experimental::this_{kind} "
        "group{::cuda::experimental::implicit_hierarchy()};"
    ) in source
    assert "group{}" not in source


def test_implicit_current_block_operations_allow_empty_launch_facts(monkeypatch):
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _thread_group
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _thread_group as provider

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", LaunchFacts)
    block = coop.this_block()

    assert (
        _thread_group._validate_sync_launch(
            block,
            feature="ThreadGroup.sync",
        )
        is block
    )
    assert (
        _thread_group._validate_query_launch(
            block,
            feature="ThreadGroup.rank",
            level="thread",
        )
        is block
    )
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        _thread_group._validate_query_launch(
            coop.this_warp(),
            feature="ThreadGroup.rank",
            level="thread",
        )
    for group in (coop.this_thread(), coop.this_block()):
        with pytest.raises(NotImplementedError, match="exact block dimensions"):
            _thread_group._validate_query_launch(
                group,
                feature="ThreadGroup.rank",
                level="warp",
            )
    for request in (
        provider._CudaxGroupRequest(group=block, op="sync"),
        provider._CudaxGroupRequest(
            group=block,
            op="rank",
            level="thread",
            result_type=Int32,
        ),
        provider._CudaxGroupRequest(group=block, op="is_member"),
    ):
        source = "\n".join(provider._render_cudax_group(request))
        assert (
            "::cuda::experimental::this_block "
            "group{::cuda::experimental::implicit_hierarchy()};"
        ) in source
        assert "::cuda::hierarchy" not in source

    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        lambda: _verified_grid_facts(exact_block_dim=48),
    )
    with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
        _thread_group._validate_query_launch(
            coop.this_warp(),
            feature="ThreadGroup.count",
            level="thread",
        )
    for group in (coop.this_thread(), coop.this_block(), coop.this_grid()):
        with pytest.raises(NotImplementedError, match="every physical warp"):
            _thread_group._validate_query_launch(
                group,
                feature="ThreadGroup.count",
                level="warp",
            )

    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        _verified_grid_facts,
    )
    for group in (coop.this_thread(), coop.this_block(), coop.this_grid()):
        resolved = _thread_group._validate_query_launch(
            group,
            feature="ThreadGroup.count",
            level="warp",
        )
        assert resolved.hierarchy.block_thread_count == 64


def test_empty_launch_facts_allow_current_physical_but_reject_mapped_groups(
    monkeypatch,
):
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _thread_group
    from cuda.coop.cutlass._compiler import _launch

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", LaunchFacts)

    for current_physical in (
        coop.this_thread(),
        coop.this_warp(),
        coop.this_block(),
        coop.this_cluster(),
        coop.this_grid(),
    ):
        assert (
            _thread_group._validate_membership_launch(
                current_physical,
                feature="ThreadGroup.is_member",
            )
            is current_physical
        )

    current = coop.this_block()
    assert (
        _thread_group._validate_sync_launch(
            current,
            feature="ThreadGroup.sync",
        )
        is current
    )
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        _thread_group._validate_membership_launch(
            coop.this_block().group_by(2),
            feature="ThreadGroup.is_member",
        )
