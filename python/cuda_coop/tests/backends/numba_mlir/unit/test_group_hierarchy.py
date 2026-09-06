# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop as portable_coop
import cuda.coop.numba_mlir as coop
from cuda.coop._core import LaunchFactOrigin, LaunchFacts, resolve_thread_group
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._group_planner import (
    CoopGroupHierarchyPlanner,
    GroupRewriteError,
    _GroupCallPlanner,
    has_group_markers,
)

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_GROUP_KINDS = (
    ("this_thread", "thread"),
    ("this_warp", "warp"),
    ("this_block", "block"),
    ("this_cluster", "cluster"),
    ("this_grid", "grid"),
)
_GLOBAL_BLOCK_GROUP = coop.this_block()
_ALTERNATE_GLOBAL_BLOCK_GROUP = coop.this_block()


def _state(function, args=()):
    return SimpleNamespace(func_ir=run_frontend(function), args=args)


def _launch(block=(64, 1, 1), grid=(1, 1, 1), cluster=None):
    return {"block": block, "grid": grid, "cluster": cluster}


def _compile_context(header_identity="headers-a"):
    return _nvrtc.CompileContext(
        toolkit_root="/toolkit",
        toolkit_version=(13, 2),
        nvrtc_path="/toolkit/lib/libnvrtc.so.13",
        nvrtc_builtins_path="/toolkit/lib/libnvrtc-builtins.so.13",
        nvjitlink_path="/toolkit/lib/libnvJitLink.so.13",
        nvrtc_version=_nvrtc.version(13, 2),
        nvjitlink_version=(13, 2),
        include_dirs=("/toolkit/include",),
        header_identity=header_identity,
    )


def _thread_group_lowering_module():
    """Import provider lowering only in tests that inspect it."""

    import cuda.coop.numba_mlir._lowering._thread_group as lowering

    return lowering


def _capture_group_method_provider(monkeypatch):
    lowering = _thread_group_lowering_module()
    created = []
    monkeypatch.setattr(lowering, "_current_cc", lambda: 90)
    monkeypatch.setattr(
        lowering,
        "RawCAbiInvocable",
        lambda **kwargs: created.append(kwargs) or SimpleNamespace(**kwargs),
    )
    return lowering, created


def _resolved_provider_group(group, *, through_level="thread", block=(64, 1, 1)):
    launch = LaunchFacts(
        exact_block_dim=block,
        exact_grid_dim=(2, 1, 1),
        cluster_launch=False,
        provenance=LaunchFactOrigin(
            fact="cluster_launch",
            source="test",
            verified=True,
        ),
    )
    return resolve_thread_group(
        group,
        launch,
        through_level=through_level,
    ).require_supported()


def _assigned_var(state, name):
    return next(
        inst.target
        for block in state.func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and inst.target.name == name
    )


def test_group_marker_detection_distinguishes_group_ir():
    def group_marker_function():
        return coop.this_block().count()

    def plain_function(value):
        return value + 1

    assert has_group_markers(run_frontend(group_marker_function))
    assert not has_group_markers(run_frontend(plain_function))


def test_planner_shares_compile_context_across_group_methods(monkeypatch):
    thread_group_lowering = _thread_group_lowering_module()
    captured = []
    context_resolutions = []
    context = _compile_context()

    def query():
        group = coop.this_block()
        return group.count("thread") + group.rank("thread")

    class _Invocable:
        def __call__(self):
            raise AssertionError("compile-time invocable executed by Python")

    def make_invocable(**kwargs):
        captured.append(kwargs)
        return _Invocable()

    monkeypatch.setattr(
        thread_group_lowering,
        "make_group_method_invocable",
        make_invocable,
    )
    monkeypatch.setattr(
        _nvrtc,
        "resolve_compile_context",
        lambda: context_resolutions.append(True) or context,
    )
    planner = _GroupCallPlanner(_state(query), _launch())

    assert planner.run()
    assert len(captured) == 2
    assert {item["operation"] for item in captured} == {"count", "rank"}
    assert all(item["level"] == "thread" for item in captured)
    assert all(item["group"].kind == "block" for item in captured)
    assert all(item["group"].group_thread_count == 64 for item in captured)
    assert all(item["compile_context"] is context for item in captured)
    assert context_resolutions == [True]


def test_planner_deduplicates_canonical_group_query_dtypes(monkeypatch):
    thread_group_lowering = _thread_group_lowering_module()
    captured = []

    def query():
        group = coop.this_block()
        return (
            group.rank()
            + group.rank_as(types.uint32)
            + group.rank_as("uint32")
            + group.rank_as(int)
            + group.rank_as(types.int32)
        )

    class _Invocable:
        def __call__(self):
            raise AssertionError("compile-time invocable executed by Python")

    monkeypatch.setattr(
        thread_group_lowering,
        "make_group_method_invocable",
        lambda **kwargs: captured.append(kwargs) or _Invocable(),
    )
    monkeypatch.setattr(_nvrtc, "resolve_compile_context", _compile_context)

    assert _GroupCallPlanner(_state(query), _launch()).run()
    assert [item["dtype"] for item in captured] == [types.uint32, types.int32]


def test_mapped_parent_queries_render_the_parent_group():
    thread_group_lowering = _thread_group_lowering_module()
    mapped = resolve_thread_group(
        coop.this_warp().group_by(8),
        LaunchFacts(exact_block_dim=64),
        through_level="warp",
    ).require_supported()

    assert thread_group_lowering._query_expr(mapped, "rank", "warp") == (
        "group.rank(group_parent)"
    )
    assert thread_group_lowering._query_expr(mapped, "count", "warp") == (
        "group.count(group_parent)"
    )
    assert thread_group_lowering._query_expr(mapped, "rank", "thread") == (
        "::cuda::gpu_thread.rank(group)"
    )


def test_mapped_queries_above_the_immediate_parent_fail_during_planning():
    def query():
        return coop.this_warp().group_by(8).count("block")

    with pytest.raises(NotImplementedError, match="immediate parent"):
        _GroupCallPlanner(_state(query), _launch()).run()


def test_block_warp_queries_reject_a_block_smaller_than_one_warp():
    def query():
        return coop.this_block().count("warp")

    with pytest.raises(
        NotImplementedError, match="at least one complete 32-thread Warp"
    ):
        _GroupCallPlanner(_state(query), _launch(block=(16, 1, 1))).run()


def test_thread_parent_warp_queries_allow_a_block_smaller_than_one_warp(monkeypatch):
    lowering = _thread_group_lowering_module()
    captured = []

    def query():
        return coop.this_thread().count("warp")

    class _Invocable:
        def __call__(self):
            raise AssertionError("compile-time invocable executed by Python")

    monkeypatch.setattr(
        lowering,
        "make_group_method_invocable",
        lambda **kwargs: captured.append(kwargs) or _Invocable(),
    )
    monkeypatch.setattr(_nvrtc, "resolve_compile_context", _compile_context)

    assert _GroupCallPlanner(_state(query), _launch(block=(16, 1, 1))).run()
    assert captured[0]["group"].kind == "thread"
    assert captured[0]["level"] == "warp"


@pytest.mark.parametrize("operation", ["sync", "sync_aligned"])
def test_grid_synchronization_fails_during_planning(operation):
    if operation == "sync":

        def grid_sync():
            coop.this_grid().sync()

    else:

        def grid_sync():
            coop.this_grid().sync_aligned()

    with pytest.raises(NotImplementedError, match="cooperative launch"):
        _GroupCallPlanner(_state(grid_sync), _launch()).run()


def _query_mapped_warp_sync():
    coop.this_block().group_by(2).sync()


def _query_mapped_warp_sync_aligned():
    coop.this_block().group_by(2).sync_aligned()


@pytest.mark.parametrize(
    "query",
    (_query_mapped_warp_sync, _query_mapped_warp_sync_aligned),
)
def test_mapped_warp_synchronization_requires_planner_owned_barriers(query):
    with pytest.raises(NotImplementedError, match="planner-owned barrier lifetime"):
        _GroupCallPlanner(_state(query), _launch()).run()


def _query_rank():
    return coop.this_block().rank()


def _query_count():
    return coop.this_block().count("block")


def _query_rank_as():
    return coop.this_block().rank_as(types.uint16)


def _query_count_as():
    return coop.this_block().count_as(types.int64, "grid")


def _query_is_member():
    return coop.this_block().is_member()


def _query_sync():
    coop.this_block().sync()


def _query_sync_aligned():
    coop.this_block().sync_aligned()


@pytest.mark.parametrize(
    ("query", "operation", "dtype", "level"),
    [
        (_query_rank, "rank", types.uint32, "thread"),
        (_query_count, "count", types.uint32, "block"),
        (_query_rank_as, "rank", types.uint16, "thread"),
        (_query_count_as, "count", types.int64, "grid"),
        (_query_is_member, "is_member", None, "thread"),
        (_query_sync, "sync", None, "thread"),
        (_query_sync_aligned, "sync_aligned", None, "thread"),
    ],
)
def test_group_method_call_shapes_reach_the_provider(
    monkeypatch,
    query,
    operation,
    dtype,
    level,
):
    thread_group_lowering = _thread_group_lowering_module()
    captured = []

    class _Invocable:
        def __call__(self):
            raise AssertionError("compile-time invocable executed by Python")

    monkeypatch.setattr(
        thread_group_lowering,
        "make_group_method_invocable",
        lambda **kwargs: captured.append(kwargs) or _Invocable(),
    )
    monkeypatch.setattr(
        _nvrtc,
        "resolve_compile_context",
        _compile_context,
    )

    assert _GroupCallPlanner(_state(query), _launch()).run()
    assert len(captured) == 1
    assert captured[0]["operation"] == operation
    assert captured[0]["dtype"] == dtype
    assert captured[0]["level"] == level


@pytest.mark.parametrize(
    ("constructor", "operation", "level", "expected_dtype"),
    [
        (coop.this_block, "rank", "thread", types.uint32),
        (coop.this_block, "count", "grid", types.uint64),
        (coop.this_grid, "rank", "thread", types.uint64),
    ],
)
def test_group_method_provider_chooses_the_default_query_dtype(
    monkeypatch,
    constructor,
    operation,
    level,
    expected_dtype,
):
    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(constructor(), through_level=level)

    result = lowering.make_group_method_invocable(
        group=group,
        operation=operation,
        level=level,
        compile_context=_compile_context(),
    )

    assert result.return_type is expected_dtype
    assert len(created) == 1
    assert created[0]["return_type"] is expected_dtype
    cpp_name = {
        types.uint32: "::cuda::std::uint32_t",
        types.uint64: "::cuda::std::uint64_t",
    }[expected_dtype]
    assert f'extern "C" __device__ {cpp_name}' in result.source
    assert f"return static_cast<{cpp_name}>" in result.source


@pytest.mark.parametrize(
    ("dtype", "expected_dtype", "cpp_name"),
    (
        (types.int8, types.int8, "::cuda::std::int8_t"),
        (types.uint8, types.uint8, "::cuda::std::uint8_t"),
        (types.int16, types.int16, "::cuda::std::int16_t"),
        (types.uint16, types.uint16, "::cuda::std::uint16_t"),
        (types.int32, types.int32, "::cuda::std::int32_t"),
        (types.uint32, types.uint32, "::cuda::std::uint32_t"),
        (types.int64, types.int64, "::cuda::std::int64_t"),
        (types.uint64, types.uint64, "::cuda::std::uint64_t"),
        (int, types.int32, "::cuda::std::int32_t"),
        ("uint16", types.uint16, "::cuda::std::uint16_t"),
    ),
)
def test_group_method_provider_accepts_explicit_integral_query_dtypes(
    monkeypatch,
    dtype,
    expected_dtype,
    cpp_name,
):
    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(coop.this_block())

    result = lowering.make_group_method_invocable(
        group=group,
        operation="count",
        dtype=dtype,
        compile_context=_compile_context(),
    )

    assert result.return_type is expected_dtype
    assert len(created) == 1
    assert f'extern "C" __device__ {cpp_name}' in result.source
    assert f"return static_cast<{cpp_name}>" in result.source


@pytest.mark.parametrize(
    "dtype",
    (
        bool,
        float,
        types.boolean,
        types.float16,
        types.float32,
        types.float64,
        types.complex64,
    ),
)
def test_group_method_provider_rejects_explicit_non_integral_query_dtypes(
    monkeypatch,
    dtype,
):
    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(coop.this_block())

    with pytest.raises(TypeError, match="query dtype must be one of"):
        lowering.make_group_method_invocable(
            group=group,
            operation="rank",
            dtype=dtype,
            compile_context=_compile_context(),
        )

    assert created == []


def test_mapped_warp_queries_and_membership_do_not_construct_barrier_group(
    monkeypatch,
):
    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(
        coop.this_block().group_by(3, exhaustive=False),
        through_level="block",
        block=(128, 1, 1),
    )
    context = _compile_context()

    rank = lowering.make_group_method_invocable(
        group=group,
        operation="rank",
        level="thread",
        compile_context=context,
    )
    count = lowering.make_group_method_invocable(
        group=group,
        operation="count",
        level="block",
        compile_context=context,
    )
    membership = lowering.make_group_method_invocable(
        group=group,
        operation="is_member",
        compile_context=context,
    )

    assert len(created) == 3
    assert "constexpr ::cuda::std::uint32_t group_warp_count = 3;" in rank.source
    assert "constexpr ::cuda::std::uint32_t grouped_warp_count = 3;" in rank.source
    assert "(group_warp_rank % group_warp_count) * 32" in rank.source
    assert "4 / group_warp_count" in count.source
    assert "group_warp_rank < grouped_warp_count ? 1u : 0u" in membership.source
    assert membership.return_type is types.uint8
    for source in (rank.source, count.source, membership.source):
        assert "::cuda::experimental::this_block group_parent{hierarchy};" in source
        assert "barrier_synchronizer" not in source
        assert "::cuda::experimental::group group{" not in source


@pytest.mark.parametrize("operation", ("sync", "sync_aligned"))
def test_mapped_warp_provider_rejects_synchronization(monkeypatch, operation):
    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(coop.this_block().group_by(2))

    with pytest.raises(NotImplementedError, match="planner-owned barrier lifetime"):
        lowering.make_group_method_invocable(
            group=group,
            operation=operation,
            compile_context=_compile_context(),
        )

    assert created == []


def test_group_method_provider_symbols_include_compile_context(monkeypatch):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    lowering, created = _capture_group_method_provider(monkeypatch)
    group = _resolved_provider_group(coop.this_block())
    context_a = _compile_context("headers-a")
    context_a_equal = _compile_context("headers-a")
    context_b = _compile_context("headers-b")

    assert context_a_equal == context_a
    assert context_a_equal is not context_a

    first = lowering.make_group_method_invocable(
        group=group,
        operation="count",
        compile_context=context_a,
    )
    equal_context = lowering.make_group_method_invocable(
        group=group,
        operation="count",
        compile_context=context_a_equal,
    )
    other_context = lowering.make_group_method_invocable(
        group=group,
        operation="count",
        compile_context=context_b,
    )

    assert equal_context is not first
    assert other_context is not first
    assert len(created) == 3
    assert created[0]["compile_context"] is context_a
    assert created[1]["compile_context"] is context_a_equal
    assert created[2]["compile_context"] is context_b
    assert created[0]["symbol"] == created[1]["symbol"]
    assert created[0]["symbol"] != created[2]["symbol"]
    assert all(item["symbol"] in item["source"] for item in created)
    assert all(item["parameters"] == () for item in created)
    assert all(item["abi_transforms"] == () for item in created)
    assert all(item["storage_abi"] is StorageABI.NONE for item in created)
    assert all(
        item["synchronization_scope"] is SynchronizationScope.NONE for item in created
    )


def test_group_marker_detection_does_not_semantically_resolve_group_by(monkeypatch):
    def grouped(count):
        return _GLOBAL_BLOCK_GROUP.group_by(count)

    monkeypatch.setattr(
        _GroupCallPlanner,
        "_group",
        lambda *_args: pytest.fail("marker detection resolved group semantics"),
    )

    assert has_group_markers(run_frontend(grouped))


def test_group_marker_detection_follows_merged_group_by_receivers(monkeypatch):
    def grouped(count, choose_alternate):
        if choose_alternate:
            parent = _ALTERNATE_GLOBAL_BLOCK_GROUP
        else:
            parent = _GLOBAL_BLOCK_GROUP
        return parent.group_by(count)

    monkeypatch.setattr(
        _GroupCallPlanner,
        "_group",
        lambda *_args: pytest.fail("marker detection resolved group semantics"),
    )

    assert has_group_markers(run_frontend(grouped))


@pytest.mark.parametrize("api", (portable_coop, coop), ids=("portable", "qualified"))
def test_device_helper_group_planner_defers_without_requesting_launch(api, monkeypatch):
    from cuda.coop.numba_mlir._compiler import _group_planner

    def device_helper(source):
        group = api.this_block()
        output = api.ThreadData(1, dtype="int32")
        return api.load(group, source, output)

    state = SimpleNamespace(
        func_ir=run_frontend(device_helper),
        args=(),
        metadata={"targetoptions": {"device": True}},
    )
    before = {label: tuple(block.body) for label, block in state.func_ir.blocks.items()}

    monkeypatch.setattr(
        _group_planner,
        "require_launch_config",
        lambda _state: pytest.fail(
            "device-function group planning requested kernel launch metadata"
        ),
    )

    assert not CoopGroupHierarchyPlanner(state).run()
    assert {
        label: tuple(block.body) for label, block in state.func_ir.blocks.items()
    } == before


@pytest.mark.parametrize("api", (portable_coop, coop), ids=("portable", "qualified"))
@pytest.mark.parametrize(("constructor_name", "kind"), _GROUP_KINDS)
def test_planner_recognizes_the_full_group_descriptor_vocabulary(
    api,
    constructor_name,
    kind,
):
    constructor = getattr(api, constructor_name)

    def describe_group():
        group = constructor()
        return group

    state = _state(describe_group)
    planner = _GroupCallPlanner(state, _launch())
    planned_group = planner._group(_assigned_var(state, "group"))

    assert planned_group is not None
    assert planned_group.kind == kind
    expected_source = "common_root" if api is portable_coop else "current"
    assert planned_group.source == expected_source


def test_group_constructor_recognition_uses_exact_callable_identity():
    def impostor_this_block():
        return coop.this_block()

    impostor_this_block.__module__ = coop.this_block.__module__
    impostor_this_block.__name__ = coop.this_block.__name__
    impostor_this_block.__cuda_coop_backend_member__ = "this_block"

    def describe_group():
        group = impostor_this_block()
        return group

    state = _state(describe_group)
    planner = _GroupCallPlanner(state, _launch())

    assert planner._group(_assigned_var(state, "group")) is None
    assert not has_group_markers(state.func_ir)


def test_planner_captures_exact_launch_facts_and_provenance():
    def plain_function():
        return None

    planner = _GroupCallPlanner(
        _state(plain_function),
        _launch(
            block=(8, 4, 2),
            grid=(6, 4, 2),
            cluster=(2, 2, 1),
        ),
    )

    assert planner.launch == LaunchFacts(
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=(6, 4, 2),
        exact_cluster_dim=(2, 2, 1),
        cluster_launch=True,
        cooperative_launch=False,
        provenance=tuple(
            LaunchFactOrigin(
                fact=fact,
                source="independent_frontend",
                verified=True,
            )
            for fact in (
                "exact_block_dim",
                "exact_grid_dim",
                "cluster_launch",
                "exact_cluster_dim",
            )
        ),
    )
    assert tuple(origin.fact for origin in planner.launch.provenance) == (
        "exact_block_dim",
        "exact_grid_dim",
        "cluster_launch",
        "exact_cluster_dim",
    )
    assert all(
        origin.source == "numba_cuda_mlir_launch_config" and origin.verified
        for origin in planner.launch.provenance
    )
    assert all(
        planner.launch.is_verified(fact)
        for fact in (
            "exact_block_dim",
            "exact_grid_dim",
            "cluster_launch",
            "exact_cluster_dim",
        )
    )
    assert not planner.launch.is_verified("cooperative_launch")


@pytest.mark.parametrize(("constructor_name", "kind"), _GROUP_KINDS)
def test_core_resolves_every_physical_group_from_exact_launch_facts(
    constructor_name,
    kind,
):
    launch = LaunchFacts(
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=(6, 4, 2),
        exact_cluster_dim=(2, 2, 1),
        cluster_launch=True,
        provenance=LaunchFactOrigin(
            fact="cluster_launch",
            source="test",
            verified=True,
        ),
    )
    group = getattr(coop, constructor_name)()
    resolved = resolve_thread_group(group, launch).require_supported()

    assert resolved.kind == kind
    if kind == "thread":
        assert resolved.hierarchy.implicit
        return
    assert resolved.hierarchy.block_dim == (8, 4, 2)
    if kind in {"cluster", "grid"}:
        assert resolved.hierarchy.cluster_dim == (2, 2, 1)
    if kind == "grid":
        assert resolved.hierarchy.grid_dim == (3, 2, 2)


@pytest.mark.parametrize(
    "constructor",
    (portable_coop.this_block, coop.this_block),
    ids=("portable", "qualified"),
)
def test_descriptor_values_cannot_escape_to_runtime(constructor):
    def escapes():
        return constructor()

    with pytest.raises(GroupRewriteError, match="would escape to runtime"):
        _GroupCallPlanner(_state(escapes), _launch()).run()
