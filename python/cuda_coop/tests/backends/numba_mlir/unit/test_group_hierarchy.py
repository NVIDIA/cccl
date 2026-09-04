# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop as portable_coop
import cuda.coop.numba_mlir as coop
from cuda.coop._core import LaunchFactOrigin, LaunchFacts, resolve_thread_group
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


def _assigned_var(state, name):
    return next(
        inst.target
        for block in state.func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and inst.target.name == name
    )


def test_group_marker_detection_distinguishes_group_ir():
    def group_marker_function():
        return coop.this_block()

    def plain_function(value):
        return value + 1

    assert has_group_markers(run_frontend(group_marker_function))
    assert not has_group_markers(run_frontend(plain_function))


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
