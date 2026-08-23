# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest

_DIRECT_CUB_BROADCAST_DIAGNOSTIC = (
    "direct CUB reduce returns a defined value only at the group root; "
    "it cannot satisfy broadcast=True"
)
_DIRECT_CUB_GROUP_DIAGNOSTIC = (
    "valid_items and explicit CUB algorithms are supported only for physical "
    "block, physical-warp, and logical-warp groups"
)
_DIRECT_CUB_WARP_ALGORITHM_DIAGNOSTIC = (
    "CUB algorithm selection applies to BlockReduce, not WarpReduce"
)
_EXCLUSIVE_SCAN_INITIAL_DIAGNOSTIC = (
    "cuda.coop.numba_mlir.scan requires initial_value for non-default exclusive scans"
)


def _multiply(lhs, rhs):
    return lhs * rhs


def _planned_factories(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return Counter(
        globals_by_name.get(inst.value.func.name)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )


@pytest.mark.parametrize(
    "operation",
    ["reduce", "inclusive_scan", "exchange", "load", "store"],
)
def test_numba_cub_scope_rejects_non_power_of_two_logical_warp(
    optional_backend,
    operation,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    group = numba_coop.this_warp().group_by(12, exhaustive=False)
    planner = object.__new__(_GroupCallPlanner)
    planner.launch = LaunchFacts(exact_block_dim=64)
    resolved_group = planner._resolve_group(group, feature=operation)

    assert resolved_group.hierarchy.block_dim == (64, 1, 1)
    assert resolved_group.source == "launch_facts"

    with pytest.raises(ValueError, match="power-of-two group width"):
        planner._scope_factory(resolved_group, operation)


@pytest.mark.evidence_for("group.reduce", backend="numba_mlir", evidence="lowering")
@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="lowering")
@pytest.mark.evidence_for("group.scan", backend="numba_mlir", evidence="lowering")
@pytest.mark.evidence_for(
    "group.exclusive_sum", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.inclusive_sum", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.evidence_for(
    "group.inclusive_scan", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_common_root_sum_scan_lower_to_numba_providers(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._block as scoped_block
    import cuda.coop.numba_mlir._warp as scoped_warp
    from cuda import coop
    from cuda.coop.numba_mlir._group_provider import group_reduce
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if group_kind == "block":

        def cohort(value):
            group = coop.this_block()
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value + 1
            total = coop.sum(group, items)
            maximum = coop.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                algorithm="raking",
            )
            default_scan = coop.scan(group, items)
            exclusive = coop.exclusive_sum(group, items)
            inclusive = coop.inclusive_sum(group, items)
            exclusive_alias = coop.exclusive_scan(group, items)
            inclusive_alias = coop.inclusive_scan(group, items)
            exclusive_max = coop.exclusive_scan(
                group,
                items,
                scan_op="max",
                initial_value=0,
            )
            inclusive_max = coop.inclusive_scan(group, items, scan_op="max")
            return (
                total,
                maximum,
                default_scan,
                exclusive,
                inclusive,
                exclusive_alias,
                inclusive_alias,
                exclusive_max,
                inclusive_max,
            )

    else:
        group_spec = (
            coop.this_warp().group_by(8)
            if group_kind == "logical_warp"
            else coop.this_warp()
        )
        valid_items = 7 if group_kind == "logical_warp" else 27

        def cohort(value):
            group = group_spec
            total = coop.sum(group, value)
            maximum = coop.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                valid_items=valid_items,
            )
            default_scan = coop.scan(group, value)
            exclusive = coop.exclusive_sum(group, value)
            inclusive = coop.inclusive_sum(group, value)
            exclusive_alias = coop.exclusive_scan(group, value)
            inclusive_alias = coop.inclusive_scan(group, value)
            exclusive_max = coop.exclusive_scan(
                group,
                value,
                scan_op="max",
                initial_value=0,
            )
            inclusive_max = coop.inclusive_scan(group, value, scan_op="max")
            return (
                total,
                maximum,
                default_scan,
                exclusive,
                inclusive,
                exclusive_alias,
                inclusive_alias,
                exclusive_max,
                inclusive_max,
            )

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    factories = Counter(
        globals_by_name.get(inst.value.func.name)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )

    assert factories[group_reduce] == 1
    if group_kind == "block":
        from cuda.coop.numba_mlir._block._block_reduce import block_reduce_builtin

        assert factories[block_reduce_builtin] == 1
        assert factories[scoped_block.scan] == 7
    else:
        from cuda.coop.numba_mlir._warp._warp_reduce import warp_reduce_builtin

        assert factories[warp_reduce_builtin] == 1
        assert factories[scoped_warp.warp_exclusive_sum] == 3
        assert factories[scoped_warp.warp_inclusive_sum] == 2
        assert factories[scoped_warp.warp_exclusive_scan] == 1
        assert factories[scoped_warp.warp_inclusive_scan] == 1


def test_qualified_logical_warp_partial_scan_lowers_aggregate_output(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._warp import _warp_scan

    def cohort(value, valid_items):
        aggregate = cuda.local.array(1, types.int32)
        result = numba_coop.scan(
            numba_coop.this_warp().group_by(8),
            value,
            mode="inclusive",
            valid_items=valid_items,
            aggregate_output=aggregate,
        )
        return result, aggregate[0]

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32, types.int32))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)
    assert _planned_factories(func_ir, ir)[_warp_scan.warp_inclusive_scan] == 1

    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    constants_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
    }
    calls = [
        inst.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
        and globals_by_name.get(inst.value.func.name) is _warp_scan.warp_inclusive_scan
    ]
    assert len(calls) == 1
    assert tuple(name for name, _ in calls[0].kws) == (
        "threads_in_warp",
        "threads_per_block",
        "scan_op",
        "valid_items",
        "warp_aggregate",
    )
    width_var = dict(calls[0].kws)["threads_in_warp"]
    assert constants_by_name[width_var.name] == 8


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_group_first_scan_accepts_shared_sum_and_multiply_spellings(
    optional_backend,
    qualified,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    import cuda.coop.numba_mlir._warp as scoped_warp
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop
    constructor = api.this_block if group_kind == "block" else api.this_warp

    if qualified:

        def scans(value):
            group = constructor()
            return (
                api.scan(group, value, mode="exclusive", scan_op="+"),
                api.exclusive_scan(group, value, scan_op="sum"),
                api.exclusive_scan(
                    group,
                    value,
                    scan_op="multiply",
                    initial_value=1,
                ),
                api.exclusive_scan(
                    group,
                    value,
                    scan_op=_multiply,
                    initial_value=1,
                ),
            )

    else:

        def scans(value):
            group = constructor()
            return (
                api.scan(group, value, mode="exclusive", scan_op="+"),
                api.exclusive_scan(group, value, scan_op="sum"),
                api.exclusive_scan(
                    group,
                    value,
                    scan_op="multiply",
                    initial_value=1,
                ),
            )

    func_ir = run_frontend(scans)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()

    factories = _planned_factories(func_ir, ir)
    if group_kind == "block":
        assert factories[scoped_block.scan] == (4 if qualified else 3)
    else:
        assert factories[scoped_warp.warp_exclusive_sum] == 2
        assert factories[scoped_warp.warp_exclusive_scan] == (2 if qualified else 1)


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize("group_kind", ["block", "warp"])
@pytest.mark.parametrize("operation_name", ["scan", "exclusive_scan"])
@pytest.mark.parametrize(
    "scan_op",
    ["multiply", _multiply],
    ids=["built_in", "callback"],
)
def test_group_first_non_sum_exclusive_scan_requires_initial_value(
    optional_backend,
    qualified,
    group_kind,
    operation_name,
    scan_op,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop
    constructor = api.this_block if group_kind == "block" else api.this_warp
    operation = getattr(api, operation_name)

    if operation_name == "scan":

        def invalid_scan(value):
            return operation(
                constructor(),
                value,
                mode="exclusive",
                scan_op=scan_op,
            )

    else:

        def invalid_scan(value):
            return operation(constructor(), value, scan_op=scan_op)

    state = SimpleNamespace(func_ir=run_frontend(invalid_scan), args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    with pytest.raises(ValueError) as error_info:
        planner.run()

    assert str(error_info.value) == _EXCLUSIVE_SCAN_INITIAL_DIAGNOSTIC


@pytest.mark.parametrize(
    "operation_name",
    [
        "scan",
        "exclusive_sum",
        "inclusive_sum",
        "exclusive_scan",
        "inclusive_scan",
    ],
)
def test_common_block_scan_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
    operation_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    common_operation = getattr(coop, operation_name)
    qualified_operation = getattr(numba_coop, operation_name)

    def common(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return common_operation(coop.this_block(), items)

    common_state = SimpleNamespace(
        func_ir=run_frontend(common),
        args=(types.int32,),
    )
    common_planner = _GroupCallPlanner(
        common_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    with pytest.raises(
        TypeError,
        match=(
            rf"cuda\.coop\.{operation_name} accepts only a scalar or "
            r"fixed-size ThreadData value payload in common V1"
        ),
    ):
        common_planner.run()

    def qualified(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return qualified_operation(numba_coop.this_block(), items)

    qualified_state = SimpleNamespace(
        func_ir=run_frontend(qualified),
        args=(types.int32,),
    )
    qualified_planner = _GroupCallPlanner(
        qualified_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    assert qualified_planner.run()


@pytest.mark.parametrize("operation_name", ["reduce", "sum"])
def test_common_reduction_rejects_local_arrays_but_qualified_keeps_them(
    optional_backend,
    operation_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    common_operation = getattr(coop, operation_name)
    qualified_operation = getattr(numba_coop, operation_name)

    def common(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return common_operation(coop.this_block(), items)

    common_state = SimpleNamespace(
        func_ir=run_frontend(common),
        args=(types.int32,),
    )
    common_planner = _GroupCallPlanner(
        common_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    with pytest.raises(
        TypeError,
        match=(
            rf"cuda\.coop\.{operation_name} accepts only a scalar or "
            r"fixed-size ThreadData value payload in common V1"
        ),
    ):
        common_planner.run()

    def qualified(value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        return qualified_operation(numba_coop.this_block(), items)

    qualified_state = SimpleNamespace(
        func_ir=run_frontend(qualified),
        args=(types.int32,),
    )
    qualified_planner = _GroupCallPlanner(
        qualified_state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    assert qualified_planner.run()


@pytest.mark.parametrize(
    ("op_name", "common_operation"),
    [
        ("warp_exclusive_sum", "exclusive_scan"),
        ("warp_inclusive_sum", "inclusive_scan"),
    ],
)
def test_physical_warp_default_scan_aliases_match_sum_factories(
    optional_backend,
    op_name,
    common_operation,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    factory_kwargs = {
        "_common_profile_operation": common_operation,
        "dtype": types.int32,
    }

    assert rewrite._extract_group_root_match_metadata(
        op_name=op_name,
        runtime_args=(object(),),
        factory_kwargs=factory_kwargs,
    ) == (False, False, False)
    assert "_common_profile_operation" not in factory_kwargs


@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize(
    "group_kind",
    ["thread", "physical_warp", "threads_within_warp", "block", "cluster"],
)
def test_common_root_sum_lowers_for_every_certified_group(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop.numba_mlir._group_provider import group_reduce
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if group_kind == "thread":

        def reduction(value):
            return coop.sum(coop.this_thread(), value)

    elif group_kind == "physical_warp":

        def reduction(value):
            return coop.sum(coop.this_warp(), value)

    elif group_kind == "threads_within_warp":

        def reduction(value):
            return coop.sum(coop.this_warp().group_by(8), value)

    elif group_kind == "block":

        def reduction(value):
            return coop.sum(coop.this_block(), value)

    else:

        def reduction(value):
            return coop.sum(coop.this_cluster(), value)

    func_ir = run_frontend(reduction)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (2, 1, 1), "cluster": (2, 1, 1)},
    ).run()
    assert not has_group_markers(func_ir)

    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    factories = [
        globals_by_name.get(inst.value.func.name)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    ]
    assert factories.count(group_reduce) == 1


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize(
    ("group_kind", "selector"),
    [("block", "valid_items"), ("block", "algorithm"), ("warp", "valid_items")],
)
def test_direct_cub_reduce_selectors_require_root_only_before_validation(
    optional_backend,
    qualified,
    group_kind,
    selector,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop
    constructor = api.this_block if group_kind == "block" else api.this_warp
    if selector == "valid_items":

        def reduction(value):
            return api.sum(constructor(), value, valid_items=0)

    else:

        def reduction(value):
            return api.sum(constructor(), value, algorithm="raking")

    state = SimpleNamespace(func_ir=run_frontend(reduction), args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )

    with pytest.raises(NotImplementedError) as error_info:
        planner.run()

    assert str(error_info.value) == _DIRECT_CUB_BROADCAST_DIAGNOSTIC


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize("selector", ["valid_items", "algorithm"])
@pytest.mark.parametrize(
    "group_kind",
    ["thread", "cluster"],
)
def test_direct_cub_reduce_selectors_require_physical_collective_group(
    optional_backend,
    qualified,
    selector,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop
    group = {
        "thread": api.this_thread(),
        "cluster": api.this_cluster(),
    }[group_kind]
    if selector == "valid_items":

        def reduction(value):
            return api.sum(group, value, valid_items=0)

    else:

        def reduction(value):
            return api.sum(group, value, algorithm="raking")

    state = SimpleNamespace(func_ir=run_frontend(reduction), args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {
            "block": (64, 1, 1),
            "grid": (2, 1, 1),
            "cluster": (2, 1, 1),
        },
    )

    with pytest.raises(NotImplementedError) as error_info:
        planner.run()

    assert str(error_info.value) == _DIRECT_CUB_GROUP_DIAGNOSTIC


@pytest.mark.parametrize("selector", ["valid_items", "algorithm"])
def test_qualified_direct_cub_reduce_selectors_reject_mapped_block_group_first(
    optional_backend,
    selector,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    group = numba_coop.this_block().group_by(1)
    if selector == "valid_items":

        def reduction(value):
            return numba_coop.sum(group, value, valid_items=0)

    else:

        def reduction(value):
            return numba_coop.sum(group, value, algorithm="raking")

    state = SimpleNamespace(func_ir=run_frontend(reduction), args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )

    with pytest.raises(NotImplementedError) as error_info:
        planner.run()

    assert str(error_info.value) == _DIRECT_CUB_GROUP_DIAGNOSTIC


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize("logical", [False, True], ids=["physical", "logical"])
def test_direct_cub_reduce_rejects_block_algorithm_on_warp_groups(
    optional_backend,
    qualified,
    logical,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop

    if logical:

        def reduction(value):
            return api.sum(
                api.this_warp().group_by(8),
                value,
                broadcast=False,
                algorithm="raking",
            )

    else:

        def reduction(value):
            return api.sum(
                api.this_warp(),
                value,
                broadcast=False,
                algorithm="raking",
            )

    state = SimpleNamespace(func_ir=run_frontend(reduction), args=(types.int32,))
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )

    with pytest.raises(NotImplementedError) as error_info:
        planner.run()

    assert str(error_info.value) == _DIRECT_CUB_WARP_ALGORITHM_DIAGNOSTIC


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_direct_cub_reduce_selectors_lower_without_broadcast_composition(
    optional_backend,
    qualified,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    import cuda.coop.numba_mlir._warp as scoped_warp
    from cuda import coop
    from cuda.coop.numba_mlir._block._block_reduce import block_reduce_builtin
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    api = numba_coop if qualified else coop
    if group_kind == "block":

        def reduction(value):
            group = api.this_block()
            partial = api.sum(
                group,
                value,
                broadcast=False,
                valid_items=31,
            )
            maximum = api.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                algorithm="raking",
            )
            return partial, maximum

    elif group_kind == "warp":

        def reduction(value):
            group = api.this_warp()
            partial = api.sum(
                group,
                value,
                broadcast=False,
                valid_items=23,
            )
            maximum = api.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                valid_items=27,
            )
            return partial, maximum

    else:

        def reduction(value):
            group = api.this_warp().group_by(8)
            partial = api.sum(
                group,
                value,
                broadcast=False,
                valid_items=7,
            )
            maximum = api.reduce(
                group,
                value,
                binary_op="max",
                broadcast=False,
                valid_items=5,
            )
            return partial, maximum

    func_ir = run_frontend(reduction)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    factories = _planned_factories(func_ir, ir)
    if group_kind == "block":
        assert factories[scoped_block.sum] == 1
        assert factories[block_reduce_builtin] == 1
    else:
        from cuda.coop.numba_mlir._warp._warp_reduce import warp_reduce_builtin

        assert factories[scoped_warp.warp_sum] == 1
        assert factories[warp_reduce_builtin] == 1
