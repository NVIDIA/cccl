# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _prefix_from_aggregate(aggregate):
    return aggregate + 11


def _multiply(left, right):
    return left * right


class _RunningPrefix:
    def __call__(self_ptr, aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + aggregate
        return previous


def _plan(function, *, arg_types, block=(128, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


def _planned_factory_calls(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return [
        (globals_by_name.get(inst.value.func.name), inst.value)
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    ]


def _constant_assignments(func_ir, ir):
    return {
        inst.target.name: inst.value.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, (ir.Const, ir.Global))
    }


def _capture_block_scan_factory(monkeypatch, **kwargs):
    import importlib

    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    module = importlib.import_module("cuda.coop.numba_mlir._lowering._scan")

    def capture(self, specialization, **materialize_kwargs):
        del self, materialize_kwargs
        return specialization

    monkeypatch.setattr(module.NumbaMlirCoreAdapter, "materialize", capture)
    monkeypatch.setattr(
        module,
        "make_invocable_from_specialization",
        lambda specialization: specialization,
    )
    return module.scan(
        dtype=types.int32,
        threads_per_block=32,
        items_per_thread=2,
        **kwargs,
    )


@pytest.mark.parametrize("keyword", ["prefix_op", "block_prefix_callback_op"])
def test_block_scan_factory_lowers_prefix_callback_aliases(monkeypatch, keyword):
    from cuda.coop._core import PythonOperator

    specialization = _capture_block_scan_factory(
        monkeypatch,
        **{keyword: _prefix_from_aggregate},
    )

    prefix = specialization.parameters[0][-1]
    assert isinstance(prefix, PythonOperator)
    assert prefix.op is _prefix_from_aggregate
    assert prefix.name == "prefix_op"


def test_block_scan_factory_lowers_stateful_running_prefix(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop._core import StatefulOperator
    from cuda.coop.numba_mlir import StatefulFunction

    running_prefix = StatefulFunction(
        _RunningPrefix,
        types.int32,
        name="test_running_prefix",
    )
    specialization = _capture_block_scan_factory(
        monkeypatch,
        prefix_op=running_prefix,
    )

    prefix = specialization.parameters[0][-1]
    assert isinstance(prefix, StatefulOperator)
    assert prefix.op is running_prefix
    assert prefix.state_dtype == types.int32


@pytest.mark.parametrize(
    "keyword",
    ["prefix_op", "block_prefix_callback_op"],
)
def test_group_first_block_scan_plans_prefix_callback_aliases(keyword):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    if keyword == "prefix_op":

        def kernel(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            return coop.scan(
                coop.this_block(),
                items,
                prefix_op=_prefix_from_aggregate,
            )

    else:

        def kernel(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            return coop.scan(
                coop.this_block(),
                items,
                block_prefix_callback_op=_prefix_from_aggregate,
            )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    provider_call = next(
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.scan
    )
    constants = _constant_assignments(func_ir, ir)
    assert len(provider_call.args) == 2
    callback_ref = dict(provider_call.kws)[keyword]
    assert constants[callback_ref.name] is _prefix_from_aggregate


def test_group_first_block_scan_plans_stateful_prefix_state():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    running_prefix = coop.StatefulFunction(
        _RunningPrefix,
        types.int32,
        name="test_running_prefix",
    )

    def kernel(value):
        items = coop.ThreadData(1, dtype=types.int32)
        state = coop.ThreadData(1, dtype=types.int32)
        items[0] = value
        state[0] = 0
        return coop.exclusive_sum(
            coop.this_block(),
            items,
            state,
            prefix_op=running_prefix,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    provider_call = next(
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.scan
    )
    assert len(provider_call.args) == 3
    callback_ref = dict(provider_call.kws)["prefix_op"]
    assert planner._constant(callback_ref) is running_prefix


def test_block_scan_prefix_allows_custom_exclusive_scan_without_initial(
    monkeypatch,
):
    specialization = _capture_block_scan_factory(
        monkeypatch,
        scan_op=_multiply,
        prefix_op=_prefix_from_aggregate,
    )

    assert specialization.method_name == "ExclusiveScan"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "prefix_op": _prefix_from_aggregate,
                "block_prefix_callback_op": _prefix_from_aggregate,
            },
            "mutually exclusive",
        ),
        (
            {"prefix_op": _prefix_from_aggregate, "initial_value": 3},
            "initial_value",
        ),
        (
            {"prefix_op": _prefix_from_aggregate, "block_aggregate": True},
            "block_aggregate",
        ),
    ],
)
def test_block_scan_prefix_rejects_incompatible_overloads(
    monkeypatch,
    kwargs,
    match,
):
    with pytest.raises(ValueError, match=match):
        _capture_block_scan_factory(monkeypatch, **kwargs)


def test_stateful_block_scan_prefix_requires_runtime_state():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import StatefulFunction
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    running_prefix = StatefulFunction(
        _RunningPrefix,
        types.int32,
        name="test_running_prefix",
    )
    rewrite = object.__new__(CoopSinglePhaseRewrite)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="requires a third runtime state argument",
    ):
        rewrite._finalize_scan_factory_kwargs(
            runtime_arg_count=2,
            factory_kwargs={"prefix_op": running_prefix},
        )

    rewrite._finalize_scan_factory_kwargs(
        runtime_arg_count=3,
        factory_kwargs={"prefix_op": running_prefix},
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="accepts only input and output runtime arguments",
    ):
        rewrite._finalize_scan_factory_kwargs(
            runtime_arg_count=3,
            factory_kwargs={"prefix_op": _prefix_from_aggregate},
        )


@pytest.mark.parametrize(
    ("group_kind", "make_group"),
    [
        ("threads_within_warp", lambda coop: coop.this_warp().group_by(8)),
        ("warps_within_block", lambda coop: coop.this_block().group_by(1)),
    ],
)
def test_reduce_lowers_logical_and_mapped_groups_to_cudax(group_kind, make_group):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering._reduce import group_reduce

    group = make_group(coop)

    def kernel(value):
        return coop.sum(group, value)

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is group_reduce
    ]
    assert len(calls) == 1
    constants = _constant_assignments(func_ir, ir)
    resolved_group = constants[dict(calls[0].kws)["group"].name]
    assert resolved_group.kind == group_kind


def test_multiple_mapped_group_reductions_keep_distinct_descriptors():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering._reduce import group_reduce

    pairs = coop.this_block().group_by(2)
    singles = coop.this_block().group_by(1)

    def kernel(value):
        return coop.sum(pairs, value), coop.sum(singles, value)

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is group_reduce
    ]
    constants = _constant_assignments(func_ir, ir)
    groups = [constants[dict(call.kws)["group"].name] for call in calls]
    assert len(groups) == 2
    assert len({group.semantic_key for group in groups}) == 2
    assert all(group.kind == "warps_within_block" for group in groups)


@pytest.mark.parametrize("group_kind", ["block", "logical_warp"])
def test_direct_valid_prefix_reduce_selects_cub(group_kind):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    group = coop.this_block() if group_kind == "block" else coop.this_warp().group_by(8)

    def kernel(value, valid_items):
        return coop.sum(
            group,
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    func_ir, planner = _plan(
        kernel,
        arg_types=(types.int32, types.int32),
    )
    assert planner.run()
    expected = lowering.sum if group_kind == "block" else lowering.warp_sum
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is expected
    ]
    assert len(calls) == 1
    keyword = "num_valid" if group_kind == "block" else "valid_items"
    assert keyword in dict(calls[0].kws)


def test_thread_data_block_reduce_requires_explicit_direct_controls():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    def kernel(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.sum(
            coop.this_block(),
            items,
            broadcast=False,
            algorithm="raking",
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.sum
    ]
    assert len(calls) == 1
    constants = _constant_assignments(func_ir, ir)
    assert constants[dict(calls[0].kws)["algorithm"].name] == "raking"


def test_qualified_block_reduce_selects_a_custom_callback_provider():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    def combine(lhs, rhs):
        return lhs + rhs

    def kernel(value):
        return coop.reduce(
            coop.this_block(),
            value,
            binary_op=combine,
            broadcast=False,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.reduce
    ]
    assert len(calls) == 1


def test_common_root_rejects_qualified_scan_callbacks():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop

    def combine(lhs, rhs):
        return lhs + rhs

    def kernel(value):
        return coop.inclusive_scan(
            coop.this_warp(),
            value,
            scan_op=combine,
        )

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(ValueError, match="built-in operators only"):
        planner.run()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("broadcast", "broadcast must be a compile-time bool"),
        ("reduce_valid", "valid_items must be an integer"),
        ("scan_valid", "valid_items must be an integer"),
    ],
)
def test_static_bool_controls_are_rejected(case, match):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    if case == "broadcast":

        def kernel(value):
            return coop.sum(coop.this_block(), value, broadcast=1)

    elif case == "reduce_valid":

        def kernel(value):
            return coop.sum(
                coop.this_block(),
                value,
                broadcast=False,
                valid_items=True,
            )

    else:

        def kernel(value):
            return coop.inclusive_sum(
                coop.this_warp(),
                value,
                valid_items=True,
            )

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(TypeError, match=match):
        planner.run()


@pytest.mark.parametrize("control_type", ["boolean", "float32"])
@pytest.mark.parametrize("operation", ["reduce", "scan"])
def test_dynamic_partial_tile_controls_require_integer_type(control_type, operation):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def reduce_kernel(value, valid_items):
        return coop.sum(
            coop.this_warp().group_by(8),
            value,
            broadcast=False,
            valid_items=valid_items,
        )

    def scan_kernel(value, valid_items):
        return coop.inclusive_scan(
            coop.this_warp().group_by(8),
            value,
            valid_items=valid_items,
        )

    function = reduce_kernel if operation == "reduce" else scan_kernel
    arg_types = (types.int32, getattr(types, control_type))
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=SimpleNamespace(refresh=lambda: None),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    block = func_ir.blocks[sorted(func_ir.blocks)[0]]
    with pytest.raises(CoopSinglePhaseRewriteError, match="must be an integer"):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)


def test_block_scan_is_out_of_place_and_forwards_aggregate_and_storage():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    def kernel(value):
        items = coop.ThreadData(2, dtype=types.int32)
        aggregate = coop.ThreadData(1, dtype=types.int32)
        storage = coop.TempStorage(4096, alignment=16)
        items[0] = value
        items[1] = value + 1
        return coop.inclusive_sum(
            coop.this_block(),
            items,
            aggregate_output=aggregate,
            temp_storage=storage,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.scan
    ]
    assert len(calls) == 1
    assert calls[0].args[0].name != calls[0].args[1].name
    assert {name for name, _ in calls[0].kws} >= {
        "block_aggregate",
        "temp_storage",
    }


def test_scan_aggregate_accepts_initial_value():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    def kernel(value):
        aggregate = coop.ThreadData(1, dtype=types.int32)
        return coop.exclusive_scan(
            coop.this_warp(),
            value,
            initial_value=7,
            aggregate_output=aggregate,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    calls = [
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is lowering.warp_exclusive_scan
    ]
    assert len(calls) == 1
    assert {name for name, _ in calls[0].kws} >= {
        "initial_value",
        "warp_aggregate",
    }


def test_scan_aggregate_requires_one_item():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        aggregate = coop.ThreadData(2, dtype=types.int32)
        return coop.exclusive_scan(
            coop.this_warp(),
            value,
            aggregate_output=aggregate,
        )

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(ValueError, match="exactly one item"):
        planner.run()


@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_non_sum_exclusive_scan_requires_initial_value(group_kind):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    group = coop.this_block() if group_kind == "block" else coop.this_warp()

    def kernel(value):
        return coop.exclusive_scan(group, value, scan_op="max")

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(ValueError, match="initial_value"):
        planner.run()


def test_common_root_keeps_mapped_block_reduction_out_of_profile():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop

    def kernel(value):
        return coop.sum(coop.this_block().group_by(1), value)

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises((NotImplementedError, ValueError), match="warps_within_block"):
        planner.run()


def test_bool_algorithms_are_not_coerced_to_cub_enums():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering._reduce import sum as block_sum
    from cuda.coop.numba_mlir._lowering._scan import scan as block_scan

    with pytest.raises(TypeError, match="algorithm must not be bool"):
        block_sum(
            dtype=types.int32,
            threads_per_block=32,
            algorithm=True,
        )
    with pytest.raises(TypeError, match="algorithm must not be bool"):
        block_scan(
            dtype=types.int32,
            threads_per_block=32,
            algorithm=True,
        )


def test_fixed_capacity_temp_storage_is_planned_for_scan():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    class FakeInvocable:
        files = ("scan-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

    def kernel(value):
        items = coop.ThreadData(2, dtype=types.int32)
        aggregate = coop.ThreadData(1, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.inclusive_sum(
            coop.this_block(),
            items,
            aggregate_output=aggregate,
            temp_storage=coop.TempStorage(4096, alignment=16),
        )

    arg_types = (types.int32,)
    func_ir, planner = _plan(kernel, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=SimpleNamespace(refresh=lambda: None),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    invocable = FakeInvocable()
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    rewrite._materialize_invocable = lambda _match: (invocable, False)
    rewrite._record_invocable_specialization = lambda _invocable: None
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block

    calls = _planned_factory_calls(func_ir, ir)
    invocable_calls = [call for factory, call in calls if factory is invocable]
    assert len(invocable_calls) == 1
    assert len(invocable_calls[0].args) == 4
    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    shared_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        shared_calls.extend(
            inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and resolver._resolve_python_value(inst.value.func) is cuda.shared.array
        )
    assert len(shared_calls) == 1
    resolver._block_defs = {
        inst.target.name: inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
    }
    assert resolver._infer_constant(shared_calls[0].args[0]) == 4096
    alignment = dict(shared_calls[0].kws)["alignment"]
    assert resolver._infer_constant(alignment) == 16
