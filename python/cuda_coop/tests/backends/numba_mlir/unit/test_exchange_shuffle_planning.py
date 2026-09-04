# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from inspect import signature
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, *, arg_types=(), block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


def _planned_factory_calls(func_ir):
    from numba_cuda_mlir.numbair_transforms import ir

    globals_by_name = {
        statement.target.name: statement.value.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign) and isinstance(statement.value, ir.Global)
    }
    return [
        (globals_by_name.get(statement.value.func.name), statement.value)
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign)
        and isinstance(statement.value, ir.Expr)
        and statement.value.op == "call"
    ]


def _provider_call(func_ir, provider):
    calls = [
        call for target, call in _planned_factory_calls(func_ir) if target is provider
    ]
    assert len(calls) == 1
    return calls[0]


def _match_before_inference(func_ir, *, arg_types):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    class _Invocable:
        files = ("exchange-shuffle-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16
        storage_abi = "leading_pointer"
        execution_scope = "block"
        synchronization_scope = "block"

        def __call__(self, *args):
            del args

    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typemap={},
        calltypes={},
        metadata={},
        typingctx=SimpleNamespace(refresh=lambda: None),
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    rewrite._materialize_invocable = lambda _match: (_Invocable(), False)
    matched = False
    for label in sorted(func_ir.blocks):
        matched |= rewrite.match(
            func_ir,
            func_ir.blocks[label],
            state.typemap,
            state.calltypes,
        )
    assert matched


def test_exchange_and_shuffle_register_declarative_result_and_rewrite_contracts():
    from cuda.coop.numba_mlir._compiler import _group_exchange, _group_shuffle
    from cuda.coop.numba_mlir._compiler._operations import (
        GroupResultSource,
        group_primitive,
        rewrite_operation,
    )

    del _group_exchange, _group_shuffle
    assert group_primitive("exchange").results == (GroupResultSource("value", "value"),)
    assert group_primitive("shuffle").results == (GroupResultSource("value", "value"),)
    assert rewrite_operation("exchange").runtime_arg_counts == frozenset({2})
    assert rewrite_operation("exchange_ranked").runtime_arg_counts == frozenset({3})
    assert rewrite_operation("exchange_flagged").runtime_arg_counts == frozenset({4})
    assert rewrite_operation("shuffle_scalar").runtime_arg_counts == frozenset({1, 2})
    assert rewrite_operation("shuffle_array").runtime_arg_counts == frozenset({2})


def test_public_shuffle_markers_do_not_advertise_boundary_outputs():
    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable

    assert tuple(signature(qualified.shuffle).parameters) == (
        "group",
        "value",
        "mode",
        "distance",
    )
    assert tuple(signature(portable.shuffle).parameters) == (
        "group",
        "value",
        "mode",
        "distance",
    )


@pytest.mark.parametrize(
    ("group_kind", "mode", "provider_name"),
    [
        ("block", "blocked_to_striped", "exchange"),
        ("warp", "blocked_to_striped", "warp_exchange"),
        ("logical_warp", "blocked_to_striped", "warp_exchange"),
        ("block", "scatter_to_blocked", "exchange_ranked"),
        ("warp", "scatter_to_striped", "warp_exchange_ranked"),
        ("logical_warp", "scatter_to_striped", "warp_exchange_ranked"),
        ("block", "scatter_to_striped_flagged", "exchange_flagged"),
    ],
)
def test_exchange_selects_fixed_arity_provider(
    group_kind,
    mode,
    provider_name,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _exchange

    uses_ranks = mode.startswith("scatter_to_")
    uses_flags = mode == "scatter_to_striped_flagged"
    group = {
        "block": coop.this_block(),
        "warp": coop.this_warp(),
        "logical_warp": coop.this_warp().group_by(8),
    }[group_kind]

    if uses_flags:

        def exchange(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value
            ranks = coop.ThreadData(2, dtype=types.int32)
            ranks[0] = 0
            ranks[1] = 1
            flags = coop.ThreadData(2, dtype=types.uint8)
            flags[0] = 1
            flags[1] = 1
            return coop.exchange(
                group,
                items,
                mode=mode,
                ranks=ranks,
                valid_flags=flags,
            )

    elif uses_ranks:

        def exchange(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value
            ranks = coop.ThreadData(2, dtype=types.int32)
            ranks[0] = 0
            ranks[1] = 1
            return coop.exchange(group, items, mode=mode, ranks=ranks)

    else:

        def exchange(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value
            return coop.exchange(group, items, mode=mode)

    func_ir, planner = _plan(exchange, arg_types=(types.int32,))
    assert planner.run()
    provider = getattr(_exchange, provider_name)
    call = _provider_call(func_ir, provider)
    assert len(call.args) == 2 + int(uses_ranks) + int(uses_flags)


def test_warp_exchange_copies_scatter_ranks_before_provider_call():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _exchange

    def exchange(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        ranks = coop.ThreadData(2, dtype=types.int32)
        ranks[0] = 0
        ranks[1] = 1
        return coop.exchange(
            coop.this_warp().group_by(8),
            items,
            mode="scatter_to_striped",
            ranks=ranks,
        )

    func_ir, planner = _plan(exchange, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _exchange.warp_exchange_ranked)
    copied_ranks = call.args[2]
    assert "preserved_ranks" in copied_ranks.name
    writes = [
        statement
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.SetItem)
        and statement.target.name == copied_ranks.name
    ]
    assert len(writes) == 2


@pytest.mark.parametrize(
    ("rank_dtype", "items_per_thread", "error"),
    [
        ("boolean", 2, "signed integer dtype"),
        ("float32", 2, "signed integer dtype"),
        ("uint32", 2, "signed integer dtype"),
        ("int32", 3, "same items_per_thread extent"),
    ],
)
def test_exchange_rejects_invalid_ranks_before_provider(
    monkeypatch,
    rank_dtype,
    items_per_thread,
    error,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_exchange

    rank_type = getattr(types, rank_dtype)

    def exchange(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        ranks = coop.ThreadData(
            items_per_thread,
            dtype=rank_type,
        )
        return coop.exchange(
            coop.this_block(),
            items,
            mode="scatter_to_blocked",
            ranks=ranks,
        )

    monkeypatch.setattr(
        _group_exchange._ExchangePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid ranks reached provider selection"
        ),
    )
    _, planner = _plan(exchange, arg_types=(types.int32,))
    with pytest.raises((TypeError, ValueError), match=error):
        planner.run()


@pytest.mark.parametrize(
    ("flag_dtype", "items_per_thread", "error"),
    [
        ("boolean", 2, "integral non-bool dtype"),
        ("float32", 2, "integral non-bool dtype"),
        ("uint8", 3, "same items_per_thread extent"),
    ],
)
def test_exchange_rejects_invalid_flags_before_provider(
    monkeypatch,
    flag_dtype,
    items_per_thread,
    error,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_exchange

    flag_type = getattr(types, flag_dtype)

    def exchange(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        ranks = coop.ThreadData(2, dtype=types.int32)
        ranks[0] = 0
        ranks[1] = 1
        flags = coop.ThreadData(items_per_thread, dtype=flag_type)
        return coop.exchange(
            coop.this_block(),
            items,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        )

    monkeypatch.setattr(
        _group_exchange._ExchangePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid valid_flags reached provider selection"
        ),
    )
    _, planner = _plan(exchange, arg_types=(types.int32,))
    with pytest.raises((TypeError, ValueError), match=error):
        planner.run()


@pytest.mark.parametrize(
    ("kind", "mode", "provider_name", "runtime_distance"),
    [
        ("scalar", "offset", "shuffle_scalar", True),
        ("scalar", "rotate", "shuffle_scalar", False),
        ("array", "up", "shuffle_array", False),
        ("array", "down", "shuffle_array", False),
    ],
)
def test_shuffle_selects_scalar_or_array_provider(
    kind,
    mode,
    provider_name,
    runtime_distance,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _shuffle

    if kind == "scalar":
        if runtime_distance:

            def shuffle(value, distance):
                return coop.shuffle(
                    coop.this_block(),
                    value,
                    mode=mode,
                    distance=distance,
                )

        else:

            def shuffle(value, distance):
                del distance
                return coop.shuffle(
                    coop.this_block(),
                    value,
                    mode=mode,
                    distance=3,
                )

    else:

        def shuffle(value, distance):
            del distance
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value
            return coop.shuffle(coop.this_block(), items, mode=mode)

    func_ir, planner = _plan(
        shuffle,
        arg_types=(types.int32, types.int32),
    )
    assert planner.run()
    provider = getattr(_shuffle, provider_name)
    call = _provider_call(func_ir, provider)
    assert len(call.args) == (2 if kind == "array" else 1)
    if runtime_distance:
        assert "shuffle_distance_i64" in dict(call.kws)["distance"].name
        assert any(
            target is types.int64 for target, _ in _planned_factory_calls(func_ir)
        )


@pytest.mark.parametrize("distance", [0, 64, -1])
def test_static_rotate_bounds_reject_before_provider(monkeypatch, distance):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_shuffle

    def shuffle(value):
        return coop.shuffle(
            coop.this_block(),
            value,
            mode="rotate",
            distance=distance,
        )

    monkeypatch.setattr(
        _group_shuffle._ShufflePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid rotate distance reached provider selection"
        ),
    )
    _, planner = _plan(shuffle, arg_types=(types.int32,))
    with pytest.raises(ValueError, match="1 <= distance < block_threads"):
        planner.run()


def test_runtime_shuffle_distance_rejects_noninteger_dtype():
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._compiler._rewrite_shuffle import (
        validate_shuffle_scalar_runtime_controls,
    )
    from cuda.coop.numba_mlir._compiler._rewrite_support import (
        CoopSinglePhaseRewriteError,
    )

    scope = ir.Scope(parent=None, loc=ir.Loc("test", 1))
    value = ir.Var(scope, "value", scope.loc)
    distance = ir.Var(scope, "distance", scope.loc)
    context = SimpleNamespace(
        numba_type=lambda var: types.float32 if var is distance else types.int32,
        dtype=lambda _var: None,
    )
    with pytest.raises(CoopSinglePhaseRewriteError, match="must be an integer"):
        validate_shuffle_scalar_runtime_controls(
            context,
            op_name="shuffle_scalar",
            runtime_args=[value, distance],
            factory_kwargs={
                "distance": ArgumentBinding.runtime(),
                "mode": "offset",
                "threads_per_block": (64, 1, 1),
            },
        )


@pytest.mark.parametrize(
    "distance_dtype",
    (
        pytest.param("boolean", id="bool"),
        pytest.param("float32", id="float"),
        pytest.param("uint64", id="uint64"),
    ),
)
def test_shuffle_planner_rejects_invalid_runtime_distance_before_cast(
    monkeypatch,
    distance_dtype,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler import _group_shuffle

    def shuffle(value, distance):
        return coop.shuffle(
            coop.this_block(),
            value,
            mode="offset",
            distance=distance,
        )

    monkeypatch.setattr(
        _group_shuffle._ShufflePlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid runtime distance reached provider selection"
        ),
    )
    _, planner = _plan(
        shuffle,
        arg_types=(types.int32, getattr(types, distance_dtype)),
    )
    with pytest.raises(TypeError, match="must be an integer|unsigned integer"):
        planner.run()


def test_planned_exchange_and_shuffle_calls_match_before_inference():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def flagged(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        ranks = coop.ThreadData(2, dtype=types.int32)
        ranks[0] = 0
        ranks[1] = 1
        flags = coop.ThreadData(2, dtype=types.uint8)
        flags[0] = 1
        flags[1] = 1
        return coop.exchange(
            coop.this_block(),
            items,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        )

    def scalar(value, distance):
        return coop.shuffle(
            coop.this_block(),
            value,
            mode="offset",
            distance=distance,
        )

    def array(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = value
        return coop.shuffle(coop.this_block(), items, mode="down")

    for function, arg_types in (
        (flagged, (types.int32,)),
        (scalar, (types.int32, types.int64)),
        (array, (types.int32,)),
    ):
        func_ir, planner = _plan(function, arg_types=arg_types)
        assert planner.run()
        _match_before_inference(func_ir, arg_types=arg_types)
