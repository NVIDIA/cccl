# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _less(lhs, rhs):
    return lhs < rhs


class _TypingContext:
    def refresh(self):
        pass


def _plan(function, *, arg_types, block=(64, 1, 1)):
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


def _group_for_kind(coop, group_kind):
    if group_kind == "block":
        return coop.this_block()
    if group_kind == "warp":
        return coop.this_warp()
    return coop.this_warp().group_by(8)


def _thread_data_function(group_kind, *, pairs):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    group = _group_for_kind(coop, group_kind)

    if pairs:

        def function(key, value):
            keys = coop.ThreadData(2, dtype=types.int32)
            values = coop.ThreadData(2, dtype=types.float32)
            keys[0] = key
            keys[1] = key - 1
            values[0] = value
            values[1] = value + 1
            return keys[0], coop.merge_sort_pairs(group, keys, values)

    else:

        def function(key):
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = key
            keys[1] = key - 1
            return keys[0], coop.merge_sort_keys(group, keys)

    return function


def _scalar_function(group_kind, *, pairs):
    import cuda.coop.numba_mlir as coop

    group = _group_for_kind(coop, group_kind)

    if pairs:

        def function(key, value):
            return coop.merge_sort_pairs(group, key, value)

    else:

        def function(key):
            return coop.merge_sort_keys(group, key)

    return function


def _common_thread_data_function(group_kind, *, pairs):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop

    group = _group_for_kind(coop, group_kind)

    if pairs:

        def function(key, value):
            keys = coop.ThreadData(2, dtype=types.int32)
            values = coop.ThreadData(2, dtype=types.float32)
            keys[0] = key
            values[0] = value
            return coop.merge_sort_pairs(group, keys, values)

    else:

        def function(key):
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = key
            return coop.merge_sort_keys(group, keys)

    return function


def test_public_exports_are_qualified_and_providers_remain_private():
    pytest.importorskip("numba_cuda_mlir")

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    assert {"merge_sort_keys", "merge_sort_pairs"} <= set(coop.__all__)
    assert lowering.__all__ == ()
    assert coop.merge_sort_keys.__module__.endswith("._group_merge_sort")
    assert coop.merge_sort_pairs.__module__.endswith("._group_merge_sort")


@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
@pytest.mark.parametrize("pairs", [False, True], ids=["keys", "pairs"])
def test_common_root_merge_sort_reaches_the_private_provider(group_kind, pairs):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._lowering as lowering

    function = _common_thread_data_function(group_kind, pairs=pairs)
    arg_types = (types.int32, types.float32) if pairs else (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()

    expected = lowering.merge_sort_pairs if pairs else lowering.merge_sort_keys
    if group_kind != "block":
        expected = (
            lowering.warp_merge_sort_pairs if pairs else lowering.warp_merge_sort_keys
        )
    provider_call = next(
        call
        for factory, call in _planned_factory_calls(func_ir, ir)
        if factory is expected
    )
    constants = {
        inst.target.name: inst.value.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
    }
    operation = "merge_sort_pairs" if pairs else "merge_sort_keys"
    marker = dict(provider_call.kws)["_common_root_operation"]
    assert constants[marker.name] == operation


def test_common_root_merge_sort_rejects_scalar_keys():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop

    def function(key):
        return coop.merge_sort_keys(coop.this_block(), key)

    _, planner = _plan(function, arg_types=(types.int32,))
    with pytest.raises(TypeError, match="fixed-size ThreadData"):
        planner.run()


def test_common_root_merge_sort_rejects_floating_point_keys():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def function(value):
        keys = coop.ThreadData(2, dtype=types.float32)
        keys[0] = value
        return coop.merge_sort_keys(coop.this_block(), keys)

    arg_types = (types.float32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    block = func_ir.blocks[min(func_ir.blocks)]
    with pytest.raises(CoopSinglePhaseRewriteError, match="through the portable API"):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)


@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
@pytest.mark.parametrize("pairs", [False, True], ids=["keys", "pairs"])
def test_thread_data_merge_sort_lowers_to_exact_private_provider(group_kind, pairs):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir._lowering as lowering
    from cuda.coop.numba_mlir._compiler._group_planner import (
        _typed_group_payload_like,
        has_group_markers,
    )

    function = _thread_data_function(group_kind, pairs=pairs)
    arg_types = (types.int32, types.float32) if pairs else (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    if group_kind == "block":
        expected = lowering.merge_sort_pairs if pairs else lowering.merge_sort_keys
    else:
        expected = (
            lowering.warp_merge_sort_pairs if pairs else lowering.warp_merge_sort_keys
        )
    calls = _planned_factory_calls(func_ir, ir)
    assert Counter(factory for factory, _ in calls)[expected] == 1
    assert Counter(factory for factory, _ in calls)[_typed_group_payload_like] == (
        2 if pairs else 1
    )
    provider_call = next(call for factory, call in calls if factory is expected)
    assert len(provider_call.args) == (2 if pairs else 1)
    if group_kind == "logical_warp":
        constants = {
            inst.target.name: inst.value.value
            for block in func_ir.blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
        }
        assert constants[dict(provider_call.kws)["threads_in_warp"].name] == 8


@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
@pytest.mark.parametrize("pairs", [False, True], ids=["keys", "pairs"])
def test_scalar_merge_sort_boxes_and_projects_fresh_results(group_kind, pairs):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir._compiler._group_planner import _typed_group_payload_like

    function = _scalar_function(group_kind, pairs=pairs)
    arg_types = (types.int32, types.float32) if pairs else (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()

    calls = _planned_factory_calls(func_ir, ir)
    assert Counter(factory for factory, _ in calls)[_typed_group_payload_like] == (
        4 if pairs else 2
    )
    setitems = [
        inst
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.SetItem)
    ]
    getitems = [
        inst
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "getitem"
    ]
    assert len(setitems) == (4 if pairs else 2)
    assert len(getitems) == (4 if pairs else 2)


@pytest.mark.parametrize(
    ("valid_items", "error_type", "message"),
    [
        (True, TypeError, "integer, not bool"),
        (np.bool_(True), TypeError, "integer, not bool"),
        (1.5, TypeError, "must be an integer"),
        (np.float32(1.5), TypeError, "must be an integer"),
        (-1, ValueError, r"must be in \[0, 128\]"),
        (129, ValueError, r"must be in \[0, 128\]"),
    ],
)
def test_static_valid_items_rejection_is_transactional(
    valid_items,
    error_type,
    message,
):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

    def function(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.merge_sort_keys(
            coop.this_block(),
            keys,
            valid_items=valid_items,
            oob_default=np.int32(2_147_483_647),
        )

    func_ir, planner = _plan(function, arg_types=(types.int32,))
    before = str(func_ir)
    with pytest.raises(error_type, match=message):
        planner.run()
    assert str(func_ir) == before
    assert has_group_markers(func_ir)


@pytest.mark.parametrize(
    ("group_kind", "maximum"),
    [("block", 64), ("warp", 32), ("logical_warp", 8)],
)
@pytest.mark.parametrize("pairs", [False, True], ids=["keys", "pairs"])
@pytest.mark.parametrize("offset", [-1, 1], ids=["below", "above"])
def test_scalar_static_valid_items_respects_group_bounds(
    group_kind,
    maximum,
    pairs,
    offset,
):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

    group = _group_for_kind(coop, group_kind)
    valid_items = -1 if offset < 0 else maximum + 1

    if pairs:

        def function(key, value):
            return coop.merge_sort_pairs(
                group,
                key,
                value,
                valid_items=valid_items,
                oob_default=np.int32(2_147_483_647),
            )

        arg_types = (types.int32, types.float32)
    else:

        def function(key):
            return coop.merge_sort_keys(
                group,
                key,
                valid_items=valid_items,
                oob_default=np.int32(2_147_483_647),
            )

        arg_types = (types.int32,)

    func_ir, planner = _plan(function, arg_types=arg_types)
    before = str(func_ir)
    with pytest.raises(ValueError, match=rf"must be in \[0, {maximum}\]"):
        planner.run()
    assert str(func_ir) == before
    assert has_group_markers(func_ir)


@pytest.mark.parametrize(
    ("valid_items_type", "message"),
    [
        ("boolean", "integer, not bool"),
        ("float32", "integer dtype"),
    ],
)
def test_dynamic_valid_items_rejects_bool_and_noninteger_compiler_types(
    valid_items_type,
    message,
):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def function(value, valid_items, oob_default):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.merge_sort_keys(
            coop.this_block(),
            keys,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    arg_types = (types.int32, getattr(types, valid_items_type), types.int32)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    block = func_ir.blocks[min(func_ir.blocks)]
    before = str(func_ir)
    with pytest.raises(CoopSinglePhaseRewriteError, match=message):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)
    assert str(func_ir) == before


def test_partial_sentinel_conversion_is_lossless_and_key_typed():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    convert = CoopSinglePhaseRewrite._lossless_merge_sort_sentinel
    assert isinstance(convert(np.int64(17), types.int32), np.int32)
    assert isinstance(convert(np.float32(1.25), types.float32), np.float32)
    with pytest.raises(CoopSinglePhaseRewriteError, match="not representable"):
        convert(1 << 31, types.int32)
    with pytest.raises(CoopSinglePhaseRewriteError, match="not losslessly"):
        convert(0.1, types.float32)
    with pytest.raises(CoopSinglePhaseRewriteError, match="integer, not bool"):
        convert(np.bool_(True), types.int32)


def test_dynamic_sentinel_must_match_the_key_dtype():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def function(value, valid_items, oob_default):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.merge_sort_keys(
            coop.this_block(),
            keys,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    arg_types = (types.int32, types.int32, types.int64)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    block = func_ir.blocks[min(func_ir.blocks)]
    with pytest.raises(CoopSinglePhaseRewriteError, match="same dtype as keys"):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)


def test_warp_merge_sort_rejects_explicit_temp_storage():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def function(value):
        storage = coop.TempStorage()
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.merge_sort_keys(
            coop.this_warp(),
            keys,
            temp_storage=storage,
        )

    _, planner = _plan(function, arg_types=(types.int32,))
    with pytest.raises(NotImplementedError, match="only for block groups"):
        planner.run()


@pytest.mark.parametrize("fixed", [False, True], ids=["deferred", "fixed"])
def test_block_storage_uses_real_capacity_alignment_and_auto_sync(fixed):
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    class FakeInvocable:
        files = ("merge-sort-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

    if fixed:

        def function(value):
            storage = coop.TempStorage(4096, alignment=32, auto_sync=False)
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.merge_sort_keys(
                coop.this_block(),
                keys,
                temp_storage=storage,
            )

    else:

        def function(value):
            storage = coop.TempStorage()
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.merge_sort_keys(
                coop.this_block(),
                keys,
                temp_storage=storage,
            )

    arg_types = (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
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
    assert len(invocable_calls[0].args) == 2

    resolver = object.__new__(CoopSinglePhaseRewrite)
    resolver._func_ir = func_ir
    shared_array_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        shared_array_calls.extend(
            inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and resolver._resolve_python_value(inst.value.func) is cuda.shared.array
        )
    assert len(shared_array_calls) == 1
    resolver._block_defs = {
        inst.target.name: inst.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
    }
    expected_size = 4096 if fixed else 64
    expected_alignment = 32 if fixed else 16
    assert resolver._infer_constant(shared_array_calls[0].args[0]) == expected_size
    alignment_var = dict(shared_array_calls[0].kws)["alignment"]
    assert resolver._infer_constant(alignment_var) == expected_alignment
    sync_calls = []
    for block in func_ir.blocks.values():
        resolver._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        sync_calls.extend(
            inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and resolver._resolve_python_value(inst.value.func) is cuda.syncthreads
        )
    assert len(sync_calls) == (0 if fixed else 1)


def test_insufficient_fixed_storage_rolls_back_the_rewrite():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    class FakeInvocable:
        files = ("merge-sort-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

    def function(value):
        storage = coop.TempStorage(8, alignment=16)
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.merge_sort_keys(
            coop.this_block(),
            keys,
            temp_storage=storage,
        )

    arg_types = (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    invocable = FakeInvocable()
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    rewrite._materialize_invocable = lambda _match: (invocable, False)
    rewrite._record_invocable_specialization = lambda _invocable: None
    block = func_ir.blocks[min(func_ir.blocks)]
    assert rewrite.match(func_ir, block, state.typemap, state.calltypes)
    before = str(func_ir)
    with pytest.raises(CoopSinglePhaseRewriteError, match="smaller than required"):
        rewrite.apply()
    assert str(func_ir) == before
