# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


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


def _single_phase_rewrite(func_ir, arg_types):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

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
    return state, rewrite


def test_public_radix_exports_keep_factories_private() -> None:
    pytest.importorskip("numba_cuda_mlir")

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering

    assert {"radix_rank", "radix_sort_keys", "radix_sort_pairs"} <= set(coop.__all__)
    assert lowering.__all__ == ()
    for name in ("radix_rank", "radix_sort_keys", "radix_sort_pairs"):
        assert getattr(coop, name).__module__.endswith("._group_radix")
    for name in (
        "radix_rank",
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_sort_pairs",
        "radix_sort_pairs_descending",
    ):
        assert callable(getattr(lowering, name))


def test_qualified_thread_data_radix_lowers_to_block_factories() -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering
    from cuda.coop.numba_mlir._compiler._group_planner import (
        _typed_group_payload_like,
        has_group_markers,
    )

    def function(key, value):
        keys = coop.ThreadData(2, dtype=types.int32)
        values = coop.ThreadData(2, dtype=types.float32)
        keys[0] = key
        keys[1] = key - 1
        values[0] = value
        values[1] = value + 1
        sorted_keys = coop.radix_sort_keys(
            coop.this_block(),
            keys,
            begin_bit=4,
            end_bit=12,
            descending=True,
        )
        pair_result = coop.radix_sort_pairs(coop.this_block(), keys, values)
        ranks = coop.radix_rank(
            coop.this_block(),
            keys,
            begin_bit=0,
            radix_bits=4,
        )
        return keys[0], sorted_keys, pair_result, ranks

    func_ir, planner = _plan(
        function,
        arg_types=(types.int32, types.float32),
    )
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    calls = _planned_factory_calls(func_ir, ir)
    counts = Counter(factory for factory, _ in calls)
    assert counts[lowering.radix_sort_keys_descending] == 1
    assert counts[lowering.radix_sort_pairs] == 1
    assert counts[lowering.radix_rank] == 1
    assert counts[_typed_group_payload_like] == 4
    provider_calls = [
        call
        for factory, call in calls
        if factory
        in {
            lowering.radix_sort_keys_descending,
            lowering.radix_sort_pairs,
            lowering.radix_rank,
        }
    ]
    assert sorted(len(call.args) for call in provider_calls) == [1, 2, 2]
    setitems = [
        inst
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.SetItem)
    ]
    # Four source initializations plus the fresh key/value result copies.
    assert len(setitems) == 10


def test_radix_rank_result_uses_int32_through_single_phase_scan() -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def function(key):
        keys = coop.ThreadData(2, dtype=types.uint32)
        keys[0] = key
        return coop.radix_rank(coop.this_block(), keys)

    arg_types = (types.uint32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    _state, rewrite = _single_phase_rewrite(func_ir, arg_types)

    assert rewrite._compute_func_temp_storage_requirements(func_ir) == {}
    assert any(
        spec.dtype == types.int32 for spec in rewrite._thread_data_specs.values()
    )


@pytest.mark.parametrize("factory_name", ["qualified", "common"])
@pytest.mark.parametrize(
    "dimension_source",
    ["dim_alias", "launch_inference", "default_radix_bits"],
)
def test_radix_rank_rejects_undersized_prefix_after_factory_finalization(
    factory_name,
    dimension_source,
) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    import cuda.coop.numba_mlir._lowering as lowering
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewriteError,
    )
    from cuda.coop.numba_mlir._lowering._radix import _common_radix_rank

    factory = lowering.radix_rank if factory_name == "qualified" else _common_radix_rank

    if dimension_source == "dim_alias":

        def function(value):
            keys = coop.ThreadData(1, dtype=types.int32)
            ranks = coop.ThreadData(1, dtype=types.int32)
            prefix = coop.ThreadData(1, dtype=types.int32)
            keys[0] = value
            factory(
                keys,
                ranks,
                dtype=types.int32,
                dim=64,
                begin_bit=0,
                end_bit=8,
                exclusive_digit_prefix=prefix,
            )
            return ranks[0]

        targetoptions = {}
    elif dimension_source == "launch_inference":

        def function(value):
            keys = coop.ThreadData(1, dtype=types.int32)
            ranks = coop.ThreadData(1, dtype=types.int32)
            prefix = coop.ThreadData(1, dtype=types.int32)
            keys[0] = value
            factory(
                keys,
                ranks,
                dtype=types.int32,
                begin_bit=0,
                end_bit=8,
                exclusive_digit_prefix=prefix,
            )
            return ranks[0]

        targetoptions = {"__launch_config__": {"block": (64, 1, 1)}}
    else:

        def function(value):
            keys = coop.ThreadData(1, dtype=types.int32)
            ranks = coop.ThreadData(1, dtype=types.int32)
            prefix = coop.ThreadData(1, dtype=types.int32)
            keys[0] = value
            factory(
                keys,
                ranks,
                dtype=types.int32,
                threads_per_block=4,
                exclusive_digit_prefix=prefix,
            )
            return ranks[0]

        targetoptions = {}

    arg_types = (types.int32,)
    func_ir, _planner = _plan(function, arg_types=arg_types)
    state, rewrite = _single_phase_rewrite(func_ir, arg_types)
    state.metadata["targetoptions"] = targetoptions
    block = func_ir.blocks[min(func_ir.blocks)]
    before = str(func_ir)
    with pytest.raises(CoopSinglePhaseRewriteError) as exc_info:
        rewrite.match(func_ir, block, state.typemap, state.calltypes)
    scope_name = "cuda.coop.numba_mlir" if factory_name == "qualified" else "cuda.coop"
    assert str(exc_info.value) == (
        f"{scope_name}.radix_rank exclusive_digit_prefix must contain "
        "4 items per thread"
    )
    assert str(func_ir) == before


def test_radix_rank_prefix_extent_validation_rejects_missing_runtime_argument() -> None:
    pytest.importorskip("numba_cuda_mlir")
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    rewrite = object.__new__(CoopSinglePhaseRewrite)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.numba_mlir\.radix_rank internal rewrite error: "
            "exclusive_digit_prefix has no runtime argument"
        ),
    ):
        rewrite._validate_radix_rank_exclusive_digit_prefix_extent(
            op_name="radix_rank",
            control_vars={},
            factory_kwargs={"exclusive_digit_prefix": True},
        )


@pytest.mark.parametrize("operation", ["keys", "pairs", "rank"])
def test_qualified_scalar_radix_boxes_and_projects_fresh_results(operation) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._group_planner import _typed_group_payload_like

    if operation == "keys":

        def function(key, value):
            del value
            return key, coop.radix_sort_keys(coop.this_block(), key)

    elif operation == "pairs":

        def function(key, value):
            return key, value, coop.radix_sort_pairs(coop.this_block(), key, value)

    else:

        def function(key, value):
            del value
            return key, coop.radix_rank(coop.this_block(), key)

    func_ir, planner = _plan(
        function,
        arg_types=(types.int32, types.float32),
    )
    assert planner.run()
    calls = _planned_factory_calls(func_ir, ir)
    expected_payloads = 4 if operation == "pairs" else 2
    assert Counter(factory for factory, _ in calls)[_typed_group_payload_like] == (
        expected_payloads
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
    expected_projections = 1 if operation == "rank" else expected_payloads
    assert len(setitems) == expected_projections
    assert len(getitems) == expected_projections


def test_common_thread_data_radix_uses_profile_factories() -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir  # noqa: F401
    from cuda import coop
    from cuda.coop.numba_mlir._lowering._radix import (
        _common_radix_rank,
        _common_radix_sort_keys,
        _common_radix_sort_pairs,
    )

    def function(key, value):
        keys = coop.ThreadData(2, dtype=int)
        values = coop.ThreadData(2, dtype=float)
        keys[0] = key
        keys[1] = key - 1
        values[0] = value
        values[1] = value + 1
        return (
            coop.radix_sort_keys(coop.this_block(), keys),
            coop.radix_sort_pairs(coop.this_block(), keys, values),
            coop.radix_rank(coop.this_block(), keys),
        )

    func_ir, planner = _plan(
        function,
        arg_types=(types.int32, types.float32),
    )
    assert planner.run()
    calls = _planned_factory_calls(func_ir, ir)
    counts = Counter(factory for factory, _ in calls)
    assert counts[_common_radix_sort_keys] == 1
    assert counts[_common_radix_sort_pairs] == 1
    assert counts[_common_radix_rank] == 1


@pytest.mark.parametrize("operation", ["radix_sort_keys", "radix_rank"])
def test_radix_is_block_only(operation) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    if operation == "radix_sort_keys":

        def function(value):
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.radix_sort_keys(coop.this_warp(), keys)

    else:

        def function(value):
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.radix_rank(coop.this_warp(), keys)

    _, planner = _plan(function, arg_types=(types.int32,))
    with pytest.raises(NotImplementedError, match="only complete physical block"):
        planner.run()


@pytest.mark.parametrize(
    ("keyword", "invalid", "error_type", "message"),
    [
        ("begin_bit", True, TypeError, "must be an integer"),
        ("begin_bit", np.bool_(True), TypeError, "must be an integer"),
        ("begin_bit", 1.5, TypeError, "must be an integer"),
        ("begin_bit", -1, ValueError, "must be non-negative"),
        ("radix_bits", 0, ValueError, "must be positive"),
        ("end_bit", 9, ValueError, "bit width must be <= 8"),
        ("descending", np.bool_(False), TypeError, "compile-time bool"),
    ],
)
def test_static_rank_control_rejection_is_transactional(
    keyword,
    invalid,
    error_type,
    message,
) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

    controls = {
        "begin_bit": 0,
        "end_bit": None,
        "radix_bits": None,
        "descending": False,
    }
    controls[keyword] = invalid
    begin_bit = controls["begin_bit"]
    end_bit = controls["end_bit"]
    radix_bits = controls["radix_bits"]
    descending = controls["descending"]

    def function(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(
            coop.this_block(),
            keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            radix_bits=radix_bits,
            descending=descending,
        )

    func_ir, planner = _plan(function, arg_types=(types.int32,))
    before = str(func_ir)
    with pytest.raises(error_type, match=message):
        planner.run()
    assert str(func_ir) == before
    assert has_group_markers(func_ir)


@pytest.mark.parametrize(
    ("keyword", "invalid", "error_type", "message"),
    [
        ("begin_bit", True, TypeError, "must be an integer"),
        ("begin_bit", np.bool_(True), TypeError, "must be an integer"),
        ("begin_bit", 1.5, TypeError, "must be an integer"),
        ("begin_bit", -1, ValueError, "must be non-negative"),
        ("end_bit", 0, ValueError, "must be positive"),
        ("end_bit", 4, ValueError, "greater than begin_bit"),
        ("descending", np.bool_(False), TypeError, "compile-time bool"),
        ("blocked_to_striped", np.bool_(False), TypeError, "compile-time bool"),
    ],
)
def test_static_sort_control_rejection_is_transactional(
    keyword,
    invalid,
    error_type,
    message,
) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._group_planner import has_group_markers

    controls = {
        "begin_bit": 4,
        "end_bit": 12,
        "descending": False,
        "blocked_to_striped": False,
    }
    controls[keyword] = invalid
    begin_bit = controls["begin_bit"]
    end_bit = controls["end_bit"]
    descending = controls["descending"]
    blocked_to_striped = controls["blocked_to_striped"]

    def function(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_sort_keys(
            coop.this_block(),
            keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            blocked_to_striped=blocked_to_striped,
        )

    func_ir, planner = _plan(function, arg_types=(types.int32,))
    before = str(func_ir)
    with pytest.raises(error_type, match=message):
        planner.run()
    assert str(func_ir) == before
    assert has_group_markers(func_ir)


@pytest.mark.parametrize(
    ("bound_type", "message"),
    [
        ("boolean", "integer dtype"),
        ("float32", "integer dtype"),
    ],
)
def test_dynamic_sort_bit_bounds_require_integer_compiler_dtypes(
    bound_type,
    message,
) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewriteError,
    )

    def function(value, begin_bit, end_bit):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_sort_keys(
            coop.this_block(),
            keys,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    arg_types = (types.int32, getattr(types, bound_type), types.int32)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state, rewrite = _single_phase_rewrite(func_ir, arg_types)
    block = func_ir.blocks[min(func_ir.blocks)]
    before = str(func_ir)
    with pytest.raises(CoopSinglePhaseRewriteError, match=message):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)
    assert str(func_ir) == before


@pytest.mark.parametrize("fixed", [False, True], ids=["deferred", "fixed"])
def test_sort_storage_uses_capacity_alignment_and_auto_sync(fixed) -> None:
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    class FakeInvocable:
        files = ("radix-sort-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

    if fixed:

        def function(value):
            storage = coop.TempStorage(4096, alignment=32, auto_sync=False)
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.radix_sort_keys(
                coop.this_block(),
                keys,
                temp_storage=storage,
            )

    else:

        def function(value):
            storage = coop.TempStorage()
            keys = coop.ThreadData(2, dtype=types.int32)
            keys[0] = value
            return coop.radix_sort_keys(
                coop.this_block(),
                keys,
                temp_storage=storage,
            )

    arg_types = (types.int32,)
    func_ir, planner = _plan(function, arg_types=arg_types)
    assert planner.run()
    state, rewrite = _single_phase_rewrite(func_ir, arg_types)
    invocable = FakeInvocable()
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
    # Payload plus actual byte-addressed scratch array.
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
    assert resolver._infer_constant(shared_array_calls[0].args[0]) == (
        4096 if fixed else 64
    )
    alignment_var = dict(shared_array_calls[0].kws)["alignment"]
    assert resolver._infer_constant(alignment_var) == (32 if fixed else 16)
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
