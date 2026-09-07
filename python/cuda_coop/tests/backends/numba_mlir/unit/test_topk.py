# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, *, arg_types, block=(64, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    func_ir = run_frontend(function)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


def _factory_calls(func_ir, ir):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return tuple(
        (globals_by_name.get(inst.value.func.name), inst.value, inst.target)
        for block in func_ir.blocks.values()
        for inst in block.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
    )


def _rewrite_with_fake(func_ir, arg_types):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    class FakeInvocable:
        files = ("topk-test.ltoir",)
        temp_storage_bytes = 64
        temp_storage_alignment = 16

        def __call__(self, *args):
            del args

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
    rewrites = 0
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
            rewrites += 1
    return invocable, rewrites


def _single_phase_matches(func_ir, arg_types):
    from cuda.coop.numba_mlir._compiler._rewrite import CoopSinglePhaseRewrite

    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=SimpleNamespace(refresh=lambda: None),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    matches = []
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        if rewrite.match(func_ir, block, state.typemap, state.calltypes):
            matches.extend(rewrite._matches.values())
    return matches


def test_common_and_qualified_topk_lower_to_fresh_payload_factories() -> None:
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as coop
    from cuda import coop as common_coop
    from cuda.coop.numba_mlir._lowering import _topk as _topk_lowering

    def kernel(key, value, k):
        keys = coop.ThreadData(2, dtype=types.int32)
        values = coop.ThreadData(2, dtype=types.float32)
        for item in range(2):
            keys[item] = key
            values[item] = value
        common_keys = common_coop.topk_max_keys(common_coop.this_block(), keys, k)
        qualified_keys = coop.topk_min_keys(coop.this_block(), keys, k, begin_bit=3)
        common_pair_keys, common_pair_values = common_coop.topk_max_pairs(
            common_coop.this_block(), keys, values, k
        )
        qualified_pair_keys, qualified_pair_values = coop.topk_min_pairs(
            coop.this_block(), keys, values, k, begin_bit=3
        )
        return (
            common_keys[0]
            + qualified_keys[0]
            + common_pair_keys[0]
            + common_pair_values[0]
            + qualified_pair_keys[0]
            + qualified_pair_values[0]
        )

    arg_types = (types.int32, types.float32, types.int32)
    func_ir, planner = _plan(kernel, arg_types=arg_types)
    assert planner.run()
    calls = _factory_calls(func_ir, ir)
    expected_factories = {
        _topk_lowering._common_topk_max_keys,
        _topk_lowering._qualified_group_topk_min_keys,
        _topk_lowering._common_topk_max_pairs,
        _topk_lowering._qualified_group_topk_min_pairs,
    }
    factories = Counter(
        factory for factory, _call, _target in calls if factory in expected_factories
    )
    assert factories == Counter({factory: 1 for factory in expected_factories})
    topk_calls = [call for factory, call, _target in calls if factory in factories]
    assert all("_result_payload_" in call.args[0].name for call in topk_calls)
    assert all(
        "_values_result_payload_" in call.args[1].name
        for call in topk_calls
        if len(call.args) == 3
    )

    invocable, rewrites = _rewrite_with_fake(func_ir, arg_types)
    assert rewrites > 0
    invocable_calls = [
        call
        for factory, call, _target in _factory_calls(func_ir, ir)
        if factory is invocable
    ]
    assert sorted(len(call.args) for call in invocable_calls) == [2, 2, 3, 3]


def test_common_topk_requires_thread_data_but_qualified_accepts_local_arrays() -> None:
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop
    from cuda import coop as common_coop

    def common(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return common_coop.topk_max_keys(common_coop.this_block(), keys, 1)

    _func_ir, planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(TypeError, match="requires keys to be coop.ThreadData"):
        planner.run()

    def qualified(value):
        keys = cuda.local.array(2, types.float32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, 1)

    _func_ir, planner = _plan(qualified, arg_types=(types.float32,))
    assert planner.run()


@pytest.mark.parametrize("control", [False, np.bool_(True), 1.5])
def test_qualified_topk_rejects_noninteger_static_controls(control) -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, control)

    _func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(TypeError, match="k must be an int-like scalar"):
        planner.run()


@pytest.mark.parametrize(
    ("k", "valid_items", "begin_bit", "end_bit", "message"),
    [
        (0, None, 0, None, "k must be positive"),
        (5, 4, 0, None, "k must be <= valid_items"),
        (1, 129, 0, None, r"valid_items must be in \[1, 128\]"),
        (1, None, -1, None, "begin_bit must be non-negative"),
        (1, None, 8, 8, "end_bit must exceed begin_bit"),
    ],
)
def test_qualified_topk_checks_static_control_ranges(
    k,
    valid_items,
    begin_bit,
    end_bit,
    message,
) -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(
            coop.this_block(),
            keys,
            k,
            valid_items=valid_items,
            begin_bit=begin_bit,
            end_bit=end_bit,
        )

    _func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises((TypeError, ValueError), match=message):
        planner.run()


@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
def test_common_topk_factory_accepts_portable_integer_keys(dtype_name) -> None:
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _topk as _topk_lowering
    from cuda.coop.numba_mlir._types import collect_specializations

    with collect_specializations() as collected:
        specialization = _topk_lowering._common_topk_max_keys(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=4,
        )
    assert specialization is collected[0][0]
    assert specialization.method_name == "max_keys_full"


def test_topk_factory_preserves_explicit_scalar_bindings() -> None:
    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _topk as _topk_lowering

    runtime = ArgumentBinding.runtime()
    static = ArgumentBinding.static(7)

    assert _topk_lowering._topk_binding(runtime, name="num_valid") is runtime
    assert _topk_lowering._topk_binding(static, name="num_valid") is static


def test_topk_rewrite_preserves_bit_control_bindings() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop._core import ArgumentBinding

    def static_bits(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, 1, begin_bit=2, end_bit=19)

    def mixed_bits(value, begin_bit):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(
            coop.this_block(), keys, 1, begin_bit=begin_bit, end_bit=19
        )

    def omitted_bits(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, 1)

    cases = (
        (
            static_bits,
            (types.int32,),
            ArgumentBinding.static(2),
            ArgumentBinding.static(19),
            2,
        ),
        (
            mixed_bits,
            (types.int32, types.int32),
            ArgumentBinding.runtime(),
            ArgumentBinding.static(19),
            3,
        ),
    )
    for function, arg_types, expected_begin, expected_end, runtime_count in cases:
        func_ir, planner = _plan(function, arg_types=arg_types)
        assert planner.run()
        match = next(
            item
            for item in _single_phase_matches(func_ir, arg_types)
            if item.op_name.endswith("topk_max_keys")
        )
        assert match.factory_kwargs["begin_bit"] == expected_begin
        assert match.factory_kwargs["end_bit"] == expected_end
        assert len(match.runtime_args) == runtime_count

    arg_types = (types.int32,)
    func_ir, planner = _plan(omitted_bits, arg_types=arg_types)
    assert planner.run()
    match = next(
        item
        for item in _single_phase_matches(func_ir, arg_types)
        if item.op_name.endswith("topk_max_keys")
    )
    assert "begin_bit" not in match.factory_kwargs
    assert "end_bit" not in match.factory_kwargs
    assert len(match.runtime_args) == 2


@pytest.mark.parametrize("dtype_name", ["boolean", "int16", "float32"])
def test_common_topk_factory_rejects_nonportable_key_dtypes(dtype_name) -> None:
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _topk as _topk_lowering

    with pytest.raises(
        TypeError,
        match=(
            r"supports key dtypes int32, uint32, int64, uint64 through the "
            r"portable API"
        ),
    ):
        _topk_lowering._common_topk_max_keys(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
        )


@pytest.mark.parametrize("control_type", ["boolean", "float32"])
def test_dynamic_topk_controls_require_integer_dtypes(control_type) -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._compiler._rewrite import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def kernel(value, k):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(coop.this_block(), keys, k)

    arg_types = (types.int32, getattr(types, control_type))
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
    block = func_ir.blocks[sorted(func_ir.blocks)[0]]
    with pytest.raises(CoopSinglePhaseRewriteError, match="k must have an integer"):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)


def test_dynamic_begin_bit_accepts_a_static_end_bit() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value, begin_bit):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.topk_max_keys(
            coop.this_block(),
            keys,
            1,
            begin_bit=begin_bit,
            end_bit=31,
        )

    arg_types = (types.int32, types.int32)
    func_ir, planner = _plan(kernel, arg_types=arg_types)
    assert planner.run()
    _invocable, rewrites = _rewrite_with_fake(func_ir, arg_types)
    assert rewrites == 1
