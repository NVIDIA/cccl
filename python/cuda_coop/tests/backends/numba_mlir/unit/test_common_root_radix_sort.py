# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from inspect import signature
from types import SimpleNamespace

import pytest

_BLOCK = (64, 1, 1)


def _plan(function, *, arg_types):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    planner = _GroupCallPlanner(
        SimpleNamespace(func_ir=func_ir, args=arg_types),
        {"block": _BLOCK, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


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


def _factory_call_keywords(func_ir, ir, factory):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return [
        tuple(name for name, _value in inst.value.kws)
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
        and globals_by_name.get(inst.value.func.name) is factory
    ]


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="numba_mlir", evidence="lowering"
)
@pytest.mark.parametrize("qualified", [False, True])
def test_group_radix_sort_begin_only_lowers_through_dtype_width(
    optional_backend,
    qualified,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_radix_sort
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    api = numba_coop if qualified else coop

    def cohort(value):
        keys = api.ThreadData(2, dtype=types.int32)
        keys[0] = value
        keys[1] = value - 1
        result = api.radix_sort_keys(api.this_block(), keys, begin_bit=8)
        return keys[0], result[0]

    func_ir, planner = _plan(cohort, arg_types=(types.int32,))
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    expected_factory = (
        numba_coop._block.radix_sort_keys
        if qualified
        else _block_radix_sort._common_radix_sort_keys
    )
    assert _planned_factories(func_ir, ir)[expected_factory] == 1
    expected_keywords = ["threads_per_block"]
    if not qualified:
        expected_keywords.append("descending")
    expected_keywords.extend(("begin_bit", "end_bit"))
    assert _factory_call_keywords(func_ir, ir, expected_factory) == [
        tuple(expected_keywords)
    ]
    # Two source writes plus two copy-before-sort writes preserve the input.
    assert (
        sum(
            isinstance(inst, ir.SetItem)
            for block_ir in func_ir.blocks.values()
            for inst in block_ir.body
        )
        == 4
    )


@pytest.mark.parametrize("qualified", [False, True])
def test_group_radix_sort_pairs_infers_keyword_thread_data_extent(
    optional_backend,
    qualified,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_radix_sort

    api = numba_coop if qualified else coop

    def cohort(key, value):
        keys = api.ThreadData(items_per_thread=2, dtype=types.int32)
        values = api.ThreadData(items_per_thread=2, dtype=types.int32)
        keys[0] = key
        keys[1] = key - 1
        values[0] = value
        values[1] = value - 1
        return api.radix_sort_pairs(api.this_block(), keys, values)

    func_ir, planner = _plan(
        cohort,
        arg_types=(types.int32, types.int32),
    )
    assert planner.run()
    expected_factory = (
        numba_coop._block.radix_sort_pairs
        if qualified
        else _block_radix_sort._common_radix_sort_pairs
    )
    assert _planned_factories(func_ir, ir)[expected_factory] == 1


@pytest.mark.parametrize("qualified", [False, True])
def test_group_radix_sort_runtime_begin_with_omitted_end_is_retained(
    optional_backend,
    qualified,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_radix_sort

    api = numba_coop if qualified else coop

    def cohort(value, begin_bit):
        keys = api.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return api.radix_sort_keys(
            api.this_block(),
            keys,
            begin_bit=begin_bit,
        )

    func_ir, planner = _plan(cohort, arg_types=(types.int32, types.int32))
    assert planner.run()
    expected_factory = (
        numba_coop._block.radix_sort_keys
        if qualified
        else _block_radix_sort._common_radix_sort_keys
    )
    expected_keywords = ["threads_per_block"]
    if not qualified:
        expected_keywords.append("descending")
    expected_keywords.extend(("begin_bit", "end_bit"))
    assert _factory_call_keywords(func_ir, ir, expected_factory) == [
        tuple(expected_keywords)
    ]


@pytest.mark.parametrize("begin_bit", [0, 8])
def test_qualified_scalar_radix_sort_boxes_copies_and_projects_result(
    optional_backend,
    begin_bit,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_radix_sort
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    if begin_bit:

        def cohort(value):
            return numba_coop.radix_sort_keys(
                numba_coop.this_block(),
                value,
                begin_bit=begin_bit,
                descending=True,
            )

        expected_factory = _block_radix_sort.radix_sort_keys_descending
    else:

        def cohort(value):
            return numba_coop.radix_sort_keys(numba_coop.this_block(), value)

        expected_factory = _block_radix_sort.radix_sort_keys

    func_ir, planner = _plan(cohort, arg_types=(types.int32,))
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)
    assert _planned_factories(func_ir, ir)[expected_factory] == 1

    instructions = [
        inst for block_ir in func_ir.blocks.values() for inst in block_ir.body
    ]
    assert sum(isinstance(inst, ir.SetItem) for inst in instructions) == 2
    assert (
        sum(
            isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "getitem"
            for inst in instructions
        )
        == 2
    )


def test_qualified_scalar_radix_sort_pairs_box_and_project_both_payloads(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_radix_sort
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def cohort(key, value):
        return numba_coop.radix_sort_pairs(
            numba_coop.this_block(),
            key,
            value,
            begin_bit=4,
            end_bit=10,
        )

    func_ir, planner = _plan(cohort, arg_types=(types.uint32, types.float32))
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)
    assert _planned_factories(func_ir, ir)[_block_radix_sort.radix_sort_pairs] == 1

    instructions = [
        inst for block_ir in func_ir.blocks.values() for inst in block_ir.body
    ]
    assert sum(isinstance(inst, ir.SetItem) for inst in instructions) == 4
    assert (
        sum(
            isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "getitem"
            for inst in instructions
        )
        == 4
    )


def test_qualified_radix_sort_lowers_blocked_to_striped(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_radix_sort

    def cohort(value):
        keys = numba_coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        keys[1] = value - 1
        return numba_coop.radix_sort_keys(
            numba_coop.this_block(),
            keys,
            blocked_to_striped=True,
        )

    func_ir, planner = _plan(cohort, arg_types=(types.int32,))
    assert planner.run()
    assert _factory_call_keywords(
        func_ir,
        ir,
        _block_radix_sort.radix_sort_keys,
    ) == [("threads_per_block", "blocked_to_striped")]


@pytest.mark.evidence_for(
    "group.radix_sort_pairs",
    backend="numba_mlir",
    evidence="lowering",
)
def test_common_radix_sort_pairs_use_restricted_pair_factory(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_radix_sort
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def cohort(key, value):
        keys = coop.ThreadData(2, dtype=types.uint32)
        values = coop.ThreadData(2, dtype=types.float32)
        keys[0] = key
        values[0] = value
        return coop.radix_sort_pairs(
            coop.this_block(), keys, values, begin_bit=4, end_bit=10
        )

    func_ir, planner = _plan(cohort, arg_types=(types.uint32, types.float32))
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)
    assert (
        _planned_factories(func_ir, ir)[_block_radix_sort._common_radix_sort_pairs] == 1
    )


@pytest.mark.parametrize(
    ("op_name", "runtime_prefix", "expected_index"),
    [
        ("_common_radix_sort_keys", 1, 2),
        ("_common_radix_sort_pairs", 2, 3),
        ("radix_sort_keys", 1, 2),
        ("radix_sort_pairs", 2, 3),
    ],
)
@pytest.mark.parametrize("begin_value", [8, "runtime"])
def test_single_phase_radix_sort_replaces_omitted_end_with_key_width(
    optional_backend,
    monkeypatch,
    op_name,
    runtime_prefix,
    expected_index,
    begin_value,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    loc = ir.Loc(__file__, 1)
    scope = ir.Scope(None, loc)
    runtime_args = [
        ir.Var(scope, f"payload_{index}", loc) for index in range(runtime_prefix)
    ]
    begin_var = ir.Var(scope, "begin_bit", loc)
    end_var = ir.Var(scope, "end_bit", loc)
    runtime_args.extend((begin_var, end_var))
    resolved = {
        "begin_bit": (
            rewrites._UNRESOLVED if begin_value == "runtime" else begin_value
        ),
        "end_bit": None,
    }
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_resolve_factory_kwarg_value",
        lambda _self, _name, value_ref: resolved[value_ref.name],
    )

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    replacements = rewrite._radix_sort_runtime_constant_replacements(
        op_name=op_name,
        runtime_args=runtime_args,
        runtime_only_kw_vars={"begin_bit": begin_var, "end_bit": end_var},
        factory_kwargs={"dtype": types.int32},
    )

    assert replacements == ((expected_index, 32),)


@pytest.mark.parametrize(
    ("begin_bit", "end_bit", "message"),
    [
        (32, None, r"begin_bit must be < 32"),
        (0, 33, r"end_bit must be <= 32"),
        (8, 8, r"end_bit must be greater than begin_bit"),
    ],
)
def test_single_phase_radix_sort_rejects_invalid_static_ranges_after_inference(
    optional_backend,
    monkeypatch,
    begin_bit,
    end_bit,
    message,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    loc = ir.Loc(__file__, 1)
    scope = ir.Scope(None, loc)
    keys_var = ir.Var(scope, "keys", loc)
    begin_var = ir.Var(scope, "begin_bit", loc)
    end_var = ir.Var(scope, "end_bit", loc)
    resolved = {"begin_bit": begin_bit, "end_bit": end_bit}
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_resolve_factory_kwarg_value",
        lambda _self, _name, value_ref: resolved[value_ref.name],
    )

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    with pytest.raises(rewrites.CoopSinglePhaseRewriteError, match=message):
        rewrite._radix_sort_runtime_constant_replacements(
            op_name="radix_sort_keys",
            runtime_args=[keys_var, begin_var, end_var],
            runtime_only_kw_vars={"begin_bit": begin_var, "end_bit": end_var},
            factory_kwargs={"dtype": types.int32},
        )


def test_common_radix_sort_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return coop.radix_sort_keys(coop.this_block(), keys)

    common_ir, common_planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.radix_sort_keys requires keys to be coop\.ThreadData",
    ):
        common_planner.run()

    def qualified(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return numba_coop.radix_sort_keys(numba_coop.this_block(), keys)

    _qualified_ir, qualified_planner = _plan(
        qualified,
        arg_types=(types.int32,),
    )
    assert qualified_planner.run()
    assert common_ir is not None


@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
def test_common_radix_sort_private_factory_normalizes_portable_bit_ranges(
    optional_backend,
    monkeypatch,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_radix_sort

    monkeypatch.setattr(
        _block_radix_sort,
        "radix_sort_keys",
        lambda **factory_kwargs: factory_kwargs,
    )

    base_kwargs = {
        "dtype": getattr(types, dtype_name),
        "threads_per_block": 64,
        "items_per_thread": 2,
    }
    default_result = _block_radix_sort._common_radix_sort_keys(**base_kwargs)
    begin_only_result = _block_radix_sort._common_radix_sort_keys(
        **base_kwargs,
        begin_bit=8,
        end_bit=None,
    )

    assert "begin_bit" not in default_result
    assert "end_bit" not in default_result
    assert begin_only_result["begin_bit"] == 8
    assert begin_only_result["end_bit"] == getattr(types, dtype_name).bitwidth


@pytest.mark.parametrize("dtype_name", ["boolean", "float32", "int16"])
def test_common_radix_sort_private_factory_rejects_nonportable_key_dtypes(
    optional_backend,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_radix_sort

    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.radix_sort_keys common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        _block_radix_sort._common_radix_sort_keys(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
        )


@pytest.mark.parametrize("descending", [1, "runtime"])
def test_group_radix_sort_requires_a_static_bool_descending(
    optional_backend,
    descending,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    if descending == "runtime":

        def cohort(value, order):
            keys = coop.ThreadData(1, dtype=types.int32)
            keys[0] = value
            return coop.radix_sort_keys(
                coop.this_block(),
                keys,
                descending=order,
            )

        arg_types = (types.int32, types.boolean)
    else:

        def cohort(value):
            keys = coop.ThreadData(1, dtype=types.int32)
            keys[0] = value
            return coop.radix_sort_keys(
                coop.this_block(),
                keys,
                descending=descending,
            )

        arg_types = (types.int32,)

    _func_ir, planner = _plan(cohort, arg_types=arg_types)
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.radix_sort_keys descending must be a compile-time bool",
    ):
        planner.run()


def test_radix_sort_factory_metadata_has_no_decomposer_surface(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites
    from cuda.coop.numba_mlir._block import _block_radix_sort

    for factory in (
        _block_radix_sort.radix_sort_keys,
        _block_radix_sort.radix_sort_keys_descending,
        _block_radix_sort.radix_sort_pairs,
        _block_radix_sort.radix_sort_pairs_descending,
    ):
        assert "decomposer" not in signature(factory).parameters

    for name in (
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_sort_pairs",
        "radix_sort_pairs_descending",
        "_common_radix_sort_keys",
    ):
        assert (
            "decomposer"
            not in rewrites.CoopSinglePhaseRewrite._OP_SPECS[name][
                "allowed_factory_kwargs"
            ]
        )
    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    assert rewrite._is_supported_factory(_block_radix_sort._common_radix_sort_keys)
