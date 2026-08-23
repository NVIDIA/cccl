# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest


def _less(lhs, rhs):
    return lhs < rhs


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


def test_warp_merge_sort_factories_expose_partial_tile_abi(
    optional_backend,
    monkeypatch,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    monkeypatch.setattr(
        _warp_merge_sort,
        "make_invocable_from_specialization",
        lambda specialization, **_kwargs: specialization,
    )

    keys = _warp_merge_sort.warp_merge_sort_keys(
        types.int32,
        2,
        _less,
        valid_items=True,
        oob_default=True,
    )
    pairs = _warp_merge_sort.warp_merge_sort_pairs(
        types.int32,
        types.float32,
        2,
        _less,
        valid_items=True,
        oob_default=True,
    )

    assert [type(parameter).__name__ for parameter in keys.parameters[0]] == [
        "Pointer",
        "Array",
        "StatelessOperator",
        "Value",
        "Reference",
    ]
    assert [type(parameter).__name__ for parameter in pairs.parameters[0]] == [
        "Pointer",
        "Array",
        "Array",
        "StatelessOperator",
        "Value",
        "Reference",
    ]


@pytest.mark.parametrize(
    ("factory_name", "factory_kwargs"),
    [
        (
            "warp_merge_sort_keys",
            {"dtype": "int32", "items_per_thread": 2, "compare_op": _less},
        ),
        (
            "warp_merge_sort_pairs",
            {
                "keys": "int32",
                "values": "float32",
                "items_per_thread": 2,
                "compare_op": _less,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("valid_items", "oob_default"),
    [(7, None), (None, -1)],
)
def test_warp_merge_sort_factories_require_complete_partial_tile_arguments(
    optional_backend,
    factory_name,
    factory_kwargs,
    valid_items,
    oob_default,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    with pytest.raises(
        ValueError,
        match="valid_items and oob_default must be provided together",
    ):
        getattr(_warp_merge_sort, factory_name)(
            **factory_kwargs,
            valid_items=valid_items,
            oob_default=oob_default,
        )


def test_single_phase_warp_merge_sort_specs_accept_only_full_or_paired_partial_abi(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    specs = CoopSinglePhaseRewrite._OP_SPECS
    assert specs["warp_merge_sort_keys"]["runtime_arg_counts"] == {1, 3}
    assert specs["warp_merge_sort_pairs"]["runtime_arg_counts"] == {2, 4}
    for name in ("warp_merge_sort_keys", "warp_merge_sort_pairs"):
        assert specs[name]["runtime_factory_kwargs"] == (
            "valid_items",
            "oob_default",
        )
        assert specs[name]["runtime_factory_kw_prerequisites"] == {
            "oob_default": "valid_items"
        }
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    assert rewrite._is_supported_factory(_block_merge_sort._common_merge_sort_keys)
    assert rewrite._is_supported_factory(_block_merge_sort._common_merge_sort_pairs)
    assert rewrite._is_supported_factory(_warp_merge_sort._common_warp_merge_sort_keys)
    assert rewrite._is_supported_factory(_warp_merge_sort._common_warp_merge_sort_pairs)


@pytest.mark.parametrize(
    ("qualified", "pairs"),
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.evidence_for(
    "group.merge_sort_pairs",
    backend="numba_mlir",
    evidence="lowering",
)
def test_physical_warp_partial_merge_sort_lowers_to_expected_factory(
    optional_backend,
    qualified,
    pairs,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    api = numba_coop if qualified else coop

    if pairs:

        def cohort(value, valid_items, oob_default):
            group = api.this_warp()
            keys = api.ThreadData(2, dtype=types.int32)
            keys[0] = value
            keys[1] = value + 1
            values = api.ThreadData(2, dtype=types.int32)
            values[0] = value + 10
            values[1] = value + 11
            return api.merge_sort_pairs(
                group,
                keys,
                values,
                valid_items=valid_items,
                oob_default=oob_default,
            )

    else:

        def cohort(value, valid_items, oob_default):
            group = api.this_warp()
            keys = api.ThreadData(2, dtype=types.int32)
            keys[0] = value
            keys[1] = value + 1
            return api.merge_sort_keys(
                group,
                keys,
                valid_items=valid_items,
                oob_default=oob_default,
            )

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(
        func_ir=func_ir,
        args=(types.int32, types.int32, types.int32),
    )
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    if pairs:
        expected = (
            _warp_merge_sort.warp_merge_sort_pairs
            if qualified
            else _warp_merge_sort._common_warp_merge_sort_pairs
        )
    elif qualified:
        expected = _warp_merge_sort.warp_merge_sort_keys
    else:
        expected = _warp_merge_sort._common_warp_merge_sort_keys
    assert _planned_factories(func_ir, ir)[expected] == 1


@pytest.mark.parametrize("qualified", [False, True], ids=["common", "qualified"])
def test_logical_warp_merge_sort_lowers_with_exact_width(
    optional_backend,
    qualified,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    api = numba_coop if qualified else coop
    group = api.this_warp().group_by(8)

    def cohort(value):
        keys = api.ThreadData(2, dtype=types.int32)
        keys[0] = value
        keys[1] = value + 1
        return api.merge_sort_keys(group, keys)

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)
    expected = (
        _warp_merge_sort.warp_merge_sort_keys
        if qualified
        else _warp_merge_sort._common_warp_merge_sort_keys
    )
    assert _planned_factories(func_ir, ir)[expected] == 1

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
        and globals_by_name.get(inst.value.func.name) is expected
    ]
    assert len(calls) == 1
    width_var = dict(calls[0].kws)["threads_in_warp"]
    assert constants_by_name[width_var.name] == 8


@pytest.mark.parametrize("group_kind", ["block", "warp"])
@pytest.mark.parametrize("partial", [False, True])
def test_qualified_scalar_merge_sort_boxes_copies_and_projects_result(
    optional_backend,
    group_kind,
    partial,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    if group_kind == "block":
        group_factory = numba_coop.this_block
        expected_factory = _block_merge_sort.merge_sort_keys
        valid_items = 63
    else:
        group_factory = numba_coop.this_warp
        expected_factory = _warp_merge_sort.warp_merge_sort_keys
        valid_items = 31

    if partial:

        def cohort(value):
            return numba_coop.merge_sort_keys(
                group_factory(),
                value,
                valid_items=valid_items,
                oob_default=2_147_483_647,
            )

    else:

        def cohort(value):
            return numba_coop.merge_sort_keys(group_factory(), value)

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
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


@pytest.mark.parametrize("group_kind", ["block", "warp"])
@pytest.mark.evidence_for(
    "group.merge_sort_pairs",
    backend="numba_mlir",
    evidence="lowering",
)
def test_qualified_scalar_merge_sort_pairs_box_and_project_both_payloads(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    if group_kind == "block":
        group_factory = numba_coop.this_block
        expected_factory = _block_merge_sort.merge_sort_pairs
    else:
        group_factory = numba_coop.this_warp
        expected_factory = _warp_merge_sort.warp_merge_sort_pairs

    def cohort(key, value):
        return numba_coop.merge_sort_pairs(group_factory(), key, value)

    func_ir = run_frontend(cohort)
    assert has_group_markers(func_ir)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32, types.float32))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)
    assert _planned_factories(func_ir, ir)[expected_factory] == 1

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


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize(
    ("valid_items", "oob_default"),
    [(7, None), (None, -1)],
)
def test_group_merge_sort_requires_complete_partial_tile_arguments(
    optional_backend,
    qualified,
    valid_items,
    oob_default,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    api = numba_coop if qualified else coop

    def cohort(value):
        keys = api.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return api.merge_sort_keys(
            api.this_warp(),
            keys,
            valid_items=valid_items,
            oob_default=oob_default,
        )

    func_ir = run_frontend(cohort)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    with pytest.raises(
        ValueError,
        match=(
            r"cuda\.coop\.numba_mlir\.merge_sort_keys valid_items and "
            r"oob_default must be provided together"
        ),
    ):
        _GroupCallPlanner(
            state,
            {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()


@pytest.mark.parametrize(
    ("valid_items", "error_type", "message"),
    [
        (True, TypeError, r"valid_items must be an integer, not bool"),
        (-1, ValueError, r"static valid_items must be in \[0, 64\]"),
        (65, ValueError, r"static valid_items must be in \[0, 64\]"),
    ],
)
def test_physical_warp_merge_sort_rejects_invalid_static_valid_items(
    optional_backend,
    valid_items,
    error_type,
    message,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    def cohort(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        keys[1] = value + 1
        return coop.merge_sort_keys(
            coop.this_warp(),
            keys,
            valid_items=valid_items,
            oob_default=0,
        )

    func_ir = run_frontend(cohort)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    with pytest.raises(error_type, match=message):
        _GroupCallPlanner(
            state,
            {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()


def test_common_merge_sort_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    def common(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return coop.merge_sort_keys(coop.this_warp(), keys)

    common_ir = run_frontend(common)
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.merge_sort_keys requires keys to be coop\.ThreadData",
    ):
        _GroupCallPlanner(
            SimpleNamespace(func_ir=common_ir, args=(types.int32,)),
            {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
        ).run()

    def qualified(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return numba_coop.merge_sort_keys(numba_coop.this_warp(), keys)

    qualified_ir = run_frontend(qualified)
    assert _GroupCallPlanner(
        SimpleNamespace(func_ir=qualified_ir, args=(types.int32,)),
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()


def test_common_merge_sort_accepts_fresh_thread_data_from_histogram(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numbair_transforms import ir

    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    def cohort(value):
        group = coop.this_block()
        samples = coop.ThreadData(2, dtype=types.int32)
        samples[0] = value
        samples[1] = value + 1
        counters = coop.histogram(
            group,
            samples,
            bins=64,
            bins_per_thread=1,
        )
        return coop.merge_sort_keys(group, counters)

    func_ir = run_frontend(cohort)
    state = SimpleNamespace(func_ir=func_ir, args=(types.int32,))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)
    assert (
        _planned_factories(func_ir, ir)[_block_merge_sort._common_merge_sort_keys] == 1
    )


@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_common_merge_sort_private_factories_accept_only_portable_key_dtypes(
    optional_backend,
    monkeypatch,
    dtype_name,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    if group_kind == "block":
        module = _block_merge_sort
        private_factory = module._common_merge_sort_keys
        public_name = "merge_sort_keys"
        kwargs = {"threads_per_block": 32}
    else:
        module = _warp_merge_sort
        private_factory = module._common_warp_merge_sort_keys
        public_name = "warp_merge_sort_keys"
        kwargs = {}

    public_factory = getattr(module, public_name)
    assert "_common_root" not in signature(public_factory).parameters
    monkeypatch.setattr(module, public_name, lambda **factory_kwargs: factory_kwargs)
    result = private_factory(
        dtype=getattr(types, dtype_name),
        items_per_thread=2,
        compare_op=_less,
        **kwargs,
    )
    assert result["dtype"] is getattr(types, dtype_name)


@pytest.mark.parametrize("group_kind", ["block", "warp"])
def test_common_merge_sort_private_factories_reject_nonportable_key_dtypes(
    optional_backend,
    group_kind,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_merge_sort
    from cuda.coop.numba_mlir._warp import _warp_merge_sort

    if group_kind == "block":
        private_factory = _block_merge_sort._common_merge_sort_keys
        kwargs = {"threads_per_block": 32}
    else:
        private_factory = _warp_merge_sort._common_warp_merge_sort_keys
        kwargs = {}

    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.merge_sort_keys common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        private_factory(
            dtype=types.float32,
            items_per_thread=2,
            compare_op=_less,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("key_dtype_name", "sentinel", "error_type", "diagnostic"),
    [
        (
            "int32",
            1.5,
            TypeError,
            r"oob_default must have the same integer dtype as keys \(int32\); got float",
        ),
        (
            "int32",
            np.int64(0),
            TypeError,
            r"oob_default must have the same integer dtype as keys \(int32\); got int64",
        ),
        (
            "int32",
            2_147_483_648,
            ValueError,
            r"oob_default=2147483648 is not representable in keys dtype int32",
        ),
        (
            "uint32",
            -1,
            ValueError,
            r"oob_default=-1 is not representable in keys dtype uint32",
        ),
    ],
)
@pytest.mark.parametrize(
    "op_name",
    ["_common_merge_sort_keys", "_common_warp_merge_sort_keys"],
)
def test_common_merge_sort_single_phase_rejects_lossy_static_sentinels(
    optional_backend,
    monkeypatch,
    key_dtype_name,
    sentinel,
    error_type,
    diagnostic,
    op_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    monkeypatch.setattr(
        rewrite,
        "_resolve_factory_kwarg_value",
        lambda _name, _value: sentinel,
    )
    with pytest.raises(
        rewrites.CoopSinglePhaseRewriteError,
        match=diagnostic,
    ) as exc_info:
        rewrite._validate_common_merge_sort_runtime_controls(
            op_name=op_name,
            runtime_factory_control_vars={"oob_default": object()},
            factory_kwargs={"dtype": getattr(types, key_dtype_name)},
        )
    assert isinstance(exc_info.value.__cause__, error_type)


@pytest.mark.parametrize(
    ("sentinel_dtype_name", "error"),
    [("int32", False), ("int64", True)],
)
def test_common_merge_sort_single_phase_validates_dynamic_sentinel_dtype(
    optional_backend,
    monkeypatch,
    sentinel_dtype_name,
    error,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    monkeypatch.setattr(
        rewrite,
        "_resolve_factory_kwarg_value",
        lambda _name, _value: rewrites._UNRESOLVED,
    )
    monkeypatch.setattr(
        rewrite,
        "_resolve_var_dtype",
        lambda _value: getattr(types, sentinel_dtype_name),
    )

    def call():
        rewrite._validate_common_merge_sort_runtime_controls(
            op_name="_common_merge_sort_keys",
            runtime_factory_control_vars={"oob_default": object()},
            factory_kwargs={"dtype": types.int32},
        )

    if not error:
        call()
        return
    with pytest.raises(
        rewrites.CoopSinglePhaseRewriteError,
        match=(
            r"oob_default must have the same integer dtype as keys \(int32\); "
            r"got int64"
        ),
    ):
        call()


def test_qualified_merge_sort_does_not_apply_common_sentinel_validation(
    optional_backend,
    monkeypatch,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    monkeypatch.setattr(
        rewrite,
        "_resolve_factory_kwarg_value",
        lambda _name, _value: pytest.fail(
            "qualified validation must not inspect value"
        ),
    )
    rewrite._validate_common_merge_sort_runtime_controls(
        op_name="merge_sort_keys",
        runtime_factory_control_vars={"oob_default": object()},
        factory_kwargs={"dtype": types.int32},
    )
