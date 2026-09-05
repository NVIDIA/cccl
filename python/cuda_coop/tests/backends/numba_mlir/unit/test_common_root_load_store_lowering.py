# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest


def _plan(function, *, arg_types):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


@pytest.mark.parametrize(
    ("op_name", "algorithm", "expected"),
    [
        ("store", "direct", False),
        ("store", "transpose", True),
        ("store", "warp_transpose", True),
        ("store", "warp_transpose_timesliced", True),
        ("store", 4, True),
        ("warp_store", "direct", False),
        ("warp_store", "transpose", True),
        ("warp_store", 4, False),
    ],
)
def test_group_store_copies_only_cub_mutating_algorithms(
    optional_backend,
    op_name,
    algorithm,
    expected,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    assert (
        CoopSinglePhaseRewrite._store_algorithm_mutates_payload(op_name, algorithm)
        is expected
    )


@pytest.mark.parametrize(
    ("op_name", "operand_dtypes", "operation"),
    [
        ("load", ("complex128", "int32"), "load"),
        ("load", ("int32", "complex128"), "load"),
        ("store", ("complex128", "int32"), "store"),
        ("store", ("int32", "complex128"), "store"),
        ("warp_load", ("complex128", "int32"), "load"),
        ("warp_store", ("int32", "complex128"), "store"),
    ],
)
def test_common_load_store_metadata_validates_both_data_operands(
    optional_backend,
    op_name,
    operand_dtypes,
    operation,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    operands = (object(), object())
    resolved_dtypes = {
        operand: getattr(types, dtype_name)
        for operand, dtype_name in zip(operands, operand_dtypes)
    }
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._resolve_var_dtype = resolved_dtypes.__getitem__
    factory_kwargs = {
        "_common_profile_operation": operation,
        "dtype": types.int32,
    }

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"cuda\.coop\.{operation} common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        rewrite._extract_group_root_match_metadata(
            op_name=op_name,
            runtime_args=operands,
            factory_kwargs=factory_kwargs,
        )


@pytest.mark.evidence_for("group.load", backend="numba_mlir", evidence="lowering")
@pytest.mark.evidence_for("group.store", backend="numba_mlir", evidence="lowering")
@pytest.mark.parametrize("group_kind", ["block", "warp", "logical_warp"])
def test_common_root_load_store_lower_to_scoped_cub_factories(
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
    from cuda.coop.numba_mlir._group_rewrites import (
        _GroupCallPlanner,
        has_group_markers,
    )

    if group_kind == "block":
        group = coop.this_block()
    elif group_kind == "warp":
        group = coop.this_warp()
    else:
        group = coop.this_warp().group_by(8)

    def memory(source, destination):
        output = coop.ThreadData(2, dtype=types.int32)
        loaded = coop.load(
            group,
            source,
            output,
            algorithm="striped",
            valid_items=31,
            oob_default=-1,
            offset=3,
        )
        coop.store(
            group,
            destination,
            loaded,
            algorithm="striped",
            valid_items=31,
            offset=3,
        )

    func_ir = run_frontend(memory)
    assert has_group_markers(func_ir)
    array_type = types.Array(types.int32, 1, "C")
    state = SimpleNamespace(func_ir=func_ir, args=(array_type, array_type))
    assert _GroupCallPlanner(
        state,
        {"block": (64, 1, 1), "grid": (1, 1, 1), "cluster": None},
    ).run()
    assert not has_group_markers(func_ir)

    if group_kind == "block":
        expected = {
            scoped_block.load: (
                "threads_per_block",
                "algorithm",
                "_common_profile_operation",
                "num_valid_items",
                "oob_default",
                "offset",
            ),
            scoped_block.store: (
                "threads_per_block",
                "algorithm",
                "_common_profile_operation",
                "num_valid_items",
                "offset",
                "_group_root_store",
            ),
        }
    else:
        expected = {
            scoped_warp.load: (
                "threads_in_warp",
                "threads_per_block",
                "algorithm",
                "_common_profile_operation",
                "num_valid_items",
                "oob_default",
                "_physical_warp_tile_origin",
                "offset",
            ),
            scoped_warp.store: (
                "threads_in_warp",
                "threads_per_block",
                "algorithm",
                "_common_profile_operation",
                "num_valid_items",
                "_physical_warp_tile_origin",
                "offset",
                "_group_root_store",
            ),
        }

    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    actual = {}
    for block_ir in func_ir.blocks.values():
        for inst in block_ir.body:
            value = getattr(inst, "value", None)
            if not isinstance(value, ir.Expr) or value.op != "call":
                continue
            factory = globals_by_name.get(value.func.name)
            if factory in expected:
                actual[factory] = tuple(name for name, _ in value.kws)
    assert actual == expected

    if group_kind == "logical_warp":
        constants_by_name = {
            inst.target.name: inst.value.value
            for block_ir in func_ir.blocks.values()
            for inst in block_ir.body
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Const)
        }
        widths = []
        for block_ir in func_ir.blocks.values():
            for inst in block_ir.body:
                value = getattr(inst, "value", None)
                if not isinstance(value, ir.Expr) or value.op != "call":
                    continue
                if globals_by_name.get(value.func.name) not in expected:
                    continue
                widths.extend(
                    constants_by_name[var.name]
                    for name, var in value.kws
                    if name == "threads_in_warp"
                )
        assert widths == [8, 8]


@pytest.mark.evidence_for("group.load", backend="numba_mlir", evidence="lowering")
def test_common_load_requires_thread_data_output_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(source):
        output = cuda.local.array(2, types.int32)
        return coop.load(coop.this_block(), source, output)

    array_type = types.Array(types.int32, 1, "C")
    _, common_planner = _plan(common, arg_types=(array_type,))
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.load requires output to be a fixed-size ThreadData "
            r"payload in common V1"
        ),
    ):
        common_planner.run()

    def qualified(source):
        output = cuda.local.array(2, types.int32)
        return numba_coop.load(numba_coop.this_block(), source, output)

    _, qualified_planner = _plan(qualified, arg_types=(array_type,))
    assert qualified_planner.run()


@pytest.mark.evidence_for("group.store", backend="numba_mlir", evidence="lowering")
def test_common_store_rejects_local_arrays_but_qualified_keeps_them(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(destination, value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        coop.store(coop.this_block(), destination, items)

    array_type = types.Array(types.int32, 1, "C")
    _, common_planner = _plan(
        common,
        arg_types=(array_type, types.int32),
    )
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.store accepts only a scalar or fixed-size ThreadData "
            r"value payload in common V1"
        ),
    ):
        common_planner.run()

    def qualified(destination, value):
        items = cuda.local.array(2, types.int32)
        items[0] = value
        items[1] = value + 1
        numba_coop.store(numba_coop.this_block(), destination, items)

    _, qualified_planner = _plan(
        qualified,
        arg_types=(array_type, types.int32),
    )
    assert qualified_planner.run()
