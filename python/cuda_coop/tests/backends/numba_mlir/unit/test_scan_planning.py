# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler planning contracts for the Numba-CUDA-MLIR Scan family."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
        if isinstance(statement, ir.Assign)
        and isinstance(statement.value, (ir.FreeVar, ir.Global))
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


def _kwarg_value(func_ir, call, name):
    from numba_cuda_mlir.numbair_transforms import ir

    variable = dict(call.kws)[name]
    definitions = [
        statement.value
        for block in func_ir.blocks.values()
        for statement in block.body
        if isinstance(statement, ir.Assign) and statement.target.name == variable.name
    ]
    assert len(definitions) == 1
    definition = definitions[0]
    assert isinstance(definition, (ir.Const, ir.FreeVar, ir.Global))
    return definition.value


def test_scan_registers_all_spellings_results_and_provider_abis():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler import _group_scan
    from cuda.coop.numba_mlir._compiler._operations import (
        GroupResultSource,
        StorageABI,
        factory_operation,
        group_primitive,
        rewrite_operation,
    )
    from cuda.coop.numba_mlir._lowering import _scan

    del _group_scan
    operations = {
        "scan",
        "exclusive_scan",
        "inclusive_scan",
        "exclusive_sum",
        "inclusive_sum",
    }
    for operation in operations:
        registration = group_primitive(operation)
        assert registration.results == (GroupResultSource("value", "value"),)

    expected = {
        _scan.block_scan_scalar: ("block_scan_scalar", "block", "block"),
        _scan.block_scan_array: ("block_scan_array", "block", "block"),
        _scan.warp_scan: ("warp_scan", "warp", "warp"),
    }
    for factory, (operation, namespace, scope) in expected.items():
        metadata = factory_operation(factory)
        assert metadata.operation == operation
        assert metadata.namespace == namespace
        assert metadata.storage_abi is StorageABI.LEADING_POINTER
        assert metadata.execution_scope is SynchronizationScope(scope)
        assert metadata.synchronization_scope is SynchronizationScope(scope)

    assert rewrite_operation("block_scan_scalar").runtime_arg_counts == frozenset(
        {1, 2, 3}
    )
    assert rewrite_operation("block_scan_array").runtime_arg_counts == frozenset(
        {2, 3, 4}
    )
    assert rewrite_operation("warp_scan").runtime_arg_counts == frozenset({1, 2, 3, 4})


def test_public_signatures_keep_portable_surface_narrow_and_omit_n6_callbacks():
    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable

    shared = {
        "scan": (
            "group",
            "value",
            "mode",
            "scan_op",
            "initial_value",
            "algorithm",
            "temp_storage",
        ),
        "exclusive_scan": (
            "group",
            "value",
            "scan_op",
            "initial_value",
            "algorithm",
            "temp_storage",
        ),
        "inclusive_scan": (
            "group",
            "value",
            "scan_op",
            "algorithm",
            "temp_storage",
        ),
        "exclusive_sum": ("group", "value", "algorithm", "temp_storage"),
        "inclusive_sum": ("group", "value", "algorithm", "temp_storage"),
    }
    for name, expected in shared.items():
        portable_parameters = tuple(signature(getattr(portable, name)).parameters)
        qualified_parameters = tuple(signature(getattr(qualified, name)).parameters)
        assert portable_parameters == expected
        assert qualified_parameters == (*expected, "valid_items", "aggregate_output")
        assert "prefix_op" not in qualified_parameters
        assert "block_prefix_callback_op" not in qualified_parameters

    package = Path(qualified.__file__).parent
    assert not (package / "_scan_op.py").exists()
    assert not (package / "_stateful_function.py").exists()


@pytest.mark.parametrize(
    ("parameter", "token"),
    (
        ("mode", "inclusive"),
        ("algorithm", "raking"),
        ("scan_op", "max"),
    ),
)
def test_portable_python_entry_point_rejects_non_string_scan_selectors(
    parameter,
    token,
):
    import importlib

    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api.thread_group import this_block

    scan_api = importlib.import_module("cuda.coop._core.api.scan")
    group = this_block()

    with _dispatch._compiler_scope("test.backend"):
        with pytest.raises(TypeError, match=rf"{parameter} must be .*string"):
            scan_api.scan(
                group,
                np.int32(1),
                **{parameter: SimpleNamespace(value=token)},
            )


@pytest.mark.parametrize("api", ("portable", "qualified"))
@pytest.mark.parametrize(
    ("parameter", "token"),
    (
        ("mode", "inclusive"),
        ("algorithm", "raking"),
        ("scan_op", "max"),
    ),
)
def test_scan_planning_rejects_non_string_selectors_before_provider(
    monkeypatch,
    api,
    parameter,
    token,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable
    from cuda.coop.numba_mlir._compiler import _group_scan

    coop = portable if api == "portable" else qualified
    selector = SimpleNamespace(value=token)
    monkeypatch.setattr(
        _group_scan._ScanPlanning,
        "_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "non-string selector reached provider selection"
        ),
    )

    if parameter == "mode":

        def kernel(value):
            return coop.scan(coop.this_block(), value, mode=selector)

    elif parameter == "algorithm":

        def kernel(value):
            return coop.inclusive_sum(coop.this_block(), value, algorithm=selector)

    else:

        def kernel(value):
            return coop.inclusive_scan(coop.this_block(), value, scan_op=selector)

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(TypeError, match=rf"{parameter} must be .*string"):
        planner.run()


def test_private_scan_selector_validation_does_not_unwrap_value_objects():
    from cuda.coop.numba_mlir._lowering import _scan

    cases = (
        (_scan._scan_mode, SimpleNamespace(value="inclusive")),
        (_scan._block_scan_algorithm, SimpleNamespace(value="raking")),
        (_scan.normalize_scan_operation, SimpleNamespace(value="max")),
    )
    for validate, value in cases:
        with pytest.raises(TypeError, match="must be .*string"):
            validate(value)


@pytest.mark.parametrize("api", ("portable", "qualified"))
@pytest.mark.parametrize(
    ("alias", "canonical"),
    (
        (" maximum ", "max"),
        ("multiply", "multiplies"),
        ("BIT-OR", "bit_or"),
    ),
)
def test_shared_scan_operator_aliases_normalize_identically(
    api,
    alias,
    canonical,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable
    from cuda.coop.numba_mlir._lowering import _scan

    coop = portable if api == "portable" else qualified

    def kernel(value):
        return coop.inclusive_scan(coop.this_block(), value, scan_op=alias)

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _scan.block_scan_scalar)
    assert _kwarg_value(func_ir, call, "scan_op") == canonical


@pytest.mark.parametrize(
    ("spelling", "expected_mode", "expected_operator"),
    (
        ("scan", "inclusive", "max"),
        ("exclusive_scan", "exclusive", "max"),
        ("inclusive_scan", "inclusive", "max"),
        ("exclusive_sum", "exclusive", None),
        ("inclusive_sum", "inclusive", None),
    ),
)
def test_all_five_spellings_plan_through_one_block_provider(
    spelling: str,
    expected_mode: str,
    expected_operator: str | None,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _scan

    if spelling == "scan":

        def kernel(value):
            return coop.scan(coop.this_block(), value, mode="inclusive", scan_op="max")

    elif spelling == "exclusive_scan":

        def kernel(value):
            return coop.exclusive_scan(
                coop.this_block(), value, scan_op="max", initial_value=-17
            )

    elif spelling == "inclusive_scan":

        def kernel(value):
            return coop.inclusive_scan(coop.this_block(), value, scan_op="max")

    elif spelling == "exclusive_sum":

        def kernel(value):
            return coop.exclusive_sum(coop.this_block(), value)

    else:

        def kernel(value):
            return coop.inclusive_sum(coop.this_block(), value)

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _scan.block_scan_scalar)
    assert len(call.args) == 1
    assert call.args[0].name == "value"
    assert _kwarg_value(func_ir, call, "mode") == expected_mode
    assert _kwarg_value(func_ir, call, "scan_op") == expected_operator


@pytest.mark.parametrize(
    "dtype_name",
    (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ),
)
def test_scan_planning_accepts_every_supported_numeric_dtype(dtype_name: str):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.inclusive_sum(coop.this_block(), value)

    _, planner = _plan(kernel, arg_types=(getattr(types, dtype_name),))
    assert planner.run()


@pytest.mark.parametrize("dtype_name", ("boolean", "float16", "complex64"))
def test_scan_planning_rejects_unsupported_payload_dtypes(dtype_name: str):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.inclusive_sum(coop.this_block(), value)

    _, planner = _plan(kernel, arg_types=(getattr(types, dtype_name),))
    with pytest.raises(TypeError, match="supports value dtypes"):
        planner.run()


def test_block_thread_data_and_local_array_plan_out_of_place_with_storage():
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _scan

    def thread_data_kernel(value):
        items = coop.ThreadData(2, dtype=types.int32)
        items[0] = value
        items[1] = types.int32(value + 1)
        return coop.inclusive_sum(coop.this_block(), items, algorithm="raking_memoize")

    func_ir, planner = _plan(thread_data_kernel, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _scan.block_scan_array)
    assert len(call.args) == 2
    assert call.args[0].name != call.args[1].name
    assert _kwarg_value(func_ir, call, "items_per_thread") == 2
    assert _kwarg_value(func_ir, call, "value_kind") == "array"

    def local_array_kernel(value):
        items = cuda.local.array(2, dtype=types.int32)
        aggregate = cuda.local.array(1, dtype=types.int32)
        items[0] = value
        items[1] = value + 1
        return coop.exclusive_scan(
            coop.this_block(),
            items,
            scan_op=np.maximum,
            initial_value=-17,
            aggregate_output=aggregate,
            temp_storage=coop.TempStorage(sharing="shared"),
        )

    func_ir, planner = _plan(local_array_kernel, arg_types=(types.int32,))
    assert planner.run()
    call = _provider_call(func_ir, _scan.block_scan_array)
    assert len(call.args) == 2
    assert {name for name, _ in call.kws} >= {
        "block_aggregate",
        "initial_value",
        "temp_storage",
    }


@pytest.mark.parametrize("qualified", (False, True), ids=("portable", "qualified"))
def test_untyped_thread_data_scan_infers_writes_and_chains_into_store(qualified):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as portable_coop
    from cuda.coop.numba_mlir._lowering import _scan

    coop = qualified_coop if qualified else portable_coop

    def kernel(value, destination):
        items = coop.ThreadData(2)
        items[0] = value
        items[1] = types.int32(value + 1)
        scanned = coop.inclusive_sum(coop.this_block(), items)
        coop.store(coop.this_block(), destination, scanned)
        return scanned

    func_ir, planner = _plan(
        kernel,
        arg_types=(types.int32, types.Array(types.int32, 1, "C")),
    )
    assert planner.run()
    call = _provider_call(func_ir, _scan.block_scan_array)
    assert _kwarg_value(func_ir, call, "dtype") is types.int32
    assert _kwarg_value(func_ir, call, "items_per_thread") == 2


def test_warp_planning_preserves_width_runtime_prefix_and_aggregate_position():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._lowering import _scan

    def kernel(value, valid_items):
        aggregate = coop.ThreadData(1, dtype=types.int32)
        return coop.inclusive_scan(
            coop.this_warp().group_by(8),
            value,
            scan_op="bit_or",
            valid_items=valid_items,
            aggregate_output=aggregate,
        )

    func_ir, planner = _plan(
        kernel,
        arg_types=(types.int32, types.uint32),
    )
    assert planner.run()
    call = _provider_call(func_ir, _scan.warp_scan)
    assert _kwarg_value(func_ir, call, "threads_in_warp") == 8
    assert "scan_valid_items_i64" in dict(call.kws)["valid_items"].name
    assert "warp_aggregate" in dict(call.kws)
    assert any(target is types.int64 for target, _ in _planned_factory_calls(func_ir))


def test_qualified_callback_plans_for_block_and_warp_but_portable_rejects_it():
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as qualified
    from cuda import coop as portable
    from cuda.coop.numba_mlir._lowering import _scan

    def combine(left, right):
        return left if left > right else right  # noqa: FURB136

    def block_kernel(value):
        return qualified.inclusive_scan(qualified.this_block(), value, scan_op=combine)

    func_ir, planner = _plan(block_kernel, arg_types=(types.int32,))
    assert planner.run()
    assert _provider_call(func_ir, _scan.block_scan_scalar)

    def warp_kernel(value):
        return qualified.exclusive_scan(
            qualified.this_warp(), value, scan_op=combine, initial_value=-17
        )

    func_ir, planner = _plan(warp_kernel, arg_types=(types.int32,))
    assert planner.run()
    assert _provider_call(func_ir, _scan.warp_scan)

    def portable_kernel(value):
        return portable.inclusive_scan(portable.this_block(), value, scan_op=combine)

    _, planner = _plan(portable_kernel, arg_types=(types.int32,))
    with pytest.raises(NotImplementedError, match="built-in operators only"):
        planner.run()


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("non_sum_without_initial", "require.*initial_value"),
        ("inclusive_initial", "do not accept initial_value"),
        ("runtime_initial_dtype", "does not match value dtype"),
        ("aggregate_extent", "exactly one item"),
        ("aggregate_dtype", "does not match value dtype"),
        ("block_valid", "WarpScan"),
        ("warp_array", "one scalar value per lane"),
    ),
)
def test_invalid_scan_shapes_and_initials_fail_during_planning(case: str, match: str):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    if case == "non_sum_without_initial":

        def kernel(value):
            return coop.exclusive_scan(coop.this_block(), value, scan_op="max")

        arg_types = (types.int32,)
    elif case == "inclusive_initial":

        def kernel(value):
            return coop.scan(
                coop.this_block(),
                value,
                mode="inclusive",
                initial_value=0,
            )

        arg_types = (types.int32,)
    elif case == "runtime_initial_dtype":

        def kernel(value, initial):
            return coop.exclusive_scan(coop.this_block(), value, initial_value=initial)

        arg_types = (types.int32, types.int64)
    elif case == "aggregate_extent":

        def kernel(value):
            aggregate = coop.ThreadData(2, dtype=types.int32)
            return coop.inclusive_sum(
                coop.this_block(), value, aggregate_output=aggregate
            )

        arg_types = (types.int32,)
    elif case == "aggregate_dtype":

        def kernel(value):
            aggregate = coop.ThreadData(1, dtype=types.float32)
            return coop.inclusive_sum(
                coop.this_block(), value, aggregate_output=aggregate
            )

        arg_types = (types.int32,)
    elif case == "block_valid":

        def kernel(value):
            return coop.inclusive_sum(coop.this_block(), value, valid_items=17)

        arg_types = (types.int32,)
    else:

        def kernel(value):
            items = coop.ThreadData(2, dtype=types.int32)
            items[0] = value
            items[1] = value
            return coop.inclusive_sum(coop.this_warp(), items)

        arg_types = (types.int32,)

    _, planner = _plan(kernel, arg_types=arg_types)
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=match):
        planner.run()


@pytest.mark.parametrize("initial", (128, -129, 1.5, np.int64(1)))
def test_static_initial_must_convert_exactly_to_payload_dtype(initial):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.exclusive_scan(coop.this_block(), value, initial_value=initial)

    _, planner = _plan(kernel, arg_types=(types.int8,))
    with pytest.raises((TypeError, OverflowError, ValueError), match="initial_value"):
        planner.run()


@pytest.mark.parametrize(
    ("dtype", "initial"),
    (
        ("int8", -128),
        ("int8", 127),
        ("int8", np.int8(7)),
        ("float32", 2),
        ("float32", 1.5),
        ("float32", np.float32(1.5)),
    ),
)
def test_static_initial_accepts_literal_boundaries_and_exact_typed_scalars(
    dtype: str,
    initial,
):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.exclusive_scan(coop.this_block(), value, initial_value=initial)

    _, planner = _plan(kernel, arg_types=(getattr(types, dtype),))
    assert planner.run()


@pytest.mark.parametrize("dtype", ("boolean", "float32", "uint64"))
def test_runtime_valid_items_rejects_invalid_dtype_before_provider(dtype: str):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value, valid_items):
        return coop.inclusive_sum(
            coop.this_warp().group_by(8), value, valid_items=valid_items
        )

    _, planner = _plan(
        kernel,
        arg_types=(types.int32, getattr(types, dtype)),
    )
    with pytest.raises(TypeError, match="must be an integer|unsigned integer"):
        planner.run()


@pytest.mark.parametrize("valid_items", (True, 0, 9))
def test_static_valid_items_rejects_bool_and_out_of_range_values(valid_items):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        return coop.inclusive_sum(
            coop.this_warp().group_by(8), value, valid_items=valid_items
        )

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        planner.run()
