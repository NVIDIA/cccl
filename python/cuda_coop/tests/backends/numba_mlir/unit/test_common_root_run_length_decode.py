# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import pytest

_BLOCK = (8, 4, 2)


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


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="numba_mlir",
    evidence="lowering",
)
def test_common_and_qualified_run_length_decode_use_provenance_marked_parents(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_run_length_decode
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def cohort(value, length, window_offset):
        common_values = coop.ThreadData(2, dtype=types.uint8)
        common_lengths = coop.ThreadData(2, dtype=types.uint64)
        qualified_values = numba_coop.ThreadData(2, dtype=types.uint8)
        qualified_lengths = numba_coop.ThreadData(2, dtype=types.uint64)
        for index in range(2):
            common_values[index] = value
            common_lengths[index] = length
            qualified_values[index] = value
            qualified_lengths[index] = length
        common = coop.run_length_decode(
            coop.this_block(),
            common_values,
            common_lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=window_offset,
        )
        qualified = numba_coop.run_length_decode(
            numba_coop.this_block(),
            qualified_values,
            qualified_lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=window_offset,
        )
        return common[0], qualified[0]

    func_ir, planner = _plan(
        cohort,
        arg_types=(types.uint8, types.uint64, types.uint64),
    )
    assert has_group_markers(func_ir)
    assert planner.run()
    assert not has_group_markers(func_ir)

    factories = _planned_factories(func_ir, ir)
    assert factories[_block_run_length_decode._common_run_length] == 1
    assert factories[_block_run_length_decode._qualified_group_run_length] == 1


def test_common_run_length_decode_requires_thread_data_inputs(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value, length):
        run_values = cuda.local.array(2, types.int32)
        run_lengths = cuda.local.array(2, types.uint32)
        run_values[0] = value
        run_lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=2,
        )

    _common_ir, common_planner = _plan(
        common,
        arg_types=(types.int32, types.uint32),
    )
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.run_length_decode requires run_values to be "
            r"coop\.ThreadData"
        ),
    ):
        common_planner.run()

    def qualified(value, length):
        run_values = cuda.local.array(2, types.float32)
        run_lengths = cuda.local.array(2, types.uint32)
        run_values[0] = value
        run_lengths[0] = length
        return numba_coop.run_length_decode(
            numba_coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=2,
        )

    _qualified_ir, qualified_planner = _plan(
        qualified,
        arg_types=(types.float32, types.uint32),
    )
    assert qualified_planner.run()


def test_common_run_length_decode_requires_matching_input_extents(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def common(value, length):
        run_values = coop.ThreadData(2, dtype=types.int32)
        run_lengths = coop.ThreadData(1, dtype=types.uint32)
        run_values[0] = value
        run_lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=2,
        )

    _func_ir, planner = _plan(
        common,
        arg_types=(types.int32, types.uint32),
    )
    with pytest.raises(
        ValueError,
        match="run_values and run_lengths must have the same items_per_thread",
    ):
        planner.run()


@pytest.mark.parametrize(
    ("common", "offset", "error_type", "message"),
    [
        (True, -1, ValueError, "must be non-negative"),
        (False, -1, ValueError, "must be non-negative"),
        (True, 1.5, TypeError, "must be an integer"),
        (False, True, TypeError, "must be an integer"),
    ],
)
def test_group_first_run_length_decode_rejects_invalid_static_offsets(
    optional_backend,
    common,
    offset,
    error_type,
    message,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    api = coop if common else numba_coop

    def invalid(value, length):
        run_values = api.ThreadData(1, dtype=types.int32)
        run_lengths = api.ThreadData(1, dtype=types.uint32)
        run_values[0] = value
        run_lengths[0] = length
        return api.run_length_decode(
            api.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=offset,
        )

    _func_ir, planner = _plan(
        invalid,
        arg_types=(types.int32, types.uint32),
    )
    scope = r"cuda\.coop" if common else r"cuda\.coop\.numba_mlir"
    with pytest.raises(
        error_type,
        match=rf"{scope}\.run_length_decode decoded_window_offset {message}",
    ):
        planner.run()


@pytest.mark.parametrize(
    ("decoded_items_per_thread", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_group_first_run_length_decode_splits_static_extent_diagnostics(
    optional_backend,
    decoded_items_per_thread,
    error_type,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def invalid(value, length):
        run_values = coop.ThreadData(1, dtype=types.int32)
        run_lengths = coop.ThreadData(1, dtype=types.uint32)
        run_values[0] = value
        run_lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=decoded_items_per_thread,
        )

    _func_ir, planner = _plan(
        invalid,
        arg_types=(types.int32, types.uint32),
    )
    with pytest.raises(
        error_type,
        match=(
            r"cuda\.coop\.run_length_decode decoded_items_per_thread must be "
            r"a compile-time positive integer"
        ),
    ):
        planner.run()


@pytest.mark.parametrize(
    ("run_values_dtype", "run_lengths_dtype"),
    [
        ("uint8", "uint64"),
        ("int32", "int32"),
        ("uint32", "uint32"),
        ("int64", "int64"),
        ("uint64", "uint32"),
    ],
)
def test_common_run_length_decode_accepts_portable_dtype_pairs(
    optional_backend,
    run_values_dtype,
    run_lengths_dtype,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._common import (
        _validate_common_run_length_decode_dtypes,
    )

    assert _validate_common_run_length_decode_dtypes(
        getattr(types, run_values_dtype),
        getattr(types, run_lengths_dtype),
    ) == (
        getattr(types, run_values_dtype),
        getattr(types, run_lengths_dtype),
    )


@pytest.mark.parametrize(
    ("run_values_dtype", "run_lengths_dtype", "parameter", "supported"),
    [
        (
            "float32",
            "uint32",
            "run_values",
            "uint8, int32, uint32, int64, uint64",
        ),
        (
            "int32",
            "uint16",
            "run_lengths",
            "int32, uint32, int64, uint64",
        ),
        (
            "int16",
            "uint32",
            "run_values",
            "uint8, int32, uint32, int64, uint64",
        ),
    ],
)
def test_common_run_length_decode_rejects_nonportable_dtype_pairs(
    optional_backend,
    run_values_dtype,
    run_lengths_dtype,
    parameter,
    supported,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._common import (
        _validate_common_run_length_decode_dtypes,
    )

    expected = (
        f"cuda.coop.run_length_decode common V1 supports {parameter} dtypes "
        f"{supported}; use a backend-qualified import for backend-specific "
        f"{parameter} dtypes"
    )
    with pytest.raises(TypeError) as exc_info:
        _validate_common_run_length_decode_dtypes(
            getattr(types, run_values_dtype),
            getattr(types, run_lengths_dtype),
        )
    assert str(exc_info.value) == expected
