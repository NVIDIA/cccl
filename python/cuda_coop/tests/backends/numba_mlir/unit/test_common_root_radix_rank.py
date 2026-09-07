# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
from collections import Counter
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


@pytest.mark.evidence_for("group.radix_rank", backend="numba_mlir", evidence="lowering")
def test_common_and_qualified_radix_rank_lower_through_static_block_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop
    from cuda.coop.numba_mlir._block import _block_radix_rank

    def cohort(value):
        common_keys = coop.ThreadData(2, dtype=types.int32)
        qualified_keys = numba_coop.ThreadData(2, dtype=types.int32)
        common_keys[0] = value
        common_keys[1] = -value
        qualified_keys[0] = value
        qualified_keys[1] = -value
        common = coop.radix_rank(
            coop.this_block(),
            common_keys,
            begin_bit=28,
            radix_bits=4,
        )
        qualified = numba_coop.radix_rank(
            numba_coop.this_block(),
            qualified_keys,
            begin_bit=28,
            end_bit=32,
        )
        return common[0], qualified[0]

    func_ir, planner = _plan(cohort, arg_types=(types.int32,))
    assert planner.run()

    factories = _planned_factories(func_ir, ir)
    assert factories[_block_radix_rank._common_radix_rank] == 1
    assert factories[numba_coop._block.radix_rank] == 1


def test_common_radix_rank_requires_thread_data_but_qualified_keeps_local_arrays(
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
        return coop.radix_rank(coop.this_block(), keys)

    _common_ir, common_planner = _plan(common, arg_types=(types.int32,))
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.radix_rank requires keys to be coop\.ThreadData",
    ):
        common_planner.run()

    def qualified(value):
        keys = cuda.local.array(2, types.int32)
        keys[0] = value
        return numba_coop.radix_rank(numba_coop.this_block(), keys)

    _qualified_ir, qualified_planner = _plan(
        qualified,
        arg_types=(types.int32,),
    )
    assert qualified_planner.run()


@pytest.mark.parametrize("dtype_name", ["int32", "uint32", "int64", "uint64"])
def test_common_radix_rank_factory_accepts_portable_dtypes_and_twiddles_signed(
    optional_backend,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types as backend_types
    from cuda.coop.numba_mlir._block import _block_radix_rank
    from cuda.coop.numba_mlir._types import collect_specializations

    dtype = getattr(types, dtype_name)
    begin_bit = dtype.bitwidth - 4
    with collect_specializations() as collected:
        specialization = _block_radix_rank._common_radix_rank(
            dtype=dtype,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=begin_bit,
            end_bit=dtype.bitwidth,
        )

    assert specialization is collected[0][0]
    keys = specialization.parameters[0][1]
    if dtype_name.startswith("int"):
        assert isinstance(keys, backend_types.TransformedArray)
        assert keys.value_dtype == dtype
        assert keys.target_dtype == getattr(types, f"u{dtype_name}")
    else:
        assert type(keys) is backend_types.Array
        assert keys.value_dtype == dtype


@pytest.mark.parametrize("dtype_name", ["boolean", "float32", "int16"])
def test_common_radix_rank_factory_rejects_nonportable_key_dtypes(
    optional_backend,
    dtype_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_radix_rank

    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.radix_rank common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        _block_radix_rank._common_radix_rank(
            dtype=getattr(types, dtype_name),
            threads_per_block=64,
            items_per_thread=2,
        )


@pytest.mark.parametrize("common", [False, True])
def test_group_first_radix_rank_default_window_does_not_silently_clamp(
    optional_backend,
    common,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._block import _block_radix_rank

    factory = (
        _block_radix_rank._common_radix_rank if common else _block_radix_rank.radix_rank
    )
    with pytest.raises(ValueError, match="end_bit must not exceed the dtype bit width"):
        factory(
            dtype=types.int32,
            threads_per_block=64,
            items_per_thread=2,
            begin_bit=30,
        )


def test_common_radix_rank_planner_uses_common_scope_for_control_diagnostics(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def invalid_begin(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(coop.this_block(), keys, begin_bit=True)

    def invalid_end(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(coop.this_block(), keys, begin_bit=8, end_bit=8)

    def invalid_width(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(coop.this_block(), keys, radix_bits=9)

    def mismatched_width(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(
            coop.this_block(),
            keys,
            end_bit=8,
            radix_bits=4,
        )

    def invalid_order(value):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = value
        return coop.radix_rank(coop.this_block(), keys, descending=1)

    cases = (
        (invalid_begin, "begin_bit must be a compile-time integer"),
        (invalid_end, "end_bit must be greater than begin_bit"),
        (invalid_width, "bit width must be <= 8"),
        (mismatched_width, "radix_bits must match end_bit - begin_bit"),
        (invalid_order, "descending must be a compile-time bool"),
    )
    for cohort, message in cases:
        _func_ir, planner = _plan(cohort, arg_types=(types.int32,))
        with pytest.raises(
            (TypeError, ValueError),
            match=rf"cuda\.coop\.radix_rank {message}",
        ):
            planner.run()


@pytest.mark.parametrize(
    ("op_name", "scope_name"),
    [
        ("_common_radix_rank", "cuda.coop"),
        ("radix_rank", "cuda.coop.numba_mlir"),
    ],
)
def test_single_phase_radix_rank_range_diagnostics_keep_api_scope(
    optional_backend,
    op_name,
    scope_name,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    with pytest.raises(
        rewrites.CoopSinglePhaseRewriteError,
        match=rf"{re.escape(scope_name)}\.radix_rank "
        r"end_bit must not exceed the dtype bit width",
    ):
        rewrite._finalize_radix_rank_factory_kwargs(
            op_name=op_name,
            runtime_arg_count=2,
            seen_factory_kwargs={"begin_bit"},
            factory_kwargs={
                "dtype": types.int32,
                "begin_bit": 30,
                "end_bit": None,
            },
        )
