# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS_PER_THREAD = 3
_BINS = 97
_BINS_PER_THREAD = 2


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


def _planned_factory_calls(func_ir, ir, factory):
    globals_by_name = {
        inst.target.name: inst.value.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Global)
    }
    return [
        inst.value
        for block_ir in func_ir.blocks.values()
        for inst in block_ir.body
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
        and globals_by_name.get(inst.value.func.name) is factory
    ]


def _plan(function, *, arg_types):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": _BLOCK, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_numba_dtype_normalization_accepts_builtin_and_numpy_aliases(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._common import normalize_dtype_param

    assert normalize_dtype_param(bool) == types.boolean
    assert normalize_dtype_param(int) == types.int32
    assert normalize_dtype_param(float) == types.float32
    assert normalize_dtype_param(complex) == types.complex128
    assert normalize_dtype_param(np.int32) == types.int32
    assert normalize_dtype_param(np.int64) == types.int64


def test_group_histogram_uses_unsigned_provider_counters(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._group_rewrites import (
        _histogram_provider_counter_dtype,
    )

    assert _histogram_provider_counter_dtype(types.int32) == types.uint32
    assert _histogram_provider_counter_dtype(types.uint32) == types.uint32
    assert _histogram_provider_counter_dtype(types.int64) == types.uint64
    assert _histogram_provider_counter_dtype(types.uint64) == types.uint64
    assert _histogram_provider_counter_dtype(types.uint8) == types.uint8


@pytest.mark.parametrize(
    ("sample_dtype", "counter_dtype", "diagnostic"),
    [
        (
            "float32",
            "int32",
            r"cuda\.coop\.histogram common V1 supports sample dtypes uint8, "
            r"int32, uint32, int64, uint64",
        ),
        (
            "int32",
            "uint8",
            r"cuda\.coop\.histogram common V1 supports counter dtypes int32, "
            r"uint32, int64, uint64",
        ),
    ],
)
def test_common_histogram_validates_sample_and_counter_dtype_families(
    optional_backend,
    sample_dtype,
    counter_dtype,
    diagnostic,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._common import _validate_common_histogram_dtypes

    with pytest.raises(TypeError, match=diagnostic):
        _validate_common_histogram_dtypes(
            getattr(types, sample_dtype),
            getattr(types, counter_dtype),
        )


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_common_and_qualified_histogram_lower_to_the_same_array_factories(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numbair_transforms import ir

    import cuda.coop.numba_mlir as numba_coop
    import cuda.coop.numba_mlir._block as scoped_block
    from cuda import coop
    from cuda.coop.numba_mlir._group_rewrites import has_group_markers

    def common(value):
        samples = coop.ThreadData(_ITEMS_PER_THREAD, dtype=np.uint8)
        for index in range(_ITEMS_PER_THREAD):
            samples[index] = value + index
        group = coop.this_block()
        atomic = coop.histogram(
            group,
            samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=int,
        )
        sorted_counts = coop.histogram(
            group,
            samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=np.int64,
            algorithm="sort",
        )
        return samples[0], atomic[0], sorted_counts[1]

    def qualified(value):
        samples = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.uint8,
        )
        for index in range(_ITEMS_PER_THREAD):
            samples[index] = value + index
        group = numba_coop.this_block()
        atomic = numba_coop.histogram(
            group,
            samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
        )
        sorted_counts = numba_coop.histogram(
            group,
            samples,
            bins=_BINS,
            bins_per_thread=_BINS_PER_THREAD,
            counter_dtype=types.int64,
            algorithm="sort",
        )
        return samples[0], atomic[0], sorted_counts[1]

    for function, is_common in ((common, True), (qualified, False)):
        func_ir, planner = _plan(function, arg_types=(types.uint8,))
        assert has_group_markers(func_ir)
        assert planner.run()
        assert not has_group_markers(func_ir)
        assert _planned_factories(func_ir, ir)[scoped_block.histogram] == 2
        histogram_calls = _planned_factory_calls(
            func_ir,
            ir,
            scoped_block.histogram,
        )
        for call in histogram_calls:
            keyword_names = {name for name, _ in call.kws}
            assert ("_common_profile_operation" in keyword_names) is is_common


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_common_histogram_rejects_scalar_samples(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def common(value):
        return coop.histogram(coop.this_block(), value, bins=64)

    _, planner = _plan(common, arg_types=(types.uint8,))
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.histogram requires samples to be coop\.ThreadData "
            r"in common V1"
        ),
    ):
        planner.run()


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_common_histogram_requires_thread_data_but_qualified_keeps_local_arrays(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value):
        samples = cuda.local.array(2, types.uint8)
        samples[0] = value
        return coop.histogram(coop.this_block(), samples, bins=64)

    _, common_planner = _plan(common, arg_types=(types.uint8,))
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.histogram requires samples to be coop\.ThreadData "
            r"in common V1"
        ),
    ):
        common_planner.run()

    def qualified(value):
        samples = cuda.local.array(2, types.uint8)
        samples[0] = value
        return numba_coop.histogram(
            numba_coop.this_block(),
            samples,
            bins=64,
        )

    _, qualified_planner = _plan(qualified, arg_types=(types.uint8,))
    assert qualified_planner.run()


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_common_histogram_rejects_insufficient_striped_capacity(optional_backend):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def common(value):
        samples = coop.ThreadData(1, dtype=types.uint8)
        samples[0] = value
        return coop.histogram(coop.this_block(), samples, bins=65)

    _, planner = _plan(common, arg_types=(types.uint8,))
    with pytest.raises(
        ValueError,
        match=(
            "histogram bins_per_thread is too small for 65 bins and block "
            "size 64; need at least 2"
        ),
    ):
        planner.run()


@pytest.mark.evidence_for("group.histogram", backend="numba_mlir", evidence="lowering")
def test_common_histogram_uses_consistent_static_positive_diagnostics(
    optional_backend,
):
    optional_backend("numba_mlir")
    pytest.importorskip("numba_cuda_mlir")

    from numba_cuda_mlir import types

    from cuda import coop

    def invalid_type(value):
        samples = coop.ThreadData(1, dtype=types.uint8)
        samples[0] = value
        return coop.histogram(coop.this_block(), samples, bins="64")

    _, planner = _plan(invalid_type, arg_types=(types.uint8,))
    with pytest.raises(
        TypeError,
        match="histogram bins must be a compile-time positive integer",
    ):
        planner.run()

    def invalid_value(value):
        samples = coop.ThreadData(1, dtype=types.uint8)
        samples[0] = value
        return coop.histogram(
            coop.this_block(),
            samples,
            bins=64,
            bins_per_thread=0,
        )

    _, planner = _plan(invalid_value, arg_types=(types.uint8,))
    with pytest.raises(
        ValueError,
        match="histogram bins_per_thread must be a compile-time positive integer",
    ):
        planner.run()
