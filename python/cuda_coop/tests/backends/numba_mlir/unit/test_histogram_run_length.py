# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _TypingContext:
    def refresh(self) -> None:
        pass


class _FakeInvocable:
    files = ("histogram-run-length-test.ltoir",)
    temp_storage_bytes = 0
    temp_storage_alignment = 1

    def __call__(self, *args) -> None:
        del args


def _plan(function, *, arg_types, block=(32, 1, 1)):
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._group_rewrites import _GroupCallPlanner

    func_ir = run_frontend(function)
    state = SimpleNamespace(func_ir=func_ir, args=arg_types)
    planner = _GroupCallPlanner(
        state,
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    )
    return func_ir, planner


def _single_phase_matches(func_ir, *, arg_types):
    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    state = SimpleNamespace(
        func_ir=func_ir,
        args=arg_types,
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    matches = []
    invocable = _FakeInvocable()
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None

    def materialize(match):
        matches.append(match)
        return invocable, False

    rewrite._materialize_invocable = materialize
    rewrite._record_invocable_specialization = lambda _invocable: None
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    return matches


def test_public_surface_exposes_only_group_first_operations() -> None:
    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._block._block_histogram import _group_histogram
    from cuda.coop.numba_mlir._block._block_run_length_decode import (
        _group_run_length_decode,
    )

    assert {"BlockHistogramAlgorithm", "histogram", "run_length_decode"} <= set(
        coop.__all__
    )
    assert inspect.signature(coop.histogram).parameters["group"].kind.name == (
        "POSITIONAL_ONLY"
    )
    assert inspect.signature(coop.run_length_decode).parameters["group"].kind.name == (
        "POSITIONAL_ONLY"
    )
    assert _group_histogram.__name__.startswith("_group_")
    assert _group_run_length_decode.__name__.startswith("_group_")
    for obsolete in ("BlockHistogram", "BlockRunLengthDecode", "run_length"):
        assert not hasattr(coop, obsolete)


def test_common_histogram_reaches_one_fused_private_provider() -> None:
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(value):
        samples = coop.ThreadData(2, dtype=types.int32)
        samples[0] = value
        samples[1] = value + 1
        return coop.histogram(
            coop.this_block(),
            samples,
            bins=64,
            bins_per_thread=2,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.int32,))
    assert planner.run()
    matches = _single_phase_matches(func_ir, arg_types=(types.int32,))
    fused = [match for match in matches if match.op_name == "_group_histogram"]
    assert len(fused) == 1
    assert fused[0].factory_kwargs == {
        "algorithm": "atomic",
        "bins": 64,
        "counter_dtype": types.uint32,
        "item_dtype": types.int32,
        "items_per_thread": 2,
        "threads_per_block": (32, 1, 1),
    }


def test_qualified_histogram_accepts_a_fixed_local_array() -> None:
    from numba_cuda_mlir import cuda, types

    import cuda.coop.numba_mlir as coop

    def kernel(value):
        samples = cuda.local.array(2, types.uint8)
        samples[0] = value
        samples[1] = value
        return coop.histogram(
            coop.this_block(),
            samples,
            bins=32,
            counter_dtype=types.uint64,
            algorithm=coop.BlockHistogramAlgorithm.SORT,
        )

    func_ir, planner = _plan(kernel, arg_types=(types.uint8,))
    assert planner.run()
    matches = _single_phase_matches(func_ir, arg_types=(types.uint8,))
    fused = next(match for match in matches if match.op_name == "_group_histogram")
    assert fused.factory_kwargs["item_dtype"] == types.uint8
    assert fused.factory_kwargs["counter_dtype"] == types.uint64


def test_common_histogram_rejects_local_arrays_and_insufficient_capacity() -> None:
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    def local_array(value):
        samples = cuda.local.array(1, types.int32)
        samples[0] = value
        return coop.histogram(coop.this_block(), samples, bins=32)

    _, planner = _plan(local_array, arg_types=(types.int32,))
    with pytest.raises(TypeError, match="requires samples to be coop.ThreadData"):
        planner.run()

    def insufficient(value):
        samples = coop.ThreadData(1, dtype=types.int32)
        samples[0] = value
        return coop.histogram(
            coop.this_block(),
            samples,
            bins=33,
            bins_per_thread=1,
        )

    _, planner = _plan(insufficient, arg_types=(types.int32,))
    with pytest.raises(ValueError, match="bins_per_thread is too small"):
        planner.run()


@pytest.mark.parametrize(("bins", "name"), [(0, "bins"), (-1, "bins_per_thread")])
def test_histogram_requires_positive_static_dimensions(bins, name) -> None:
    from numba_cuda_mlir import types

    from cuda import coop

    if name == "bins":

        def kernel(value):
            samples = coop.ThreadData(1, dtype=types.int32)
            samples[0] = value
            return coop.histogram(coop.this_block(), samples, bins=bins)

    else:

        def kernel(value):
            samples = coop.ThreadData(1, dtype=types.int32)
            samples[0] = value
            return coop.histogram(
                coop.this_block(),
                samples,
                bins=32,
                bins_per_thread=bins,
            )

    _, planner = _plan(kernel, arg_types=(types.int32,))
    with pytest.raises(
        ValueError,
        match=rf"histogram {name} must be a compile-time positive integer",
    ):
        planner.run()


def test_common_and_qualified_decode_share_the_fused_provider_shape() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    def common(value, length):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=2,
        )

    def qualified(value, length, offset):
        values = numba_coop.ThreadData(1, dtype=types.float32)
        lengths = numba_coop.ThreadData(1, dtype=types.uint32)
        total = numba_coop.ThreadData(1, dtype=types.uint32)
        relative = numba_coop.ThreadData(2, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return numba_coop.run_length_decode(
            numba_coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=2,
            decoded_window_offset=offset,
            relative_offsets=relative,
            total_decoded_size=total,
        )

    common_types = (types.int32, types.uint32)
    common_ir, planner = _plan(common, arg_types=common_types)
    assert planner.run()
    common_match = next(
        match
        for match in _single_phase_matches(common_ir, arg_types=common_types)
        if match.op_name == "_group_run_length_decode"
    )
    assert common_match.factory_kwargs["with_relative_offsets"] is False
    assert common_match.factory_kwargs["decoded_offset_dtype"] == types.uint32

    qualified_types = (types.float32, types.uint32, types.uint32)
    qualified_ir, planner = _plan(qualified, arg_types=qualified_types)
    assert planner.run()
    qualified_match = next(
        match
        for match in _single_phase_matches(qualified_ir, arg_types=qualified_types)
        if match.op_name == "_group_run_length_decode"
    )
    assert qualified_match.factory_kwargs == {
        "decoded_items_per_thread": 2,
        "decoded_offset_dtype": types.uint32,
        "item_dtype": types.float32,
        "relative_offset_dtype": types.uint32,
        "run_length_dtype": types.uint32,
        "runs_per_thread": 1,
        "threads_per_block": (32, 1, 1),
        "total_decoded_size_dtype": types.uint32,
        "with_relative_offsets": True,
    }


def test_decode_rejects_extent_and_static_offset_errors() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop

    def mismatched(value, length):
        values = coop.ThreadData(2, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=1,
        )

    _, planner = _plan(mismatched, arg_types=(types.int32, types.uint32))
    with pytest.raises(ValueError, match="must have the same items_per_thread"):
        planner.run()

    def negative(value, length):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=-1,
        )

    _, planner = _plan(negative, arg_types=(types.int32, types.uint32))
    with pytest.raises(ValueError, match="must be non-negative"):
        planner.run()


def test_decode_rejects_mismatched_output_and_dynamic_offset_dtypes() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def mismatched_total(value, length):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint64)
        total = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=1,
            total_decoded_size=total,
        )

    arg_types = (types.int32, types.uint64)
    func_ir, planner = _plan(mismatched_total, arg_types=arg_types)
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
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="total_decoded_size dtype must match run_lengths",
    ):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)

    def bad_offset(value, length, offset):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=offset,
        )

    arg_types = (types.int32, types.uint32, types.int64)
    func_ir, planner = _plan(bad_offset, arg_types=arg_types)
    assert planner.run()
    state.func_ir = func_ir
    state.args = arg_types
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    block = func_ir.blocks[min(func_ir.blocks)]
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="decoded_window_offset dtype must match run_lengths",
    ):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)


def test_decode_rejects_unrepresentable_static_offset() -> None:
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewrite,
        CoopSinglePhaseRewriteError,
    )

    def kernel(value, length):
        values = coop.ThreadData(1, dtype=types.int32)
        lengths = coop.ThreadData(1, dtype=types.uint32)
        values[0] = value
        lengths[0] = length
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=1,
            decoded_window_offset=1 << 32,
        )

    arg_types = (types.int32, types.uint32)
    func_ir, planner = _plan(kernel, arg_types=arg_types)
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
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="must be representable in the run_lengths dtype",
    ):
        rewrite.match(func_ir, block, state.typemap, state.calltypes)
