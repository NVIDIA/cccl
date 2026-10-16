# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The typed pipeline's Numba-CUDA-MLIR patches: when they apply, their pins,
and how kernels without cooperative types behave under them.

Activation tests start a fresh interpreter per case and run in any mode; the
kernels need typed mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._support import TYPED, requires_typed_pipeline, run_python

_TARGETS = """
from numba_cuda_mlir.typing import cuda as cuda_typing
from numba_cuda_mlir.numba_cuda.core import typeinfer
from numba_cuda_mlir import descriptor

owners = (
    typeinfer,
    descriptor._ArgMarshaller,
    cuda_typing.GenericArrayTemplate,
    cuda_typing.LocalArrayTemplate,
    typeinfer.Propagate,
    typeinfer.CallConstraint,
    typeinfer.StaticGetItemConstraint,
    typeinfer.BuildTupleConstraint,
)
before = [dict(vars(owner)) for owner in owners]
"""

_REPORT = (
    _TARGETS
    + """
import cuda.coop.numba_mlir

changed = sorted(
    f"{owner.__name__.rsplit('.', 1)[-1]}.{name}"
    for owner, old in zip(owners, before)
    for name in set(old) | set(vars(owner))
    if old.get(name) is not vars(owner).get(name)
)
print("changed", changed)
"""
)

PATCHED = [
    "BuildTupleConstraint.__call__",
    "BuildTupleConstraint.refine",
    "CallConstraint.refine",
    "LocalArrayTemplate.generic",
    "Propagate.__call__",
    "StaticGetItemConstraint.__call__",
    "StaticGetItemConstraint.refine",
    "_ArgMarshaller._launch",
    "typeinfer.TypeInferer",
]


@pytest.mark.parametrize("pipeline", [None, "planner"])
def test_the_default_pipeline_leaves_the_compiler_untouched(pipeline):
    result = run_python(_REPORT, pipeline)
    assert result.returncode == 0, result.stderr
    assert "changed []" in result.stdout


def test_the_typed_pipeline_replaces_only_its_patch_targets():
    result = run_python(_REPORT, "typed")
    assert result.returncode == 0, result.stderr
    assert f"changed {PATCHED}" in result.stdout


def test_patch_target_change_fails():
    code = (
        _TARGETS
        + """
cuda_typing.GenericArrayTemplate.generic = lambda self: None
try:
    import cuda.coop.numba_mlir._pipeline as pipeline
    pipeline._activate()
except ImportError as exc:
    print(type(exc).__name__, exc)
print("local-array", "generic" in vars(cuda_typing.LocalArrayTemplate))
"""
    )
    result = run_python(code, "typed")
    assert result.returncode == 0, result.stderr
    assert (
        "TypedPipelineUnavailableError The cuda.coop typed pipeline's "
        "'argument-sized cuda.local.array' patch was written against "
        "Numba-CUDA-MLIR 0.5.4, but GenericArrayTemplate.generic has "
        "changed. The patch is retired by U-P-1"
    ) in result.stdout
    assert "local-array False" in result.stdout


def test_another_compiler_version_fails_closed():
    code = """
import numba_cuda_mlir
numba_cuda_mlir.__version__ = "0.5.5"
try:
    import cuda.coop.numba_mlir._pipeline as pipeline
    pipeline._activate()
except ImportError as exc:
    print(type(exc).__name__, exc)
"""
    result = run_python(code, "typed")
    assert result.returncode == 0, result.stderr
    assert (
        "TypedPipelineUnavailableError The cuda.coop typed pipeline's "
        "'argument-sized cuda.local.array' patch was written against "
        "Numba-CUDA-MLIR 0.5.4, but the installed compiler is 0.5.5."
    ) in result.stdout


@requires_typed_pipeline
def test_apply_is_idempotent():
    from cuda.coop.numba_mlir._pipeline import _ncm_patches

    replaced = _ncm_patches.replaced()
    _ncm_patches.apply()
    assert _ncm_patches.replaced() == replaced
    assert (
        sorted(f"{o.__name__.rsplit('.', 1)[-1]}.{n}" for o, n in replaced)
        == PATCHED
    )


@pytest.mark.skipif(TYPED, reason="checks the pins without applying them")
def test_the_pins_match_the_installed_compiler():
    from cuda.coop.numba_mlir._pipeline import _ncm_patches

    for patch in _ncm_patches.PATCHES:
        _ncm_patches.check(patch)


# Argument-sized cuda.local.array (U-P-1) ------------------------------------


@requires_typed_pipeline
@pytest.mark.runtime
@pytest.mark.gpu
def test_a_plain_kernel_with_an_argument_sized_local_array_compiles():
    from numba_cuda_mlir import cuda, types

    @cuda.jit
    def triangle(output, n):
        values = cuda.local.array(n, types.int32)
        for i in range(n):
            values[i] = i + 1
        total = 0
        for i in range(n):
            total += values[i]
        output[cuda.threadIdx.x] = total

    output = np.zeros(32, np.int32)
    for n in (3, 5, 3):
        triangle[1, 32](output, n)
        np.testing.assert_array_equal(output, n * (n + 1) // 2)
    assert len(triangle.overloads) == 2


@requires_typed_pipeline
@pytest.mark.runtime
@pytest.mark.gpu
def test_a_computed_local_array_shape_is_still_refused():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    @cuda.jit
    def computed(output, n):
        values = cuda.local.array(n + 1, types.int32)
        values[0] = 1
        output[0] = values[0]

    with pytest.raises(TypingError, match=r"Cannot request literal type"):
        computed[1, 1](np.zeros(1, np.int32), 2)


@requires_typed_pipeline
@pytest.mark.runtime
@pytest.mark.gpu
def test_a_tuple_local_array_shape_from_an_argument_is_still_refused():
    from numba_cuda_mlir import cuda, types
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    @cuda.jit
    def tuple_shape(output, n):
        values = cuda.local.array((n, 2), types.int32)
        values[0, 0] = 1
        output[0] = values[0, 0]

    with pytest.raises(TypingError, match=r"No implementation of function"):
        tuple_shape[1, 1](np.zeros(1, np.int32), 2)


# Payload through a tuple alias (U-H1-4) -------------------------------------


def test_tuple_refinement_is_scoped_to_thread_data():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._pipeline._ncm_patches import holds_thread_data
    from cuda.coop.numba_mlir._pipeline._types import ThreadDataType

    payload = ThreadDataType(types.float32, 4)
    plain = types.float32[::1]
    assert holds_thread_data(types.Tuple((plain, types.UniTuple(payload, 1))))
    assert not holds_thread_data(types.Tuple((plain, types.int64)))
    assert not holds_thread_data(plain)


@requires_typed_pipeline
@pytest.mark.runtime
@pytest.mark.gpu
def test_a_plain_kernel_projecting_tuples_is_unchanged():
    from numba_cuda_mlir import cuda, types

    @cuda.jit
    def projected(source, output, scale):
        values = cuda.local.array(4, types.float32)
        pair = (values, scale)
        alias = pair[0]
        for i in range(4):
            alias[i] = source[i] * pair[1]
        nested = ((values,), 1)
        output[cuda.threadIdx.x] = nested[0][0][3] + nested[-1]

    source = np.arange(4, dtype=np.float32) + 1
    output = np.zeros(32, np.float32)
    projected[1, 32](source, output, np.float32(2))
    np.testing.assert_array_equal(output, 9)


# Phi of two ThreadData constructors (U-H1-3) --------------------------------


@requires_typed_pipeline
@pytest.mark.runtime
@pytest.mark.gpu
def test_a_plain_kernel_joining_arrays_and_scalars_is_unchanged():
    from numba_cuda_mlir import cuda, types

    @cuda.jit
    def joined(output, first):
        a = cuda.local.array(4, types.float32)
        b = cuda.local.array(4, types.float32)
        if first[0]:
            values = a
            scale = 2
        else:
            values = b
            scale = 0.5
        for i in range(4):
            values[i] = i * scale
        output[cuda.threadIdx.x] = values[3]

    output = np.zeros(32, np.float32)
    for first, expected in ((1, 6), (0, 1.5)):
        joined[1, 32](output, np.array([first], np.int32))
        np.testing.assert_array_equal(output, expected)
