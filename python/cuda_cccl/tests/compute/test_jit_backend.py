# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""How cuda.compute drives the numba-cuda-mlir backend."""

import pytest

from cuda.compute import _mlir


def _pointer_signature():
    int32_ptr = _mlir.types.CPointer(_mlir.types.int32)
    return _mlir.types.void(int32_ptr, int32_ptr)


def _wrapper(a, r):
    r[0] = a[0] * 3 + 1


@pytest.mark.parametrize("cc", [(7, 5), (8, 9), (10, 0)])
def test_llvm_ir_extraction_supports_target_arches(cc):
    """LLVM IR is extracted for any target arch.

    The gpu.module is translated directly, so the extraction does not depend on
    which NVVM dialect the target arch would otherwise be lowered through.
    """
    text_ir = _mlir.compile_to_llvm_ir(
        _wrapper, _pointer_signature(), f"extract_sm_{cc[0]}{cc[1]}", cc
    )
    assert "define" in text_ir


def test_llvm_ir_extraction_does_not_request_lto_codegen(monkeypatch):
    """Extraction must not ask for an LTO-IR output.

    Requesting one runs a full LTO codegen whose result is discarded; the
    optimized MLIR the extraction consumes is produced either way.
    """
    import numba_cuda_mlir.compiler as backend_compiler

    recorded = {}
    original = backend_compiler.compile_mlir

    def record(pyfunc, sig, **kwargs):
        recorded.update(kwargs)
        return original(pyfunc, sig, **kwargs)

    monkeypatch.setattr(backend_compiler, "compile_mlir", record)

    _mlir.compile_to_llvm_ir(_wrapper, _pointer_signature(), "extract_no_lto", (8, 9))

    assert recorded.get("output") is None
    assert recorded.get("lto") is False


def test_stateful_wrapper_accepts_numpy_integer_shapes():
    """A state shape of numpy integers must not leak into the generated source.

    The shape is interpolated into the wrapper's source, where a numpy integer
    would render as ``np.int64(4)`` and reference a name the wrapper's namespace
    does not define.
    """
    import numpy as np

    from cuda.compute import _odr_helpers

    def add_state(state, x):
        return x + state[0]

    signature = _mlir.types.int32(
        _mlir.types.Array(_mlir.types.int32, 1, "C"), _mlir.types.int32
    )

    wrapper, wrapper_signature = _odr_helpers.create_stateful_op_void_ptr_wrapper(
        add_state, signature, [_mlir.types.int32], [(np.int64(4),)]
    )

    text_ir = _mlir.compile_to_llvm_ir(
        wrapper, wrapper_signature, "numpy_shape_wrapper", (8, 9)
    )
    assert "define" in text_ir
