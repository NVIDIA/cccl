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
