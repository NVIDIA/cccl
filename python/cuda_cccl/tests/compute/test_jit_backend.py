# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""How cuda.compute drives the numba-cuda-mlir backend."""

import subprocess
import sys
import textwrap

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


def test_missing_backend_error_names_the_backend(monkeypatch):
    """Without a JIT backend the error names the package to install."""
    import builtins

    from cuda.compute import op as op_module

    real_import = builtins.__import__

    def block_backend(name, *args, **kwargs):
        if name.split(".")[0] == "numba_cuda_mlir" or name.endswith("_jit"):
            raise ModuleNotFoundError("No module named 'numba_cuda_mlir'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_backend)

    adapter = op_module._jit_op_adapter_factory()

    with pytest.raises(ImportError, match="numba-cuda-mlir"):
        adapter(lambda x: x)


@pytest.mark.parametrize(
    "op_name, expected_prefix",
    [("named_op", "wrapped_named_op_"), ("<lambda>", "wrapped__lambda__")],
)
def test_generated_wrapper_compiles_under_its_sanitized_name(op_name, expected_prefix):
    """The wrapper source is exec'd and compiled under the sanitized symbol.

    The operator's name reaches the generated source through exec and ends up as
    the emitted symbol, so a name needing sanitization has to survive the whole
    way.
    """
    from cuda.compute._odr_helpers import create_op_void_ptr_wrapper

    def add(a, b):
        return a + b

    add.__name__ = op_name

    signature = _mlir.types.int32(_mlir.types.int32, _mlir.types.int32)
    wrapper, wrapper_signature = create_op_void_ptr_wrapper(add, signature)

    # Two input pointers plus the result pointer.
    assert len(wrapper_signature.args) == 3
    assert wrapper.__name__.startswith(expected_prefix)

    text_ir = _mlir.compile_to_llvm_ir(
        wrapper, wrapper_signature, wrapper.__name__, (8, 9)
    )
    assert f"define void @{wrapper.__name__}" in text_ir


def test_operator_device_code_is_textual_llvm_ir():
    """The v2 backend hands an operator's LLVM IR over as text.

    The reader accepts either the textual or the bitcode encoding, so the IR
    needs no conversion to bitcode.
    """
    from cuda.compute._jit import _compile_op_to_llvm_ir
    from cuda.compute._odr_helpers import create_op_void_ptr_wrapper

    def add(a, b):
        return a + b

    signature = _mlir.types.int32(_mlir.types.int32, _mlir.types.int32)
    wrapper, wrapper_signature = create_op_void_ptr_wrapper(add, signature)

    code = _compile_op_to_llvm_ir(wrapper, wrapper_signature, (8, 9))

    # Bitcode would start with the "BC" magic; this is text.
    assert not code.startswith(b"BC")
    text = code.decode("utf-8")
    assert f"define void @{wrapper.__name__}" in text
    # Dropped so the module adopts the HostJIT module's layout when linked.
    assert "target datalayout" not in text


def test_return_type_inference_works_without_a_prior_compile():
    """Inferring a return type must not depend on something having compiled first.

    Runs in a fresh interpreter because any earlier compilation in this process
    would already have built the JIT backend's typing and target contexts, which
    is what resolving the operator below needs.
    """
    program = textwrap.dedent(
        """
        from cuda.compute import _mlir

        def add_one(a):
            return a + 1

        types = _mlir.types
        # Integer width follows numba's promotion rules; the point is that the
        # operator resolves at all.
        assert _mlir.infer_return_type(add_one, (types.int32,)) in (
            types.int32,
            types.int64,
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
