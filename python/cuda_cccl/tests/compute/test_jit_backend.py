# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""How cuda.compute drives the numba-cuda-mlir backend."""

import os
import subprocess
import sys
import textwrap

import pytest

from cuda.compute import _mlir

# numba-cuda-mlir's direct MLIR-to-LLVM translation is unconditionally unavailable
# on Windows (lowering_utilities.llvm_utils._get_capi raises there outright), and
# it is what every v2 (HostJIT) operator compile depends on -- not just these
# tests. numba-cuda-mlir's Windows-available alternatives (PTX/LTO-IR via its
# NVVM-downgrade bridge) target libnvvm's IR dialect, which is not a substitute
# for the plain LLVM IR HostJIT's own LLVM/Clang consumes. This blocks v2 on
# Windows entirely at the numba-cuda-mlir layer, independent of CI matrix
# configuration, until numba-cuda-mlir closes the gap.
requires_llvm_ir_extraction = pytest.mark.skipif(
    sys.platform == "win32",
    reason="numba-cuda-mlir cannot translate MLIR to LLVM IR on Windows",
)


def _pointer_signature():
    int32_ptr = _mlir.types.CPointer(_mlir.types.int32)
    return _mlir.types.void(int32_ptr, int32_ptr)


def _wrapper(a, r):
    r[0] = a[0] * 3 + 1


@requires_llvm_ir_extraction
@pytest.mark.parametrize("cc", [(7, 5), (8, 9), (10, 0)])
def test_llvm_ir_extraction_supports_target_arches(cc):
    """LLVM IR is extracted for any target arch.

    The gpu.module is translated directly and no device code is generated, so
    the extraction works for an arch the installed CUDA toolkit predates.
    """
    text_ir = _mlir.compile_to_llvm_ir(
        _wrapper, _pointer_signature(), f"extract_sm_{cc[0]}{cc[1]}", cc
    )
    assert "define" in text_ir


@requires_llvm_ir_extraction
def test_llvm_ir_extraction_generates_no_device_code(monkeypatch):
    """Extraction stops at LLVM IR rather than generating device code.

    The generated code would be discarded, and producing it ties the extraction
    to the target arches the installed libnvvm knows.
    """
    import numba_cuda_mlir.mlir_optimization as backend_optimization

    calls = []
    for name in ("_compile_to_ptx", "_call_llvm70_capi"):
        original = getattr(backend_optimization, name)

        def record(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(backend_optimization, name, record)

    _mlir.compile_to_llvm_ir(
        _wrapper, _pointer_signature(), "extract_no_codegen", (8, 9)
    )

    assert calls == []


@requires_llvm_ir_extraction
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
@requires_llvm_ir_extraction
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


@requires_llvm_ir_extraction
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


def _run_without_a_device(program):
    """Run ``program`` in a fresh interpreter with the devices hidden."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )


def test_infers_a_return_type_without_a_device():
    """Inference does not need a GPU.

    Constructing an operator on a build machine with no device must not fail
    before a target can even be named.
    """
    result = _run_without_a_device(
        """
        from cuda.compute import _mlir

        def add_one(a):
            return a + 1

        assert _mlir.infer_return_type(add_one, (_mlir.types.int32,)) is not None
        """
    )
    assert result.returncode == 0, result.stderr


@requires_llvm_ir_extraction
def test_compiles_for_a_named_target_without_a_device():
    """Naming a target compiles on a machine with no GPU.

    Building for a compute capability the build machine does not have is the
    point of naming one, so the extraction must not require a device either.
    """
    result = _run_without_a_device(
        """
        from cuda.compute import _mlir

        types = _mlir.types

        def scale(a, r):
            r[0] = a[0] * 3

        signature = types.void(
            types.CPointer(types.int32), types.CPointer(types.int32)
        )
        assert "define" in _mlir.compile_to_llvm_ir(
            scale, signature, "no_device", (8, 9)
        )
        """
    )
    assert result.returncode == 0, result.stderr


def test_operator_compiles_after_the_backend_has_already_compiled():
    """An operator still compiles when something else compiled first.

    The JIT backend builds its typing and target contexts on first use and then
    freezes them, so the wrapper's conversion has to be registered before
    anything else compiles or be re-read afterwards. Runs in a fresh interpreter
    because the ordering is the whole point.
    """
    program = textwrap.dedent(
        """
        import numpy as np
        from numba_cuda_mlir import cuda as backend_cuda

        @backend_cuda.jit
        def touch(a):
            i = backend_cuda.grid(1)
            if i < a.size:
                a[i] += 1

        backend_cuda.to_device(np.zeros(4, dtype=np.int32))
        touch[1, 4](backend_cuda.to_device(np.zeros(4, dtype=np.int32)))

        # Only now import cuda.compute and compile an operator through it.
        from cuda.compute import _mlir
        from cuda.compute._jit import _compile_op_impl
        from cuda.compute._caching import CachableFunction

        int32 = _mlir.types.int32
        _compile_op_impl(CachableFunction(lambda x: x + 1), (int32,), int32, (8, 9))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("result", ["scalar", "struct"])
def test_stateful_operator_infers_its_output_type(result):
    """A stateful operator compiles without being told its output type.

    Inferring it needs no code generation and no device, and the descriptor can
    describe a gpu_struct result, which a NumPy dtype cannot. No algorithm
    reaches this path today because they all pass an output type.
    """
    import numpy as np
    from _utils.device_array import DeviceArray

    from cuda.compute import gpu_struct
    from cuda.compute._jit import (
        _numba_type_to_type_descriptor,
        to_jit_op_adapter,
    )

    state = DeviceArray.from_numpy(np.array([5], dtype=np.int64))
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    if result == "scalar":

        def op(x):
            return x + state[0]

    else:

        def op(x):
            return Pair(x + state[0], x)

    int32 = _numba_type_to_type_descriptor(_mlir.types.int32)
    compiled = to_jit_op_adapter(op).compile((int32,))

    assert compiled.name.startswith("wrapped_op")


@pytest.mark.thread_unsafe(
    reason="Clears the process-wide caches, which a concurrent instance would race."
)
def test_clear_all_caches_drops_compiled_device_code():
    """clear_all_caches() must make the next build cold, not just empty the memos.

    A build after a clear is only genuinely cold if the JIT-compiled operator,
    the NVRTC-compiled iterator wrapper, and select's always-false predicate are
    recompiled too; otherwise the native build reruns while the (dominant) JIT
    cost is served from memo. Device code memoized on a live iterator object is
    object state the clear does not reach -- that caveat is pinned down here so
    it stays documented behavior rather than an accident.
    """
    import numpy as np
    from _utils.device_array import DeviceArray

    import cuda.compute
    from cuda.compute import CountingIterator, OpKind, TransformIterator
    from cuda.compute._cpp_compile import compile_cpp_op_code, compile_cpp_to_ltoir
    from cuda.compute._jit import _compile_op_impl
    from cuda.compute.algorithms._select import _always_false_op

    def add_one(x):
        return x + 1

    def make_iter():
        return TransformIterator(CountingIterator(np.int32(0)), add_one)

    def run_reduce(d_in):
        d_out = DeviceArray.empty(1, np.int32)
        cuda.compute.reduce_into(
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            h_init=np.array([0], dtype=np.int32),
            num_items=8,
        )
        assert d_out.copy_to_host()[0] == sum(range(1, 9))

    def run_select():
        h_in = np.arange(8, dtype=np.int32)
        d_out = DeviceArray.empty(8, np.int32)
        d_num = DeviceArray.empty(1, np.uint64)
        cuda.compute.select(
            d_in=DeviceArray.from_numpy(h_in),
            d_out=d_out,
            d_num_selected_out=d_num,
            cond=lambda x: x % 2 == 0,
            num_items=8,
        )
        assert int(d_num.copy_to_host()[0]) == 4

    memos = (_compile_op_impl, compile_cpp_op_code, _always_false_op)

    first = make_iter()
    run_reduce(first)
    run_select()
    assert all(m.cache_info().currsize > 0 for m in memos)

    cuda.compute.clear_all_caches()
    assert all(m.cache_info().currsize == 0 for m in memos)
    assert compile_cpp_to_ltoir.cache_info().currsize == 0

    # cache_clear() also zeroes the counters, so a rebuild with fresh objects
    # is cold exactly when every memo misses again.
    run_reduce(make_iter())
    run_select()
    assert all(m.cache_info().misses > 0 for m in memos)

    # Reusing the live iterator relinks the ops memoized on it: no recompile.
    cuda.compute.clear_all_caches()
    run_reduce(first)
    assert _compile_op_impl.cache_info().misses == 0
    assert compile_cpp_op_code.cache_info().misses == 0
