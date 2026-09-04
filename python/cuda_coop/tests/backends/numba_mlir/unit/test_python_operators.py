# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _combine(lhs, rhs):
    return lhs + rhs


def _make_same_named_operator(offset):
    def combine(lhs, rhs):
        return lhs + rhs + offset

    combine.__module__ = "operator_collision_test"
    combine.__qualname__ = "combine"
    return combine


def test_core_adapter_lowers_stateless_operator_and_emits_scalar_wrapper(
    monkeypatch,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import PythonOperator, SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    compile_calls = []

    def compile_operator(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return b"callback-ltoir"

    monkeypatch.setattr(_types, "_compile_device_ltoir", compile_operator)
    operator = PythonOperator(
        ret_dtype=types.int32,
        arg_dtypes=(types.int32, types.int32),
        op=_combine,
        name="binary_op",
    )
    dependent = NumbaMlirCoreAdapter().lower_python_operator(
        operator,
        specialization=object(),
    )
    specialized = dependent.specialize({})

    assert isinstance(specialized, _types.StatelessOperator)
    assert specialized.ltoir == b"callback-ltoir"
    assert specialized.name.startswith("cuda_coop_numba_mlir_F")
    assert compile_calls[0][0] is _combine
    assert compile_calls[0][1]["sig"].return_type == types.int32
    assert compile_calls[0][1]["abi_info"] == {"abi_name": specialized.name}

    algorithm = _types.Algorithm(
        "BlockReduce<::cuda::std::int32_t, 32>",
        "Reduce",
        "callback_reduce",
        ["cub/block/block_reduce.cuh"],
        [],
        [
            [
                _types.Pointer(types.uint8),
                _types.Reference(types.int32),
                specialized,
                _types.Reference(types.int32, is_output=True),
            ]
        ],
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
        type_definitions=[_types.numba_type_to_wrapper(types.int32)],
        compile_context=object(),
    )
    source, support_ltoirs, _, declarations = algorithm._source_code(
        compile_identity=(90, True, "lto", (), "test-toolchain")
    )

    int32_cpp = "::cuda::std::int32_t"
    assert list(declarations) == [specialized.name]
    assert declarations[specialized.name] in source
    assert (
        f'extern "C" __device__ {int32_cpp} {specialized.name}'
        f"({int32_cpp}, {int32_cpp});"
    ) in source
    assert "auto param_1 = []" in source
    assert f"return {specialized.name}(wp_0, wp_1);" in source
    assert ".Reduce(param_0, param_1);" in source
    assert support_ltoirs == [b"callback-ltoir"]


def test_core_adapter_normalizes_device_dispatcher_to_python_function(monkeypatch):
    from numba_cuda_mlir import cuda, types

    from cuda.coop._core import PythonOperator
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    monkeypatch.setattr(
        _types,
        "_compile_device_ltoir",
        lambda fn, **kwargs: b"dispatcher-callback-ltoir",
    )
    dispatcher = cuda.jit(device=True)(_combine)
    dependent = NumbaMlirCoreAdapter().lower_python_operator(
        PythonOperator(
            ret_dtype=types.int32,
            arg_dtypes=(types.int32, types.int32),
            op=dispatcher,
        ),
        specialization=object(),
    )

    assert dependent.op.resolve({}) is _combine
    assert isinstance(dependent.specialize({}), _types.StatelessOperator)


def test_callable_symbols_and_lto_cache_use_callable_semantics(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    first = _make_same_named_operator(1)
    second = _make_same_named_operator(2)
    first_symbol = _types._python_operator_symbol_name(
        first,
        types.int32,
        (types.int32, types.int32),
    )
    second_symbol = _types._python_operator_symbol_name(
        second,
        types.int32,
        (types.int32, types.int32),
    )

    assert first_symbol == _types._python_operator_symbol_name(
        first,
        types.int32,
        (types.int32, types.int32),
    )
    assert first_symbol != second_symbol

    compile_calls = []

    def compile_operator(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return f"ltoir-{len(compile_calls)}".encode(), None

    monkeypatch.setattr(_types.cuda, "compile", compile_operator)
    _types._DEVICE_LTOIR_CACHE.clear()
    try:
        first_ltoir = _types._compile_device_ltoir(
            first,
            sig="int32(int32, int32)",
            abi_info={"abi_name": first_symbol},
        )
        repeated_ltoir = _types._compile_device_ltoir(
            first,
            sig="int32(int32, int32)",
            abi_info={"abi_name": first_symbol},
        )
        second_ltoir = _types._compile_device_ltoir(
            second,
            sig="int32(int32, int32)",
            abi_info={"abi_name": second_symbol},
        )
    finally:
        _types._DEVICE_LTOIR_CACHE.clear()

    assert first_ltoir == repeated_ltoir == b"ltoir-1"
    assert second_ltoir == b"ltoir-2"
    assert [call[0] for call in compile_calls] == [first, second]
    assert all(call[1]["output"] == "ltoir" for call in compile_calls)
    assert all(call[1]["forceinline"] for call in compile_calls)


def test_pointer_abi_adapts_aggregate_operator(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    captured = {}

    def compile_operator(fn, **kwargs):
        captured.update(fn=fn, **kwargs)
        return b"aggregate-callback-ltoir"

    monkeypatch.setattr(_types, "_compile_device_ltoir", compile_operator)
    specialized = _types.DependentPythonOperator(
        _types.Constant(types.complex128),
        (
            _types.Constant(types.complex128),
            _types.Constant(types.complex128),
        ),
        _types.Constant(_combine),
    ).specialize({})

    assert isinstance(specialized, _types.StatelessOperator)
    assert specialized.ret_cpp_type == "storage_t"
    assert specialized.arg_cpp_types == ("storage_t", "storage_t")
    assert captured["sig"].return_type == types.void
    assert captured["sig"].args == (
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
        types.CPointer(types.complex128),
    )
    assert captured["semantic_identity"] == (
        "numba-cuda-mlir-stateless-python-operator-abi-v1",
        _combine,
        True,
        (True, True),
    )
    assert specialized.forward_decl().endswith("(const void*, const void*, void*);")
    wrapper = specialized.wrap_decl("binary_op")
    assert "const storage_t& wp_0" in wrapper
    assert f"{specialized.name}(&wp_0, &wp_1, &result);" in wrapper


def test_provider_artifacts_include_callback_and_wrapper_ltoir(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    operator = _types.StatelessOperator(
        "cuda_coop_numba_mlir_Ftest",
        "::cuda::std::int32_t",
        ("::cuda::std::int32_t", "::cuda::std::int32_t"),
        b"callback-ltoir",
    )
    algorithm = _types.Algorithm(
        "FakeAlgorithm<::cuda::std::int32_t>",
        "Reduce",
        "artifact_reduce",
        [],
        [],
        [
            [
                _types.Value(types.int32),
                operator,
                _types.Value(types.int32, is_output=True),
            ]
        ],
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
        compile_context=object(),
    )
    compile_identity = (90, True, "lto", (), "test-toolchain")
    compile_calls = []

    def compile_provider(**kwargs):
        compile_calls.append(kwargs)
        return object(), b"provider-ltoir"

    monkeypatch.setattr(_types.nvrtc, "compile", compile_provider)
    monkeypatch.setattr(_types, "_ltoir_to_ptx", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        algorithm,
        "_current_provider_compile_identity",
        lambda: compile_identity,
    )

    invocable = _types.make_invocable_from_specialization(algorithm)
    artifacts = [Path(path) for path in invocable.files]

    assert [artifact.suffix for artifact in artifacts] == [".ltoir", ".ltoir"]
    assert [artifact.read_bytes() for artifact in artifacts] == [
        b"callback-ltoir",
        b"provider-ltoir",
    ]
    assert algorithm.lto_irs == [b"callback-ltoir", b"provider-ltoir"]
    assert compile_calls[0]["code"] == "lto"
    assert operator.forward_decl() in compile_calls[0]["cpp"]
    assert invocable.temp_storage_bytes == 0
    assert invocable.temp_storage_alignment == 1


def test_stateful_operator_remains_explicitly_unsupported():
    from numba_cuda_mlir import types

    from cuda.coop._core import StatefulOperator
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    operator = StatefulOperator(
        op=_combine,
        state_dtype=types.int32,
        ret_dtype=types.int32,
        arg_dtypes=(types.int32, types.int32),
        name="binary_op",
    )
    with pytest.raises(NotImplementedError, match="stateful callbacks"):
        NumbaMlirCoreAdapter().lower_stateful_operator(
            operator,
            specialization=object(),
        )
