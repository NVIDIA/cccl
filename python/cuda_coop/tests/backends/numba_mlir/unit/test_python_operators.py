# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from types import SimpleNamespace

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


class _RunningPrefix:
    def __call__(self_ptr, aggregate):
        previous = self_ptr[0]
        self_ptr[0] = previous + aggregate
        return previous


class _OffsetPrefix:
    def __call__(self_ptr, aggregate):
        return self_ptr[0] + aggregate


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
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
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
    assert specialized.compute_capability == (9, 0)
    assert specialized.name.startswith("cuda_coop_numba_mlir_F")
    assert compile_calls[0][0] is _combine
    assert compile_calls[0][1]["sig"].return_type == types.int32
    assert compile_calls[0][1]["abi_info"] == {"abi_name": specialized.name}
    assert compile_calls[0][1]["compute_capability"] == (9, 0)

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
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
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
            compute_capability=(9, 0),
        )
        repeated_ltoir = _types._compile_device_ltoir(
            first,
            sig="int32(int32, int32)",
            abi_info={"abi_name": first_symbol},
            compute_capability=(9, 0),
        )
        second_ltoir = _types._compile_device_ltoir(
            second,
            sig="int32(int32, int32)",
            abi_info={"abi_name": second_symbol},
            compute_capability=(9, 0),
        )
    finally:
        _types._DEVICE_LTOIR_CACHE.clear()

    assert first_ltoir == repeated_ltoir == b"ltoir-1"
    assert second_ltoir == b"ltoir-2"
    assert [call[0] for call in compile_calls] == [first, second]
    assert all(call[1]["output"] == "ltoir" for call in compile_calls)
    assert all(call[1]["forceinline"] for call in compile_calls)
    assert all(call[1]["cc"] == (9, 0) for call in compile_calls)


def test_callback_lto_cache_separates_compute_capabilities(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    target = [(8, 0)]
    compile_targets = []

    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=target[0]),
    )

    def compile_operator(fn, **kwargs):
        del fn
        compile_targets.append(kwargs["cc"])
        return f"callback-{kwargs['cc'][0]}{kwargs['cc'][1]}".encode(), None

    monkeypatch.setattr(_types.cuda, "compile", compile_operator)
    dependent = _types.DependentPythonOperator(
        _types.Constant(types.int32),
        (_types.Constant(types.int32), _types.Constant(types.int32)),
        _types.Constant(_combine),
    )
    _types._DEVICE_LTOIR_CACHE.clear()
    try:
        first = dependent.specialize({})
        repeated = dependent.specialize({})
        target[0] = (9, 0)
        second = dependent.specialize({})
    finally:
        _types._DEVICE_LTOIR_CACHE.clear()

    assert first.ltoir == repeated.ltoir == b"callback-80"
    assert first.compute_capability == repeated.compute_capability == (8, 0)
    assert second.ltoir == b"callback-90"
    assert second.compute_capability == (9, 0)
    assert compile_targets == [(8, 0), (9, 0)]


def test_pointer_abi_adapts_aggregate_operator(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    captured = {}

    def compile_operator(fn, **kwargs):
        captured.update(fn=fn, **kwargs)
        return b"aggregate-callback-ltoir"

    monkeypatch.setattr(_types, "_compile_device_ltoir", compile_operator)
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
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
        compute_capability=(9, 0),
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


def test_provider_rejects_callback_lto_for_a_different_target():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    operator = _types.StatelessOperator(
        "cuda_coop_numba_mlir_Fwrong_target",
        "::cuda::std::int32_t",
        ("::cuda::std::int32_t", "::cuda::std::int32_t"),
        b"callback-ltoir",
        compute_capability=(8, 0),
    )
    algorithm = _types.Algorithm(
        "FakeAlgorithm<::cuda::std::int32_t>",
        "Reduce",
        "target_guard_reduce",
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

    with pytest.raises(
        RuntimeError,
        match=r"callback 8\.0, provider 9\.0",
    ):
        algorithm._source_code(compile_identity=(90, True, "lto", ()))


def test_core_adapter_lowers_stateful_operator_and_emits_state_capture(
    monkeypatch,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import StatefulOperator, SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    compile_calls = []

    def compile_operator(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return b"stateful-callback-ltoir"

    monkeypatch.setattr(_types, "_compile_device_ltoir", compile_operator)
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    operator = StatefulOperator(
        op=_RunningPrefix,
        state_dtype=types.int32,
        ret_dtype=types.int32,
        arg_dtypes=(types.int32,),
        name="running_prefix",
    )
    dependent = NumbaMlirCoreAdapter().lower_stateful_operator(
        operator,
        specialization=object(),
    )
    specialized = dependent.specialize({})

    assert isinstance(specialized, _types.StatefulOperator)
    assert specialized.state_dtype == types.int32
    assert specialized.ltoir == b"stateful-callback-ltoir"
    assert specialized.compute_capability == (9, 0)
    assert compile_calls[0][0] is _RunningPrefix.__call__
    assert compile_calls[0][1]["sig"].return_type == types.int32
    assert compile_calls[0][1]["sig"].args == (
        types.CPointer(types.int32),
        types.int32,
    )

    algorithm = _types.Algorithm(
        "BlockScan<::cuda::std::int32_t, 32>",
        "ExclusiveSum",
        "stateful_scan",
        ["cub/block/block_scan.cuh"],
        [],
        [
            [
                _types.Pointer(types.uint8),
                _types.Reference(types.int32),
                _types.Reference(types.int32, is_output=True),
                specialized,
            ]
        ],
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
        fake_return=True,
        compile_context=object(),
    )
    source, support_ltoirs, _, declarations = algorithm._source_code(
        compile_identity=(90, True, "lto", (), "test-toolchain")
    )

    int32_cpp = "::cuda::std::int32_t"
    assert declarations[specialized.name] in source
    assert (
        f'extern "C" __device__ {int32_cpp} {specialized.name}'
        f"(char *state, {int32_cpp});"
    ) in source
    assert "auto param_2 = [param_2_state]" in source
    assert f"return {specialized.name}(param_2_state, wp_0);" in source
    assert ".ExclusiveSum(param_0, param_1, param_2);" in source
    assert support_ltoirs == [b"stateful-callback-ltoir"]


def test_stateful_symbols_use_callable_semantics_not_diagnostic_name(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import StatefulFunction, _types
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    monkeypatch.setattr(
        _types,
        "_compile_device_ltoir",
        lambda fn, **kwargs: repr((fn, kwargs["abi_info"])).encode(),
    )
    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )

    def specialize(op, *, name="shared_name", state_dtype=types.int32):
        return _types.DependentStatefulOperator(
            _types.Constant(state_dtype),
            _types.Constant(types.int32),
            (_types.Constant(types.int32),),
            _types.Constant(op),
            name=name,
        ).specialize({})

    first = specialize(_RunningPrefix)
    repeated = specialize(_RunningPrefix)
    renamed = specialize(_RunningPrefix, name="another_display_name")
    wider_state = specialize(_RunningPrefix, state_dtype=types.int64)
    second = specialize(_OffsetPrefix)
    stateless_name = _types._python_operator_symbol_name(
        _RunningPrefix.__call__,
        types.int32,
        (types.int32,),
    )

    assert first.name == repeated.name == renamed.name
    assert first.name != wider_state.name
    assert first.name != stateless_name
    assert first.name != second.name
    assert _numba_semantic_token(
        StatefulFunction(_RunningPrefix, types.int32, name="first")
    ) == _numba_semantic_token(
        StatefulFunction(_RunningPrefix, types.int32, name="second")
    )
