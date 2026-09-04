# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_SOURCE = 'extern "C" __device__ int raw_add_one(int value) { return value + 1; }'


def _compile_context():
    from cuda.coop.numba_mlir._compiler import _nvrtc
    from cuda.coop.numba_mlir._compiler._artifacts import version

    return _nvrtc.CompileContext(
        toolkit_root="/toolkit",
        toolkit_version=(13, 0),
        nvrtc_path="/toolkit/lib/libnvrtc.so",
        nvrtc_builtins_path="/toolkit/lib/libnvrtc-builtins.so",
        nvjitlink_path="/toolkit/lib/libnvJitLink.so",
        nvrtc_version=version(13, 0),
        nvjitlink_version=(13, 0),
        include_dirs=("/headers",),
        header_identity="headers",
    )


def _make_raw(monkeypatch, **overrides):
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    compile_calls = []

    def compile_source(**kwargs):
        compile_calls.append(kwargs)
        return object(), b"raw-ltoir"

    monkeypatch.setattr(_types.nvrtc, "compile", compile_source)
    kwargs = {
        "source": _SOURCE,
        "symbol": "raw_add_one",
        "return_type": types.int32,
        "parameters": (_types.ExactValue(types.int32),),
        "abi_transforms": ("value",),
        "cc": 90,
        "compile_context": _compile_context(),
        "storage_abi": StorageABI.NONE,
        "execution_scope": SynchronizationScope.GROUP,
        "synchronization_scope": SynchronizationScope.NONE,
    }
    kwargs.update(overrides)
    return _types.RawCAbiInvocable(**kwargs), compile_calls


def _registry_sizes():
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.extending import typeof_impl, typing_registry

    return (
        len(typeof_impl.registry),
        len(typing_registry.functions),
        len(typing_registry.globals),
        len(typing_registry.attributes),
        len(mlir_target.typing_context._functions),
        len(mlir_target.typing_context._globals),
        len(mlir_target.typing_context._attributes),
    )


def test_raw_c_abi_invocable_compiles_exact_source_and_exposes_contract(
    monkeypatch,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    invocable, compile_calls = _make_raw(monkeypatch)
    artifact = Path(invocable.files[0])
    try:
        assert compile_calls == [
            {
                "cpp": _SOURCE,
                "cc": 90,
                "rdc": True,
                "code": "lto",
                "context": invocable.compile_context,
            }
        ]
        assert artifact.read_bytes() == b"raw-ltoir"
        assert invocable.source == _SOURCE
        assert invocable.symbol == "raw_add_one"
        assert invocable.return_type is types.int32
        assert invocable.abi_types == (types.int32,)
        assert invocable.storage_abi is StorageABI.NONE
        assert invocable.execution_scope is SynchronizationScope.GROUP
        assert invocable.synchronization_scope is SynchronizationScope.NONE
        assert invocable.temp_storage_bytes == 0
        assert invocable.temp_storage_alignment == 1
        assert invocable.specialization is None
    finally:
        invocable._temp_file_finalizer()
    assert not artifact.exists()


def test_raw_c_abi_invocable_typing_is_local_and_owns_artifact(monkeypatch):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.descriptor import mlir_target

    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._numba_mlir_compat import (
        _get_numba_mlir_compat,
    )

    registry_sizes = _registry_sizes()
    invocable, _ = _make_raw(
        monkeypatch,
        parameters=(_types.ExactValue(types.int32), types.int32),
        abi_transforms=("value", "value"),
    )
    artifact = Path(invocable.files[0])
    invocable_ref = weakref.ref(invocable)

    invocable_type = mlir_target.typing_context.resolve_value_type(invocable)
    assert len(invocable_type.templates) == 1
    assert issubclass(
        invocable_type.templates[0],
        _get_numba_mlir_compat().overload_function_template,
    )
    assert invocable.abi_types == (types.int32, types.int32)
    overload = invocable_type.templates[0]._overload_func
    implementation = overload(types.int32, types.int32)
    assert callable(implementation)
    assert overload(types.int64, types.int32) is None
    assert _registry_sizes() == registry_sizes

    del invocable
    gc.collect()
    assert invocable_ref() is not None
    assert artifact.is_file()

    del implementation, invocable_type, overload
    gc.collect()
    assert invocable_ref() is None
    assert not artifact.exists()
    assert _registry_sizes() == registry_sizes


def test_raw_c_abi_invocable_metadata_is_checked_by_generic_rewrite(monkeypatch):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
    )
    from cuda.coop.numba_mlir._compiler._rewrite_invocables import (
        _InvocableRewrite,
    )
    from cuda.coop.numba_mlir._compiler._rewrite_support import (
        CoopSinglePhaseRewriteError,
    )

    invocable, _ = _make_raw(monkeypatch)
    artifact = Path(invocable.files[0])
    metadata = FactoryOperation(
        operation="raw",
        namespace="cudax",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.GROUP,
        synchronization_scope=SynchronizationScope.NONE,
    )
    try:
        _InvocableRewrite._validate_invocable(invocable, metadata)
        with pytest.raises(
            CoopSinglePhaseRewriteError,
            match="execution_scope=.*registered 'block'",
        ):
            _InvocableRewrite._validate_invocable(
                invocable,
                replace(metadata, execution_scope=SynchronizationScope.BLOCK),
            )
    finally:
        invocable._temp_file_finalizer()
    assert not artifact.exists()


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    [
        ({"abi_transforms": ()}, ValueError, "inconsistent arity"),
        ({"parameters": (object(),)}, TypeError, "Parameter objects"),
        ({"abi_transforms": ("other",)}, ValueError, "'ptr' or 'value'"),
        ({"storage_abi": "leading_pointer"}, ValueError, "storage_abi='none'"),
        ({"synchronization_scope": "group"}, ValueError, "scope='none'"),
    ],
)
def test_raw_c_abi_invocable_rejects_invalid_abi_metadata(
    monkeypatch,
    overrides,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        _make_raw(monkeypatch, **overrides)
