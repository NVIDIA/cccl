# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib

from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _caching
from cuda.coop.numba_mlir._semantic import _numba_semantic_token


class _CallableCacheSettings:
    offset = 1


_CALLABLE_CACHE_SETTINGS = _CallableCacheSettings()


class _CallableCachePolicy:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value + 1


class _CompilerStatePoison:
    def __getattribute__(self, name):
        raise AssertionError(f"compiler state was inspected through {name}")


def _callable_cache_op(value):
    return value + _CALLABLE_CACHE_SETTINGS.offset


def _aggregate_cache_op(left, right):
    return left + right


def _callable_class_cache_op(value):
    return _CallableCachePolicy(value).result()


def _make_named_stateful_callback(offset):
    def callback(left, right):
        return left + right + offset

    return callback


def test_disk_cache_disables_itself_when_its_directory_is_unavailable(monkeypatch):
    calls = []

    def unavailable(_cache_identity):
        raise PermissionError("cache directory is read-only")

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(_caching, "_cache_identity_path", unavailable)

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return value + 1

    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == [3, 3]
    assert not _caching._CACHE_USABLE


def test_disk_cache_bypasses_unsupported_key_values(monkeypatch, tmp_path):
    calls = []
    unsupported = object()

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(
        _caching,
        "_cache_identity_path",
        lambda _cache_identity: str(tmp_path),
    )

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return 4 if value is unsupported else value + 1

    assert compute(unsupported) == 4
    assert compute(unsupported) == 4
    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == [unsupported, unsupported, 3]
    assert _caching._CACHE_USABLE


def test_disk_cache_bypasses_unsupported_result_values(monkeypatch, tmp_path):
    calls = []
    unsupported = object()

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(
        _caching,
        "_cache_identity_path",
        lambda _cache_identity: str(tmp_path),
    )

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return unsupported if value == "unsupported" else value + 1

    assert compute("unsupported") is unsupported
    assert compute("unsupported") is unsupported
    assert not tuple(tmp_path.glob(".*"))
    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == ["unsupported", "unsupported", 3]
    assert _caching._CACHE_USABLE


def test_callable_symbol_tracks_qualified_global_attributes(monkeypatch):
    original = _types._callable_symbol_component(_callable_cache_op)

    monkeypatch.setattr(_CallableCacheSettings, "offset", 2)

    assert original != _types._callable_symbol_component(_callable_cache_op)


def test_callable_symbol_tracks_referenced_global_class_bodies(monkeypatch):
    original = _types._callable_symbol_component(_callable_class_cache_op)

    monkeypatch.setattr(
        _CallableCachePolicy,
        "result",
        lambda self: self.value - 1,
    )

    assert original != _types._callable_symbol_component(_callable_class_cache_op)


def test_callable_symbol_uses_numba_dispatcher_python_function():
    dispatcher = _types.cuda.jit(device=True)(_callable_cache_op)
    dispatcher.typingctx = _CompilerStatePoison()
    dispatcher.targetctx = _CompilerStatePoison()

    assert _types._callable_symbol_component(
        dispatcher
    ) == _types._callable_symbol_component(_callable_cache_op)
    assert _numba_semantic_token(dispatcher) == _numba_semantic_token(
        _callable_cache_op
    )


def test_numba_dispatcher_symbol_tracks_python_function_globals(monkeypatch):
    dispatcher = _types.cuda.jit(device=True)(_callable_cache_op)
    original = _types._callable_symbol_component(dispatcher)

    monkeypatch.setattr(_CallableCacheSettings, "offset", 2)

    assert original != _types._callable_symbol_component(dispatcher)


def test_stateful_symbol_keeps_callable_identity_with_explicit_name():
    first = _make_named_stateful_callback(1)
    second = _make_named_stateful_callback(2)

    first_symbol = _types._python_operator_symbol_name(
        first,
        _types.types.int32,
        (_types.types.int32, _types.types.int32),
        stateful_name="shared_name",
    )
    second_symbol = _types._python_operator_symbol_name(
        second,
        _types.types.int32,
        (_types.types.int32, _types.types.int32),
        stateful_name="shared_name",
    )

    assert first_symbol != second_symbol
    assert first_symbol == _types._python_operator_symbol_name(
        first,
        _types.types.int32,
        (_types.types.int32, _types.types.int32),
        stateful_name="shared_name",
    )
    assert "shared_name" in first_symbol


def test_symbol_hash_uses_numba_type_identity():
    assert _numba_semantic_token(_types.types.int32) == _numba_semantic_token(
        (
            "numba-cuda-mlir-type",
            type(_types.types.int32).__module__,
            type(_types.types.int32).__qualname__,
            "int32",
        )
    )
    int32 = hashlib.sha1()
    _types._hash_symbol_value(int32, _types.types.int32)
    another_int32 = hashlib.sha1()
    _types._hash_symbol_value(another_int32, _types.types.int32)
    float32 = hashlib.sha1()
    _types._hash_symbol_value(float32, _types.types.float32)

    assert int32.digest() == another_int32.digest()
    assert int32.digest() != float32.digest()


def test_numba_type_tokens_remain_distinct():
    values = (
        _types.types.int32,
        _types.types.float32,
        _types.types.float64,
        _types.types.uint32,
    )
    tokens = tuple(_numba_semantic_token(value) for value in values)

    assert len(set(tokens)) == len(values)


def test_device_ltoir_cache_tracks_callable_semantics(monkeypatch):
    calls = []

    def compile_device(fn, **kwargs):
        del fn, kwargs
        calls.append(_CALLABLE_CACHE_SETTINGS.offset)
        return f"ltoir-{_CALLABLE_CACHE_SETTINGS.offset}".encode(), None

    monkeypatch.setattr(_types.cuda, "compile", compile_device)
    _types._DEVICE_LTOIR_CACHE.clear()
    try:
        first = _types._compile_device_ltoir(
            _callable_cache_op,
            sig="int32(int32)",
            abi_info={"abi": "c"},
        )
        monkeypatch.setattr(_CallableCacheSettings, "offset", 2)
        second = _types._compile_device_ltoir(
            _callable_cache_op,
            sig="int32(int32)",
            abi_info={"abi": "c"},
        )
    finally:
        _types._DEVICE_LTOIR_CACHE.clear()

    assert first == b"ltoir-1"
    assert second == b"ltoir-2"
    assert calls == [1, 2]


def test_aggregate_python_operator_cache_uses_source_identity(monkeypatch):
    semantic_values = []
    compiled = []

    def semantic_token(value):
        semantic_values.append(value)
        return ("semantic-token", len(semantic_values))

    def compile_device(fn, **kwargs):
        compiled.append((fn, kwargs))
        return b"aggregate-ltoir", None

    monkeypatch.setattr(_types, "_numba_semantic_token", semantic_token)
    monkeypatch.setattr(_types.cuda, "compile", compile_device)
    operator = _types.DependentPythonOperator(
        _types.Constant(_types.types.complex128),
        [
            _types.Constant(_types.types.complex128),
            _types.Constant(_types.types.complex128),
        ],
        _types.Constant(_aggregate_cache_op),
    )
    _types._DEVICE_LTOIR_CACHE.clear()
    try:
        specialized = operator.specialize({})
    finally:
        _types._DEVICE_LTOIR_CACHE.clear()

    assert isinstance(specialized, _types.StatelessOperator)
    assert semantic_values == [
        _aggregate_cache_op,
        (
            "numba-cuda-mlir-python-operator-abi-v1",
            _aggregate_cache_op,
            False,
            True,
            (True, True),
        ),
    ]
    assert len(compiled) == 1
    compile_op, compile_kwargs = compiled[0]
    assert any(
        getattr(cell.cell_contents, "py_func", None) is _aggregate_cache_op
        for cell in compile_op.__closure__
    )
    assert compile_kwargs["sig"].return_type == _types.types.void
