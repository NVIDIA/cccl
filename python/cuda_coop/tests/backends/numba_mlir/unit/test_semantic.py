# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_HELPER = None


def _global_callback(left, right):
    return _HELPER(left, right)


def _helper(offset=0, **options):
    from numba_cuda_mlir import cuda

    @cuda.jit(device=True, **options)
    def add(left, right):
        return left + right + offset

    return add


def _callback(helper):
    def apply(left, right):
        return helper(left, right)

    return apply


def _array_callback(table):
    def apply(left, right):
        return left + right + table[1000]

    return apply


def test_captured_arrays_use_all_contents_independently_of_print_options():
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    first = np.zeros(2000, dtype=np.int32)
    second = first.copy()
    second[1000] = 1
    assert repr(first) == repr(second)
    first_token = _numba_semantic_token(_array_callback(first))
    second_token = _numba_semantic_token(_array_callback(second))
    assert first_token != second_token
    assert first_token == _numba_semantic_token(_array_callback(first.copy()))
    with np.printoptions(threshold=1, precision=0):
        assert first_token == _numba_semantic_token(_array_callback(first))
        assert second_token == _numba_semantic_token(_array_callback(second))


@pytest.mark.parametrize(
    "other",
    [
        np.zeros(8, dtype=np.float32),
        np.zeros((2, 4), dtype=np.int32),
        np.zeros(16, dtype=np.int32)[::2],
        np.zeros(8, dtype=np.int32)[::-1],
    ],
    ids=["dtype", "shape", "strided", "reversed"],
)
def test_array_identity_preserves_dtype_shape_and_strides(other):
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    first = np.zeros(8, dtype=np.int32)
    assert first.tobytes() == other.tobytes()
    assert _numba_semantic_token(first) != _numba_semantic_token(other)


def test_array_identity_preserves_record_fields():
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    first = np.zeros(1, dtype=[("value", np.int32, (2,))])
    second = np.zeros(1, dtype=[("value", np.float32, (2,))])
    assert first.dtype.str == second.dtype.str
    assert first.tobytes() == second.tobytes()
    assert _numba_semantic_token(first) != _numba_semantic_token(second)


@pytest.mark.parametrize("dtype", [object, [("value", object)]])
def test_array_identity_rejects_object_dtypes(dtype):
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    with pytest.raises(TypeError, match="NumPy arrays with object dtypes"):
        _numba_semantic_token(_array_callback(np.zeros(2000, dtype=dtype)))


@pytest.mark.parametrize("location", ("global", "closure", "two_helpers"))
def test_nested_dispatchers_do_not_traverse_compiler_implementation(
    monkeypatch, location
):
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.descriptor import MLIRDispatcher

    from cuda.coop._core import _symbols
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    original = _symbols._type_dependency_token

    def reject_dispatcher_class(value, state):
        assert value is not MLIRDispatcher, "fingerprinting compiler implementation"
        return original(value, state)

    monkeypatch.setattr(_symbols, "_type_dependency_token", reject_dispatcher_class)
    helper = _helper()
    if location == "global":
        monkeypatch.setitem(globals(), "_HELPER", helper)
        callback = _global_callback
    else:
        if location == "two_helpers":
            helper = cuda.jit(device=True)(_callback(helper))
        callback = _callback(helper)

    first = _numba_semantic_token(callback)
    helper._diagnostic_cycle = helper
    assert first == _numba_semantic_token(callback)


def test_nested_helper_closures_and_defaults_change_identity():
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    first = _numba_semantic_token(_callback(_helper(1)))
    assert first == _numba_semantic_token(_callback(_helper(1)))
    assert first != _numba_semantic_token(_callback(_helper(2)))

    helper = _helper(1)
    callback = _callback(helper)
    before = _numba_semantic_token(callback)
    helper.py_func.__defaults__ = (3,)
    assert before != _numba_semantic_token(callback)


@pytest.mark.parametrize("options", ({"fastmath": True}, {"inline": False}))
def test_nested_helper_compile_options_change_identity(options):
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    assert _numba_semantic_token(_callback(_helper())) != _numba_semantic_token(
        _callback(_helper(**options))
    )


def test_frozen_signatures_matter_but_lazy_overloads_do_not(monkeypatch):
    from numba_cuda_mlir import types
    from numba_cuda_mlir.descriptor import MLIRDispatcher

    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    helper = _helper()
    callback = _callback(helper)
    observed = []
    signature = SimpleNamespace(
        return_type=types.int32, args=(types.int32, types.int32)
    )

    def signatures(self):
        observed.append(self)
        return [signature]

    monkeypatch.setattr(MLIRDispatcher, "nopython_signatures", property(signatures))
    lazy = _numba_semantic_token(callback)
    assert not observed
    helper._can_compile = False
    frozen = _numba_semantic_token(callback)
    assert observed
    assert frozen != lazy
    signature.return_type = types.int64
    assert frozen != _numba_semantic_token(callback)


def test_nested_helper_semantics_control_symbols_and_lto_reuse(monkeypatch):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types

    compile_calls = []

    def compile_callback(fn, **kwargs):
        compile_calls.append(fn)
        return f"callback-{len(compile_calls)}".encode(), None

    monkeypatch.setattr(_types.cuda, "compile", compile_callback)
    monkeypatch.setattr(_types, "_DEVICE_LTOIR_CACHE", {})
    helper = _helper(1)
    monkeypatch.setitem(globals(), "_HELPER", helper)
    sig = (types.int32, types.int32)

    def symbol():
        return _types._python_operator_symbol_name(_global_callback, types.int32, sig)

    def compile_lto():
        return _types._compile_device_ltoir(
            _global_callback, sig=sig, abi_info={}, compute_capability=(9, 0)
        )

    first_symbol, first_lto = symbol(), compile_lto()
    helper._diagnostic_cycle = helper
    assert symbol() == first_symbol
    assert compile_lto() == first_lto
    assert len(compile_calls) == 1
    monkeypatch.setitem(globals(), "_HELPER", _helper(2))
    assert symbol() != first_symbol
    assert compile_lto() != first_lto
    assert len(compile_calls) == 2


def test_recursive_helper_graph_has_repeatable_identity():
    from numba_cuda_mlir import cuda

    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    @cuda.jit(device=True)
    def left(value):
        return right(value)

    @cuda.jit(device=True)
    def right(value):
        return left(value)

    assert _numba_semantic_token(left) == _numba_semantic_token(left)


def test_top_level_compile_options_remain_owned_by_coop():
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    # The outer callback is compiled from py_func with cuda.coop's options.
    assert _numba_semantic_token(_helper()) == _numba_semantic_token(
        _helper(fastmath=True)
    )


def test_captured_numba_types_use_type_key_instead_of_display_name():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    class SameNameType(types.Type):
        def __init__(self, key):
            self.identity = key
            super().__init__(name="same-name")

        @property
        def key(self):
            return self.identity

    def callback(dtype):
        def apply(value):
            return dtype(value)

        return apply

    first, second = SameNameType(1), SameNameType(2)
    assert str(first) == str(second)
    assert first != second
    assert _numba_semantic_token(callback(first)) != _numba_semantic_token(
        callback(second)
    )


def test_core_operator_preserves_backend_policy_for_nested_helper(monkeypatch):
    from numba_cuda_mlir.descriptor import MLIRDispatcher

    from cuda.coop._core import INT32, PythonOperator, _symbols, semantic_token
    from cuda.coop.numba_mlir._semantic import _numba_semantic_token

    original = _symbols._type_dependency_token

    def reject_dispatcher_class(value, state):
        assert value is not MLIRDispatcher, "fingerprinting compiler implementation"
        return original(value, state)

    monkeypatch.setattr(_symbols, "_type_dependency_token", reject_dispatcher_class)
    monkeypatch.setitem(globals(), "_HELPER", _helper(1))
    operator = PythonOperator(
        ret_dtype=INT32,
        arg_dtypes=(INT32, INT32),
        op=_global_callback,
        op_tokenizer=_numba_semantic_token,
    )
    first = semantic_token(operator)
    assert first == semantic_token(operator)
    monkeypatch.setitem(globals(), "_HELPER", _helper(2))
    assert first != semantic_token(operator)
