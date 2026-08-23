# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from types import SimpleNamespace

import pytest

import cuda.coop.numba_mlir as coop


class _FakeLocal:
    def __init__(self):
        self.calls = []

    def array(self, shape, dtype, *, alignas, **kwargs):
        call = (shape, dtype, alignas, kwargs)
        self.calls.append(call)
        return call


class _FakeRuntime:
    def __init__(self):
        self.local = _FakeLocal()


def _call(*args, **kwargs):
    return SimpleNamespace(func=None, args=args, kws=tuple(kwargs.items()))


def _qualified_rewrite(rewrites):
    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    rewrite._is_common_root_member = lambda _value, _name: False
    return rewrite


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = _FakeRuntime()
    monkeypatch.setattr(coop, "_runtime_import_error", None)
    monkeypatch.setattr(coop, "_cuda_module", runtime)
    return runtime


def test_thread_data_accepts_items_per_thread_keyword(fake_runtime):
    result = coop.ThreadData(
        items_per_thread=4,
        dtype="int32",
        alignas=16,
    )

    assert result == (4, "int32", 16, {})
    assert fake_runtime.local.calls == [result]


@pytest.mark.parametrize(
    "call", [lambda: coop.ThreadData(4), lambda: coop.ThreadData(items_per_thread=4)]
)
def test_thread_data_accepts_canonical_extent_forms(fake_runtime, call):
    assert call() == (4, None, 8, {})


@pytest.mark.parametrize(
    "call",
    [
        lambda: coop.ThreadData(shape=4),
        lambda: coop.ThreadData(4, address_space="local"),
    ],
)
def test_thread_data_rejects_legacy_or_forwarded_keywords(fake_runtime, call):
    del fake_runtime
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: coop.ThreadData(4, items_per_thread=4),
    ],
)
def test_thread_data_rejects_duplicate_items_per_thread(call):
    with pytest.raises(
        TypeError,
        match="multiple values for argument 'items_per_thread'",
    ):
        call()


def test_thread_data_requires_an_extent():
    with pytest.raises(
        TypeError, match="required positional argument: 'items_per_thread'"
    ):
        coop.ThreadData()


@pytest.mark.parametrize(
    ("items_per_thread", "error_type", "message"),
    [
        (True, TypeError, "items_per_thread must be an integer"),
        (1.5, TypeError, "items_per_thread must be an integer"),
        (0, ValueError, "items_per_thread must be a positive integer"),
        (-1, ValueError, "items_per_thread must be a positive integer"),
    ],
)
def test_thread_data_validates_items_per_thread(
    fake_runtime,
    items_per_thread,
    error_type,
    message,
):
    del fake_runtime
    with pytest.raises(error_type, match=message):
        coop.ThreadData(items_per_thread)


def test_single_phase_parser_accepts_the_canonical_extent_forms(monkeypatch):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    rewrite = _qualified_rewrite(rewrites)
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_infer_constant",
        lambda self, value: value,
    )

    canonical = rewrite._extract_thread_data_spec(_call(items_per_thread=4))
    positional = rewrite._extract_thread_data_spec(_call(2))

    assert canonical.items_per_thread == 4
    assert positional.items_per_thread == 2


@pytest.mark.parametrize(
    ("dtype_alias", "expected_dtype"),
    [
        (bool, "boolean"),
        (int, "int32"),
        (float, "float32"),
        (complex, "complex128"),
    ],
)
def test_single_phase_parser_maps_python_builtin_dtype_aliases(
    monkeypatch,
    dtype_alias,
    expected_dtype,
):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    from numba_cuda_mlir import types

    rewrite = _qualified_rewrite(rewrites)
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_infer_constant",
        lambda self, value: value,
    )

    spec = rewrite._extract_thread_data_spec(_call(2, dtype=dtype_alias))

    assert spec.dtype == getattr(types, expected_dtype)


@pytest.mark.parametrize("dtype_alias", [bool, complex])
def test_common_thread_data_builtin_extensions_reach_profile_diagnostic(
    dtype_alias,
):
    from cuda.coop.numba_mlir._common import _validate_common_numeric_dtype

    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.ThreadData common V1 supports dtypes uint8, int32, "
            r"uint32, int64, uint64, float32, float64; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        _validate_common_numeric_dtype(dtype_alias, operation="ThreadData")


@pytest.mark.parametrize(
    "call",
    [
        _call(4, items_per_thread=4),
    ],
)
def test_single_phase_parser_rejects_duplicate_items_per_thread(call):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    rewrite = _qualified_rewrite(rewrites)

    with pytest.raises(rewrites.CoopSinglePhaseRewriteError):
        rewrite._extract_thread_data_spec(call)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (_call(shape=4), "ThreadData got unexpected keyword.*shape"),
        (
            _call(address_space="local"),
            "ThreadData got unexpected keyword.*address_space",
        ),
    ],
)
def test_single_phase_parser_rejects_legacy_or_forwarded_keywords(call, message):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    rewrite = _qualified_rewrite(rewrites)

    with pytest.raises(
        rewrites.CoopSinglePhaseRewriteError,
        match=message,
    ):
        rewrite._extract_thread_data_spec(call)


@pytest.mark.parametrize(
    ("items_per_thread", "message"),
    [
        (True, "items_per_thread must be an integer"),
        (1.5, "items_per_thread must be an integer"),
        (0, "items_per_thread must be a positive integer"),
        (-1, "items_per_thread must be a positive integer"),
    ],
)
def test_single_phase_parser_validates_items_per_thread(
    monkeypatch,
    items_per_thread,
    message,
):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    rewrite = _qualified_rewrite(rewrites)
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_infer_constant",
        lambda self, value: value,
    )

    with pytest.raises(rewrites.CoopSinglePhaseRewriteError, match=message):
        rewrite._extract_thread_data_spec(_call(items_per_thread))


def test_temp_storage_uses_canonical_defaults_and_normalization():
    shared = coop.TempStorage(sharing=" SHARED ")
    exclusive = coop.TempStorage(sharing=" Exclusive ")

    assert shared.sharing == "shared"
    assert shared.auto_sync is True
    assert exclusive.sharing == "exclusive"
    assert exclusive.auto_sync is False


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"size_in_bytes": True},
            TypeError,
            "TempStorage size_in_bytes must be an integer or None",
        ),
        (
            {"size_in_bytes": 1.5},
            TypeError,
            "TempStorage size_in_bytes must be an integer or None",
        ),
        (
            {"size_in_bytes": 0},
            ValueError,
            "TempStorage size_in_bytes must be a positive integer",
        ),
        (
            {"alignment": False},
            TypeError,
            "TempStorage alignment must be an integer or None",
        ),
        (
            {"alignment": 3},
            ValueError,
            "TempStorage alignment must be a power of 2",
        ),
        (
            {"auto_sync": 1},
            TypeError,
            "TempStorage auto_sync must be None/True/False",
        ),
        (
            {"sharing": 1},
            TypeError,
            "TempStorage sharing must be a string",
        ),
        (
            {"sharing": "exclusive", "auto_sync": True},
            ValueError,
            "sharing='exclusive'.*auto_sync=True",
        ),
    ],
)
def test_temp_storage_validation_mirrors_cutlass(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        coop.TempStorage(**kwargs)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((True,), "TempStorage size_in_bytes must be an integer or None"),
        ((1.5,), "TempStorage size_in_bytes must be an integer or None"),
        ((0,), "TempStorage size_in_bytes must be a positive integer"),
        ((None, False), "TempStorage alignment must be an integer or None"),
        ((None, 3), "TempStorage alignment must be a power of 2"),
    ],
)
def test_single_phase_parser_validates_temp_storage(monkeypatch, args, message):
    rewrites = pytest.importorskip("cuda.coop.numba_mlir._single_phase_rewrites")
    rewrite = _qualified_rewrite(rewrites)
    monkeypatch.setattr(
        rewrites.CoopSinglePhaseRewrite,
        "_infer_constant",
        lambda self, value: value,
    )

    with pytest.raises(rewrites.CoopSinglePhaseRewriteError, match=message):
        rewrite._extract_temp_storage_ctor_spec(_call(*args))
