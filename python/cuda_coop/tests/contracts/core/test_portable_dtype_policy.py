# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module

import numpy as np
import pytest

from cuda.coop._core.api._dispatch import _compiler_scope
from cuda.coop._core.api._payload import (
    _validate_common_merge_sort_oob_default,
)
from cuda.coop._core.dtype_policy import (
    validate_portable_integer_key_dtype_name,
    validate_portable_integer_value_dtype_name,
    validate_portable_numeric_dtype_name,
)
from tests.support.group_planning import this_block


class _TypedPayload:
    def __init__(self, dtype=np.int32, *, items_per_thread=2, length=None):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * (items_per_thread if length is None else length)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value


@pytest.mark.parametrize(
    ("dtype_names", "validator", "parameter"),
    [
        (
            ("uint8", "int32", "uint32", "int64", "uint64", "float32", "float64"),
            validate_portable_numeric_dtype_name,
            None,
        ),
        (
            ("uint8", "int32", "uint32", "int64", "uint64"),
            validate_portable_integer_value_dtype_name,
            "sample",
        ),
        (
            ("int32", "uint32", "int64", "uint64"),
            validate_portable_integer_key_dtype_name,
            "key",
        ),
    ],
)
def test_portable_dtype_policy_accepts_supported_families(
    dtype_names,
    validator,
    parameter,
) -> None:
    kwargs = {"operation": "histogram"}
    if parameter is not None:
        kwargs["parameter"] = parameter
    for dtype_name in dtype_names:
        assert validator(dtype_name, **kwargs) == dtype_name


@pytest.mark.parametrize(
    "dtype_name",
    ["bool", "int8", "int16", "uint16", "float16", "complex64", "complex128"],
)
def test_portable_numeric_dtype_policy_rejects_backend_extensions(dtype_name) -> None:
    with pytest.raises(
        TypeError,
        match=(
            r"cuda\.coop\.exchange supports dtypes uint8, int32, uint32, int64, "
            r"uint64, float32, float64 through the portable API; use a "
            r"backend-qualified import for backend-specific dtypes"
        ),
    ):
        validate_portable_numeric_dtype_name(dtype_name, operation="exchange")


@pytest.mark.parametrize(
    ("validator", "dtype_name", "parameter", "supported"),
    [
        (
            validate_portable_integer_value_dtype_name,
            "float32",
            "sample",
            "uint8, int32, uint32, int64, uint64",
        ),
        (
            validate_portable_integer_key_dtype_name,
            "uint8",
            "counter",
            "int32, uint32, int64, uint64",
        ),
    ],
)
def test_portable_integer_dtype_diagnostics_are_backend_neutral(
    validator,
    dtype_name,
    parameter,
    supported,
) -> None:
    expected = (
        f"cuda.coop.histogram supports {parameter} dtypes {supported} through the "
        "portable API; "
        f"use a backend-qualified import for backend-specific {parameter} dtypes"
    )
    with pytest.raises(TypeError) as exc_info:
        validator(
            dtype_name,
            operation="histogram",
            parameter=parameter,
        )
    assert str(exc_info.value) == expected


@pytest.mark.parametrize(
    ("module_name", "operation", "kwargs"),
    [
        ("adjacent_difference", "adjacent_difference", {}),
        ("discontinuity", "discontinuity", {}),
        ("exchange", "exchange", {}),
        ("shuffle", "shuffle", {}),
        ("histogram", "histogram", {"bins": 4}),
    ],
)
def test_portable_data_collectives_validate_thread_data_payloads(
    monkeypatch,
    module_name,
    operation,
    kwargs,
) -> None:
    api = import_module(f"cuda.coop._core.api.{module_name}")
    function = getattr(api, operation)
    delegated = object()
    calls = []

    def marker(*args, **marker_kwargs):
        calls.append((args, marker_kwargs))
        return delegated

    monkeypatch.setattr(api, "_group_primitive_marker", marker)
    with _compiler_scope("test.backend"):
        assert function(this_block(), _TypedPayload(), **kwargs) is delegated
        with pytest.raises(TypeError, match="fixed-size ThreadData"):
            function(this_block(), object(), **kwargs)
        with pytest.raises(ValueError, match="must match the payload item count"):
            function(this_block(), _TypedPayload(length=1), **kwargs)
        with pytest.raises(TypeError, match="portable API"):
            function(this_block(), _TypedPayload(complex), **kwargs)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("dtype", "sentinel", "out_of_range"),
    [
        (np.int32, -(1 << 31), 1 << 31),
        (np.uint32, (1 << 32) - 1, -1),
        (np.int64, -(1 << 63), 1 << 63),
        (np.uint64, (1 << 64) - 1, 1 << 64),
    ],
)
def test_merge_sort_accepts_representable_python_integer_sentinels(
    dtype,
    sentinel,
    out_of_range,
) -> None:
    keys = _TypedPayload(dtype)

    _validate_common_merge_sort_oob_default("merge_sort_keys", keys, sentinel)
    _validate_common_merge_sort_oob_default("merge_sort_keys", keys, dtype(sentinel))
    with pytest.raises(ValueError, match="not representable"):
        _validate_common_merge_sort_oob_default("merge_sort_keys", keys, out_of_range)


@pytest.mark.parametrize(
    ("key_dtype", "typed_sentinel"),
    [
        (np.int32, np.uint32(1)),
        (np.uint32, np.int32(-1)),
        (np.int64, np.uint64(1)),
        (np.uint64, np.int64(1)),
    ],
)
def test_merge_sort_rejects_mismatched_typed_integer_sentinels(
    key_dtype,
    typed_sentinel,
) -> None:
    with pytest.raises(TypeError, match="same integer dtype as keys"):
        _validate_common_merge_sort_oob_default(
            "merge_sort_keys",
            _TypedPayload(key_dtype),
            typed_sentinel,
        )


@pytest.mark.parametrize("operation", ["merge_sort_keys", "merge_sort_pairs"])
def test_merge_sort_public_api_validates_python_and_typed_sentinels(
    operation,
    monkeypatch,
) -> None:
    merge_sort_api = import_module("cuda.coop._core.api.merge_sort")
    delegated = object()
    calls = []

    def marker(*args, **kwargs):
        calls.append((args, kwargs))
        return delegated

    monkeypatch.setattr(merge_sort_api, "_group_primitive_marker", marker)
    keys = _TypedPayload(np.uint32)
    args = [this_block(), keys]
    if operation == "merge_sort_pairs":
        args.append(_TypedPayload(np.float32))

    with _compiler_scope("test.backend"):
        result = getattr(merge_sort_api, operation)(
            *args,
            valid_items=1,
            oob_default=(1 << 32) - 1,
        )
    assert result is delegated
    assert len(calls) == 1

    calls.clear()
    with _compiler_scope("test.backend"):
        with pytest.raises(TypeError, match="same integer dtype as keys"):
            getattr(merge_sort_api, operation)(
                *args,
                valid_items=1,
                oob_default=np.int32(-1),
            )
    assert calls == []
