# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

typing = pytest.importorskip("cutlass.base_dsl.typing")
ir = import_module("cutlass._mlir.ir")
_types = import_module("cuda.coop.cutlass._compiler._types")
ThreadData = import_module("cuda.coop.cutlass._thread_data").ThreadData

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]

_DTYPE_CASES = [
    (np.int8, typing.Int8),
    (np.uint8, typing.Uint8),
    (np.int16, typing.Int16),
    (np.uint16, typing.Uint16),
    (np.int32, typing.Int32),
    (np.uint32, typing.Uint32),
    (np.int64, typing.Int64),
    (np.uint64, typing.Uint64),
    (np.float32, typing.Float32),
    (np.float64, typing.Float64),
]
_RESOLVE_TYPE = _types.make_provider_type_resolver(
    scope="cuda.coop.cutlass",
    root_scope="cuda.coop.cutlass",
    namespace="thread_group",
)


@pytest.mark.parametrize(("numpy_type", "dsl_type"), _DTYPE_CASES)
def test_all_common_numeric_types(numpy_type, dsl_type):
    for value in (
        numpy_type,
        np.dtype(numpy_type),
        numpy_type(3),
        dsl_type,
        dsl_type(3),
    ):
        assert _types.canonical_dsl_type(value) is dsl_type
        assert (
            _types._validate_common_root_numeric_dtype(value, operation="store")
            is dsl_type
        )
    assert _types.type_size_bytes(dsl_type) == np.dtype(numpy_type).itemsize
    assert _types.TYPE_SPECS[dsl_type].width_bits == 8 * np.dtype(numpy_type).itemsize


@pytest.mark.parametrize(("numpy_type", "dsl_type"), _DTYPE_CASES[:8])
def test_integer_literal_range_checks(numpy_type, dsl_type):
    limits = np.iinfo(numpy_type)
    for value in (int(limits.min), int(limits.max)):
        assert (
            _types.coerce_plain_scalar(
                value,
                dsl_type,
                name="item",
                scope="test",
                allow_nonfinite=False,
                convert=False,
            )
            == value
        )
    for value in (int(limits.min) - 1, int(limits.max) + 1):
        with pytest.raises(ValueError, match="representable"):
            _types.coerce_plain_scalar(
                value, dsl_type, name="item", scope="test", allow_nonfinite=False
            )


@pytest.mark.parametrize(("numpy_type", "dsl_type"), _DTYPE_CASES[:8])
def test_integer_ir_requires_signedness_or_explicit_payload_dtype(numpy_type, dsl_type):
    width = 8 * np.dtype(numpy_type).itemsize
    signed = np.issubdtype(numpy_type, np.signedinteger)
    item = SimpleNamespace(type=f"i{width}", signed=bool(signed))
    assert _types.canonical_dsl_type(item) is dsl_type
    signless = SimpleNamespace(type=f"i{width}")
    with pytest.raises(TypeError, match="signedness"):
        _types.canonical_dsl_type(signless)
    data = ThreadData.from_values(signless, dtype=dsl_type)
    resolved, values = _types.resolve_thread_data_value_type(
        data,
        allowed=_types.ALL_PROVIDER_TYPES,
        feature="store",
        scope="cuda.coop.cutlass",
        resolve_type=_RESOLVE_TYPE,
    )
    assert resolved is dsl_type
    assert values == (signless,)


def test_payload_dtype():
    inferred = ThreadData.from_values(np.int16(2), np.int16(4))
    resolved, values = _types.resolve_thread_data_value_type(
        inferred,
        allowed=_types.ALL_PROVIDER_TYPES,
        feature="store",
        scope="cuda.coop.cutlass",
        resolve_type=_RESOLVE_TYPE,
    )
    assert resolved is typing.Int16
    assert values == (2, 4)
    for invalid in (
        ThreadData.from_values(np.int16(2), np.int32(4)),
        ThreadData.from_values(np.int32(2), dtype=np.int16),
    ):
        with pytest.raises(TypeError, match="homogeneous|does not match"):
            _types.resolve_thread_data_value_type(
                invalid,
                allowed=_types.ALL_PROVIDER_TYPES,
                feature="store",
                scope="cuda.coop.cutlass",
                resolve_type=_RESOLVE_TYPE,
            )


@pytest.mark.parametrize(
    "value", [bool, True, np.bool_(True), np.float16, np.complex64]
)
def test_unsupported_common_types(value):
    with pytest.raises(TypeError, match="dtypes"):
        _types._validate_common_root_numeric_dtype(value, operation="store")


def test_float_literal_must_match_and_fit_dtype():
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            _types.coerce_plain_scalar(
                value, typing.Float32, name="item", scope="test", allow_nonfinite=False
            )
    with pytest.raises(ValueError, match="representable"):
        _types.coerce_plain_scalar(
            1e100, typing.Float32, name="item", scope="test", allow_nonfinite=True
        )
    with pytest.raises(TypeError, match="does not match"):
        _types.coerce_plain_scalar(
            1.0, typing.Int32, name="item", scope="test", allow_nonfinite=True
        )


@pytest.mark.parametrize("value_type", (typing.Float32, typing.Float64))
def test_integer_literal_float_target(value_type):
    assert (
        _types.coerce_plain_scalar(
            2,
            value_type,
            name="initial",
            scope="test",
            allow_nonfinite=False,
            convert=False,
        )
        == 2
    )
    with pytest.raises(ValueError, match="representable"):
        _types.coerce_plain_scalar(
            10**400,
            value_type,
            name="initial",
            scope="test",
            allow_nonfinite=False,
        )


@pytest.mark.parametrize("value", [-1, 1 << 31, np.uint64(1 << 32)])
def test_static_valid_items_cannot_wrap(value):
    with pytest.raises(ValueError, match="between"):
        _types.as_valid_items_arg(value, scope="test")


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.5])
def test_valid_items_requires_integer(value):
    with pytest.raises(TypeError, match="integer"):
        _types.as_valid_items_arg(value, scope="test")


@pytest.mark.parametrize(
    ("dtype", "value", "valid"),
    [
        (typing.Int64, -1, False),
        (typing.Int64, 1 << 32, False),
        (typing.Int64, 4, True),
        (typing.Uint32, 1 << 31, False),
        (typing.Uint32, 4, True),
        (typing.Uint64, 1 << 32, False),
        (typing.Uint64, 4, True),
    ],
)
def test_wide_valid_items_retains_original_range(dtype, value, valid):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            count = _types.as_valid_items_arg(dtype(value), scope="test")
        select = count.ir_value().owner
        condition = select.operands[0].owner.attributes["value"]
        assert bool(ir.IntegerAttr(condition).value) is valid
        # An invalid wide count reaches the provider as its rejected sentinel,
        # even when its low 32 bits would be a valid tile count.
        fallback = select.operands[2].owner.attributes["value"]
        assert ir.IntegerAttr(fallback).value == -1
        assert module.operation.verify()
