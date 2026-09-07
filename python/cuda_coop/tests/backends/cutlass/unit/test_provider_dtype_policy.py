# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy

import numpy as np
import pytest


def test_cutlass_provider_canonicalizes_every_common_numeric_dtype(
    optional_backend,
) -> None:
    optional_backend("cutlass")

    from cuda.coop._core.dtype_policy import COMMON_V1_NUMERIC_DTYPE_NAMES
    from cuda.coop.cutlass._dsl import _provider

    ordinary = {
        int: "int32",
        float: "float32",
        np.uint8: "uint8",
        np.int32: "int32",
        np.uint32: "uint32",
        np.int64: "int64",
        np.uint64: "uint64",
        np.float32: "float32",
        np.float64: "float64",
    }
    assert tuple(_provider.PROVIDER_TYPE_NAMES.values()) == tuple(
        COMMON_V1_NUMERIC_DTYPE_NAMES
    )
    for ordinary_type, dtype_name in ordinary.items():
        provider_type = _provider.PROVIDER_TYPE_NAMES[
            _provider.canonical_dsl_type(ordinary_type)
        ]
        assert provider_type == dtype_name
        if ordinary_type not in {int, float}:
            assert (
                _provider.PROVIDER_TYPE_NAMES[
                    _provider.canonical_dsl_type(np.dtype(ordinary_type))
                ]
                == dtype_name
            )
            assert (
                _provider.PROVIDER_TYPE_NAMES[
                    _provider.canonical_dsl_type(ordinary_type(1))
                ]
                == dtype_name
            )


@pytest.mark.parametrize("value", [bool, complex, np.bool_, np.int16, np.complex128])
def test_cutlass_provider_does_not_promote_nonportable_ordinary_types(
    optional_backend,
    value,
) -> None:
    optional_backend("cutlass")

    from cuda.coop.cutlass._dsl import _provider

    assert _provider.canonical_dsl_type(value) not in _provider.ALL_PROVIDER_TYPES


def _resolve_declared_thread_data(optional_backend, dtype, *values):
    cutlass_coop = optional_backend("cutlass")

    from cuda.coop.cutlass._dsl import _provider

    resolve_type = _provider.make_provider_type_resolver(
        scope="cuda.coop.cutlass",
        root_scope="cuda.coop.cutlass",
        namespace="thread_group",
    )
    data = cutlass_coop.ThreadData.from_values(*values, dtype=dtype)
    return _provider.resolve_thread_data_value_type(
        data,
        allowed=_provider.ALL_PROVIDER_TYPES,
        feature="test_primitive",
        scope="cuda.coop.cutlass",
        resolve_type=resolve_type,
    )


@pytest.mark.parametrize(
    ("dtype_name", "values"),
    (
        ("Uint8", (7, 250)),
        ("Int32", (7, -3)),
        ("Uint32", (7, 4_000_000_000)),
        ("Int64", (7, -(1 << 40))),
        ("Uint64", (7, 1 << 63)),
        ("Float32", (0.5, -3.25)),
        ("Float64", (0.5, -3.25)),
    ),
)
def test_declared_thread_data_converts_plain_literals_primitive_neutrally(
    optional_backend,
    dtype_name,
    values,
) -> None:
    from cutlass.base_dsl import typing as cutlass_types

    dtype = getattr(cutlass_types, dtype_name)
    value_type, converted = _resolve_declared_thread_data(
        optional_backend,
        dtype,
        *values,
    )

    assert value_type is dtype
    assert all(isinstance(value, dtype) for value in converted)


@pytest.mark.parametrize(
    ("dtype_name", "value", "error", "message"),
    (
        ("Uint8", -1, ValueError, "not representable"),
        ("Uint8", 256, ValueError, "not representable"),
        ("Int32", 1.5, TypeError, "dtype does not match"),
        ("Float32", 1e39, ValueError, "not representable"),
    ),
)
def test_declared_thread_data_rejects_incompatible_plain_literals(
    optional_backend,
    dtype_name,
    value,
    error,
    message,
) -> None:
    from cutlass.base_dsl import typing as cutlass_types

    dtype = getattr(cutlass_types, dtype_name)
    with pytest.raises(error, match=message):
        _resolve_declared_thread_data(optional_backend, dtype, value)


def test_declared_thread_data_preserves_nonfinite_floating_payloads(
    optional_backend,
) -> None:
    from cutlass.base_dsl.typing import Float32

    _, converted = _resolve_declared_thread_data(
        optional_backend,
        Float32,
        float("inf"),
        float("nan"),
    )

    assert all(isinstance(value, Float32) for value in converted)


def test_declared_thread_data_requires_exact_explicit_item_dtypes(
    optional_backend,
) -> None:
    from cutlass.base_dsl.typing import Int32, Int64

    _, converted = _resolve_declared_thread_data(
        optional_backend,
        Int32,
        np.int32(1),
    )
    assert isinstance(converted[0], np.int32)

    with pytest.raises(TypeError, match="dtype does not match initialized item types"):
        _resolve_declared_thread_data(optional_backend, Int64, np.int32(1))


def test_provider_common_provenance_changes_only_the_root_diagnostic(
    optional_backend,
) -> None:
    optional_backend("cutlass")

    from cuda.coop._core import root_api
    from cuda.coop.cutlass._dsl import _provider

    kwargs = {
        "allowed": _provider.ALL_PROVIDER_TYPES,
        "feature": "sum",
        "root_scope": "cuda.coop.cutlass",
        "namespace": "block",
        "canonical_type": _provider.canonical_dsl_type,
    }
    with pytest.raises(NotImplementedError, match=r"provider sum currently supports"):
        _provider.resolve_provider_type(complex, **kwargs)
    with root_api._common_root_operation_scope("sum"):
        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.sum common V1 supports dtypes",
        ):
            _provider.resolve_provider_type(complex, **kwargs)

    assert root_api._common_root_operation_name() is None


def test_common_thread_data_enforces_explicit_and_inferred_numeric_profile(
    optional_backend,
) -> None:
    optional_backend("cutlass")

    from cuda import coop
    from cuda.coop._core import root_api

    portable_dtypes = (
        int,
        float,
        np.uint8,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        assert coop.ThreadData(1).dtype is None
        for dtype in portable_dtypes:
            assert coop.ThreadData(1, dtype).dtype is dtype

        for dtype in (bool, complex, np.bool_, np.int16, np.complex128):
            with pytest.raises(
                TypeError,
                match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
            ):
                coop.ThreadData(1, dtype)

        for value in (True, np.bool_(True), 1 + 2j, np.complex128(1 + 2j)):
            inferred = coop.ThreadData(1)
            with pytest.raises(
                TypeError,
                match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
            ):
                inferred[0] = value
            inferred[0] = 1
            assert inferred[0] == 1


def test_common_thread_data_provenance_survives_preserving_adapters(
    optional_backend,
) -> None:
    cutlass_coop = optional_backend("cutlass")

    from cuda import coop
    from cuda.coop._core import root_api

    class ExistingPayload:
        def __init__(self, payload):
            self.payload = payload

        def __cuda_coop_thread_data_load__(self):
            return self.payload

    with root_api._compiler_scope("cuda.coop.cutlass"):
        common = coop.ThreadData(1)
    common[0] = 7

    copied = copy.copy(common)
    deepcopied = copy.deepcopy(common)
    adapted_same = cutlass_coop.ThreadData.from_payload(common)
    adapted_typed = cutlass_coop.ThreadData.from_payload(common, dtype=np.int32)
    loaded_same = cutlass_coop.ThreadData.load(ExistingPayload(common))
    loaded_typed = cutlass_coop.ThreadData.load(
        ExistingPayload(common),
        dtype=np.int32,
    )
    uninitialized = common._new_uninitialized(dtype=np.int32)

    assert adapted_same is common
    assert loaded_same is common
    for preserved in (
        copied,
        deepcopied,
        adapted_same,
        adapted_typed,
        loaded_same,
        loaded_typed,
        uninitialized,
    ):
        for value in (True, 1 + 2j):
            with pytest.raises(
                TypeError,
                match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
            ):
                preserved[0] = value

    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
    ):
        common._new_uninitialized(dtype=complex)
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
    ):
        cutlass_coop.ThreadData.from_payload(common, dtype=complex)
    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
    ):
        cutlass_coop.ThreadData.load(ExistingPayload(common), dtype=complex)


def test_qualified_thread_data_keeps_generic_values_and_dtypes(
    optional_backend,
) -> None:
    cutlass_coop = optional_backend("cutlass")

    class CustomValue:
        pass

    first_custom = CustomValue()
    second_custom = CustomValue()
    cases = (
        (complex, 1 + 2j, 3 + 4j),
        (bool, True, False),
        (CustomValue, first_custom, second_custom),
    )
    for dtype, initial, replacement in cases:
        values = cutlass_coop.ThreadData(
            1,
            dtype=dtype,
            values=[initial],
        )
        assert values[0] is initial
        values[0] = replacement
        assert values[0] is replacement

    untyped = cutlass_coop.ThreadData.from_values(first_custom)
    for preserved in (
        copy.copy(untyped),
        copy.deepcopy(untyped),
        cutlass_coop.ThreadData.from_payload(untyped, dtype=CustomValue),
        untyped._new_uninitialized(dtype=CustomValue),
    ):
        preserved[0] = second_custom
        assert preserved[0] is second_custom


def test_qualified_thread_data_casts_ordinary_dtype_classes(
    optional_backend,
) -> None:
    cutlass_coop = optional_backend("cutlass")

    class Vector:
        shape = (2,)

        def __getitem__(self, index):
            return index + 0.75

    class Producer:
        def __cuda_coop_thread_data_load__(self):
            return Vector()

    generated = cutlass_coop.ThreadData.from_fn(
        2,
        lambda index: index + 0.75,
        dtype=int,
    )
    vector = cutlass_coop.ThreadData.from_vector(Vector(), dtype=np.int32)
    payload = cutlass_coop.ThreadData.from_payload(Vector(), dtype=float)
    loaded = cutlass_coop.ThreadData.load(Producer(), dtype=np.float32)

    assert generated.values("generated") == (0, 1)
    assert all(type(value) is int for value in generated.values("generated"))
    assert vector.values("vector") == (np.int32(0), np.int32(1))
    assert all(type(value) is np.int32 for value in vector.values("vector"))
    assert payload.values("payload") == (0.75, 1.75)
    assert all(type(value) is float for value in payload.values("payload"))
    assert loaded.values("loaded") == (np.float32(0.75), np.float32(1.75))
    assert all(type(value) is np.float32 for value in loaded.values("loaded"))


def test_common_specialized_operations_reject_nonportable_role_dtypes(
    optional_backend,
) -> None:
    cutlass_coop = optional_backend("cutlass")

    from cuda import coop
    from cuda.coop._core import root_api

    float_items = cutlass_coop.ThreadData.from_values(
        np.float32(1),
        dtype=np.float32,
    )
    integer_items = cutlass_coop.ThreadData.from_values(
        np.int32(1),
        dtype=np.int32,
    )

    with root_api._compiler_scope("cuda.coop.cutlass"):
        block = coop.this_block()
        for operation in (
            coop.merge_sort_keys,
            coop.radix_sort_keys,
            coop.radix_rank,
        ):
            with pytest.raises(TypeError, match=r"key dtypes"):
                operation(block, float_items)

        with pytest.raises(TypeError, match=r"key dtypes"):
            coop.topk_max_keys(block, float_items, 1)
        with pytest.raises(TypeError, match=r"histogram.*sample dtypes"):
            coop.histogram(block, float_items, bins=32)
        with pytest.raises(TypeError, match=r"histogram.*counter dtypes"):
            coop.histogram(
                block,
                integer_items,
                bins=32,
                counter_dtype=np.float32,
            )
        with pytest.raises(TypeError, match=r"run_length_decode.*run_values dtypes"):
            coop.run_length_decode(
                block,
                float_items,
                integer_items,
                decoded_items_per_thread=1,
            )
