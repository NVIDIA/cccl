# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import copy, deepcopy
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

from cuda.coop._core.api._dispatch import _common_root_operation_scope

cute = pytest.importorskip("cutlass.cute")
ir = import_module("cutlass._mlir.ir")
Int32 = import_module("cutlass.base_dsl.typing").Int32
_payload = import_module("cuda.coop.cutlass._thread_data")
_coerce_thread_payload = _payload._coerce_thread_payload
_make_rmem_tensor = _payload._make_rmem_tensor
_UNSET = _payload._UNSET
ThreadData = _payload.ThreadData

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


@pytest.mark.parametrize("alignment", [None, 1, 16, 32, 64, 128])
def test_payload_storage_and_copy(alignment):
    data = ThreadData(np.int64(3), np.int32, alignment=alignment)
    data[0] = 4
    data[-1] = 8
    assert data.items_per_thread == len(data) == 3
    assert data.dtype is np.int32
    assert data.alignment == alignment
    assert data[0] == 4
    assert data[2] == 8

    for cloned in (copy(data), deepcopy(data), data._new_uninitialized()):
        assert cloned.dtype is np.int32
        assert cloned.alignment == alignment
        assert cloned._values[1] is _UNSET
        with pytest.raises(ValueError, match="uninitialized"):
            cloned[1]

    for cloned in (copy(data), deepcopy(data)):
        cloned[0] = 42
        assert data[0] == 4
        cloned[1] = 6
        assert cloned.values("test") == (42, 6, 8)
        assert tuple(cloned) == (42, 6, 8)


@pytest.mark.parametrize(
    ("extent", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_extent_validation(extent, error):
    with pytest.raises(error, match="items_per_thread"):
        ThreadData(extent)


@pytest.mark.parametrize(
    ("alignment", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-2, ValueError),
        (3, ValueError),
    ],
)
def test_alignment_validation(alignment, error):
    with pytest.raises(error, match="alignment"):
        ThreadData(2, alignment=alignment)


def test_payload_indices():
    data = ThreadData(2)
    with pytest.raises(ValueError, match="initialized"):
        tuple(data)
    with pytest.raises(ValueError, match="initialized"):
        data.values("store")
    for index in (True, 1.0):
        with pytest.raises(TypeError, match="compile-time integer"):
            data[index]
        with pytest.raises(TypeError, match="compile-time integer"):
            data[index] = 1
    for index in (-3, 2):
        with pytest.raises(IndexError):
            data[index] = 1
    with pytest.raises(ValueError, match="values length"):
        ThreadData(2, values=[1])


def test_common_root_restrictions_survive_copy():
    with _common_root_operation_scope("ThreadData"):
        data = ThreadData(1, np.int16, alignment=64)
        with pytest.raises(TypeError, match="dtypes"):
            ThreadData(1, np.bool_)
    for cloned in (data, copy(data), deepcopy(data), data._new_uninitialized()):
        cloned[0] = np.int16(3)
        with pytest.raises(TypeError, match="dtypes"):
            cloned[0] = True


class _Vector:
    dtype = np.int16
    shape = (3,)

    def __getitem__(self, index):
        return index + 2


def test_qualified_vector_conversion_and_common_boundary():
    data = ThreadData.from_vector(_Vector())
    assert data.dtype is np.int16
    assert data.values("test") == (2, 3, 4)
    assert all(isinstance(value, np.int16) for value in data)
    assert ThreadData.from_payload(data) is data
    assert ThreadData.from_payload(_Vector()).values("test") == (2, 3, 4)
    with pytest.raises(ValueError, match="does not match"):
        ThreadData.from_vector(_Vector(), items_per_thread=2)
    with _common_root_operation_scope("store"):
        with pytest.raises(TypeError, match="backend-qualified"):
            _coerce_thread_payload(
                _Vector(),
                scope="cuda.coop.cutlass",
                primitive_name="store",
                arg_name="value",
                common_root_payload_kind="scalar_or_thread_data",
            )


def test_memory_payloads_require_explicit_register_conversion():
    global_tensor = SimpleNamespace(memspace=cute.AddressSpace.gmem, shape=(2,))
    with pytest.raises(TypeError, match="rmem"):
        ThreadData.from_register_tensor(global_tensor)
    with pytest.raises(TypeError, match="per-thread"):
        ThreadData.from_vector(global_tensor)
    with pytest.raises(TypeError, match="per-thread"):
        ThreadData.from_payload(np.array([1, 2], dtype=np.int32))


@pytest.mark.parametrize("value", [1, 2.0, np.int16(3), np.float64(4)])
def test_scalar_payloads_are_not_converted_to_vectors(value):
    assert (
        _coerce_thread_payload(
            value,
            scope="cuda.coop.cutlass",
            primitive_name="store",
            arg_name="value",
            common_root_payload_kind="scalar_or_thread_data",
        )
        is value
    )


@pytest.mark.parametrize("alignment", [None, 1, 32, 64, 128])
def test_rmem_allocation_honors_minimum_alignment(alignment):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = _make_rmem_tensor(2, Int32, alignment)
        expected = max(32, alignment or 1)
        assert f"align<{expected}>" in str(tensor.type)
        assert module.operation.verify()


def test_cute_register_conversions_use_payload_dtype_and_alignment():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            data = ThreadData(4, np.int32, values=[1, 3, 5, 7], alignment=128)
            ssa = data.to_tensor_ssa(shape=(2, 2))
            assert ssa.dtype is Int32
            assert ssa.shape == (2, 2)
            tensor = data.to_register_tensor()
            assert "align<128>" in str(tensor.type)
            copied = ThreadData.from_register_tensor(tensor)
            assert copied.dtype is Int32
            assert copied.items_per_thread == 4
            assert all(isinstance(value, Int32) for value in copied)
            vector = ThreadData.from_vector(data.to_tensor_ssa())
            assert vector.dtype is Int32
            assert vector.items_per_thread == 4
        assert module.operation.verify()


def test_export_rejects_incomplete_or_inconsistent_payload():
    with pytest.raises(ValueError, match="initialized"):
        ThreadData(2, Int32).to_tensor_ssa()
    with pytest.raises(ValueError, match="exactly"):
        ThreadData.from_values(1, 2, dtype=Int32).to_tensor_ssa(shape=(3,))
    with pytest.raises(ValueError, match="positive"):
        ThreadData.from_values(1, 2, dtype=Int32).to_tensor_ssa(shape=(2, 0))
    with pytest.raises(TypeError, match="requires dtype"):
        ThreadData.from_values(1, 2).to_tensor_ssa()


@pytest.mark.parametrize(
    "dtype",
    tuple(import_module("cuda.coop.cutlass._compiler._types").ALL_PROVIDER_TYPES),
)
@pytest.mark.parametrize("inferred", (False, True))
def test_control_flow_roundtrip(dtype, inferred):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            data = ThreadData(
                2,
                dtype=None if inferred else dtype,
                values=[dtype(3), dtype(5)],
                alignment=128,
            )
            data._common_root = True
            original = tuple(data._values)
            values = data.__extract_mlir_values__()
            assert len(values) == 2
            rebuilt = data.__new_from_mlir_values__(values)
            assert rebuilt is not data
            assert rebuilt.items_per_thread == 2
            assert rebuilt.dtype is dtype
            assert rebuilt.alignment == 128
            assert rebuilt._common_root
            assert all(isinstance(item, dtype) for item in rebuilt)
            assert all(before is after for before, after in zip(original, data._values))
            with pytest.raises(TypeError, match="dtypes"):
                rebuilt[0] = True
        assert module.operation.verify()


def test_control_flow_rejects_unset():
    data = ThreadData(3, Int32, values=[1, _UNSET, 3])
    with pytest.raises(ValueError, match=r"missing index\(es\): 1"):
        data.__extract_mlir_values__()
    with pytest.raises(ValueError, match="one value per item"):
        data.__new_from_mlir_values__([])


@pytest.mark.parametrize("dtype", (None, Int32))
def test_control_flow_checks_literals(dtype):
    with pytest.raises(ValueError, match="not representable"):
        ThreadData(2, dtype, values=[1 << 40, 2]).__extract_mlir_values__()


def test_control_flow_checks_lane_types():
    Uint32 = import_module("cutlass.base_dsl.typing").Uint32
    with pytest.raises(TypeError, match="homogeneous"):
        ThreadData(2, values=[Int32(1), Uint32(2)]).__extract_mlir_values__()
