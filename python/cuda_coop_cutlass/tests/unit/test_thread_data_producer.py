# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check producer invocation and metadata-only register reconstruction."""

from types import SimpleNamespace

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass._mlir import ir

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _Vector:
    shape = (3,)
    dtype = np.int32

    def __getitem__(self, index):
        return np.int32(index + 2)


class _Producer:
    # The producer, not its storage metadata, selects a legal register load.
    memspace = object()

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __cuda_coop_thread_data_load__(self):
        self.calls += 1
        return self.payload


def test_producer_load_once_and_existing_payload_identity():
    source = _Producer(_Vector())
    loaded = coop.ThreadData.load(source, items_per_thread=3, dtype=np.int32)
    assert source.calls == 1
    # Public protocols support their documented generic annotation at runtime.
    assert coop.ThreadDataLoadSource[np.int32] is not None
    assert coop.ThreadDataSource[np.int32] is not None
    assert loaded.dtype is np.int32
    assert loaded.values("test") == (2, 3, 4)
    original = coop.ThreadData(2, np.int32, values=[5, 7], alignment=128)
    source = _Producer(original)
    assert coop.ThreadData.load(source) is original
    assert source.calls == 1
    assert original.alignment == 128


@pytest.mark.parametrize("count", (True, 0, -1, 1.5))
def test_invalid_extent_does_not_consume_producer(count):
    source = _Producer(_Vector())
    with pytest.raises((TypeError, ValueError), match="items_per_thread"):
        coop.ThreadData.load(source, items_per_thread=count)
    assert source.calls == 0


def test_bare_memory_and_dynamic_hook_do_not_choose_copy_policy():
    memory = SimpleNamespace(memspace=cute.AddressSpace.tmem, shape=(2,))
    with pytest.raises(TypeError, match="bare memory-backed tensors.*TMEM"):
        coop.ThreadData.load(memory)
    with pytest.raises(TypeError, match="per-thread register payload"):
        coop.ThreadData.from_payload(memory)

    class DynamicHook:
        def __getattr__(self, name):
            if name == "__cuda_coop_thread_data_load__":
                raise AssertionError("implicit hook must not be read")
            raise AttributeError(name)

    with pytest.raises(TypeError, match="requires source to define"):
        coop.ThreadData.load(DynamicHook())


def test_hook_access_and_execution_errors():
    class InaccessibleHook:
        @property
        def __cuda_coop_thread_data_load__(self):
            raise RuntimeError("access failed")

    with pytest.raises(TypeError, match="could not access") as error:
        coop.ThreadData.load(InaccessibleHook())
    assert isinstance(error.value.__cause__, RuntimeError)
    with pytest.raises(TypeError, match="hook to be callable"):
        coop.ThreadData.load(SimpleNamespace(__cuda_coop_thread_data_load__=1))

    failure = RuntimeError("producer failure")

    class FailingProducer:
        def __cuda_coop_thread_data_load__(self):
            raise failure

    with pytest.raises(RuntimeError) as error:
        coop.ThreadData.load(FailingProducer())
    assert error.value is failure


def test_result_contract_diagnostics_and_explicit_indexable_extent():
    class Indexable:
        dtype = np.int32

        def __getitem__(self, index):
            return np.int32(index)

    source = _Producer(Indexable())
    with pytest.raises(ValueError, match="_Producer returned Indexable"):
        coop.ThreadData.load(source)
    assert source.calls == 1
    assert tuple(coop.ThreadData.load(source, items_per_thread=2)) == (0, 1)
    with pytest.raises(ValueError, match="_Producer returned _Vector"):
        coop.ThreadData.load(_Producer(_Vector()), items_per_thread=2)
    with pytest.raises(TypeError, match="_Producer returned ThreadData"):
        coop.ThreadData.load(
            _Producer(coop.ThreadData.from_values(np.int32(1), dtype=np.int32)),
            dtype=np.float32,
        )
    memory = SimpleNamespace(memspace=cute.AddressSpace.smem, shape=(2,))
    with pytest.raises(TypeError, match="per-thread register payload"):
        coop.ThreadData.load(_Producer(memory))


def test_like_metadata_preserves_item_order_and_explicit_precedence():
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            values = coop.ThreadData.from_values(
                1, 2, 3, 4, dtype=cutlass.Int32
            )
            like = SimpleNamespace(shape=((2, 1), 2), dtype=cutlass.Float32)
            result = values.to_tensor_ssa(like=like)
            assert result.shape == ((2, 1), 2)
            assert result.dtype is cutlass.Float32
            roundtrip = coop.ThreadData.from_payload(result)
            assert roundtrip.items_per_thread == 4
            assert all(isinstance(item, cutlass.Float32) for item in roundtrip)

            # Explicit metadata avoids even reading the corresponding property.
            class InaccessibleMetadata:
                @property
                def shape(self):
                    raise AssertionError("shape accessed")

                @property
                def dtype(self):
                    raise AssertionError("dtype accessed")

            explicit = values.to_tensor_ssa(
                shape=(4,), dtype=cutlass.Int32, like=InaccessibleMetadata()
            )
            assert explicit.shape == (4,)
            assert explicit.dtype is cutlass.Int32
        assert module.operation.verify()


@pytest.mark.parametrize(
    "metadata,kwargs,message",
    [
        (object(), {}, "accessible dtype"),
        (SimpleNamespace(dtype=None), {}, "non-None dtype"),
        (object(), {"dtype": cutlass.Int32}, "accessible shape"),
        (
            SimpleNamespace(shape=None),
            {"dtype": cutlass.Int32},
            "non-None shape",
        ),
        (SimpleNamespace(shape=(3,), dtype=cutlass.Int32), {}, "shape"),
    ],
)
def test_like_metadata_validation(metadata, kwargs, message):
    values = coop.ThreadData.from_values(1, 2, dtype=cutlass.Int32)
    with pytest.raises((TypeError, ValueError), match=message):
        values.to_tensor_ssa(like=metadata, **kwargs)
