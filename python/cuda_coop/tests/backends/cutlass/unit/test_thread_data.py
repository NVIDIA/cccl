# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

import pytest

from ..support._subprocess import run_python_with_source


def test_portable_thread_data_rejects_a_qualified_only_dtype():
    pytest.importorskip("cutlass.cute.ffi")

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core.api._dispatch import _compiler_scope

    class StructuralUint64:
        width = 64

    with _compiler_scope("cuda.coop.cutlass"):
        with pytest.raises(TypeError, match="through the portable API"):
            coop.ThreadData(1, dtype=StructuralUint64)

    qualified = cutlass_coop.ThreadData(1, dtype=StructuralUint64)
    assert qualified.dtype is StructuralUint64


def test_thread_data_from_vector_like_payload():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class FakeVector:
            shape = (3,)
            dtype = "i32"

            def numel(self):
                return 3

            def __getitem__(self, idx):
                return f"item{idx}"

        items = coop.ThreadData.from_vector(FakeVector())

        assert len(items) == 3
        assert items.items_per_thread == 3
        assert items.dtype == "i32"
        assert items.values("test") == ("item0", "item1", "item2")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_fn_builds_generated_payloads():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class StaticInt:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class FakeDslUint32:
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        generated = coop.ThreadData.from_fn(
            StaticInt(3),
            lambda item: item + 10,
            dtype=FakeDslUint32,
        )
        assert generated.items_per_thread == 3
        assert generated.dtype is FakeDslUint32
        assert [value.value for value in generated.values("generated")] == [
            10,
            11,
            12,
        ]

        try:
            coop.ThreadData.from_fn(2, None)
        except TypeError as exc:
            assert "requires a callable" in str(exc)
        else:
            raise AssertionError("non-callable from_fn argument should fail")

        def fail(_):
            raise RuntimeError("boom")

        try:
            coop.ThreadData.from_fn(2, fail)
        except TypeError as exc:
            assert "callable failed for item 0" in str(exc)
        else:
            raise AssertionError("raising from_fn callable should fail")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_vector_infers_metadata_fallbacks():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class StaticInt:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class ShapeOnlyVector:
            _shape = (StaticInt(2),)
            _dtype = "fallback-i32"

            def numel(self):
                raise RuntimeError("probe fallback")

            def __getitem__(self, idx):
                return idx + 20

        class ElementTypeVector:
            shape = StaticInt(2)
            element_type = "element-i64"

            def __getitem__(self, idx):
                return idx + 30

        class RaisingNumelVector:
            shape = (2,)
            dtype = "raising-numel-i32"

            @property
            def numel(self):
                raise RuntimeError("numel probe failed")

            def __getitem__(self, idx):
                return idx + 35

        class ExplicitVector:
            def __getitem__(self, idx):
                return idx + 40

        class FakeDslUint32:
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        class RaisingDtypeVector:
            shape = (2,)

            @property
            def dtype(self):
                raise RuntimeError("dtype probe failed")

            @property
            def _dtype(self):
                raise RuntimeError("_dtype probe failed")

            @property
            def element_type(self):
                raise RuntimeError("element_type probe failed")

            def __getitem__(self, idx):
                return idx + 50

        shape_items = coop.ThreadData.from_vector(ShapeOnlyVector())
        assert shape_items.items_per_thread == 2
        assert shape_items.dtype == "fallback-i32"
        assert shape_items.values("shape") == (20, 21)

        element_items = coop.ThreadData.from_vector(ElementTypeVector())
        assert element_items.items_per_thread == 2
        assert element_items.dtype == "element-i64"
        assert element_items.values("element") == (30, 31)

        raising_numel_items = coop.ThreadData.from_vector(RaisingNumelVector())
        assert raising_numel_items.items_per_thread == 2
        assert raising_numel_items.dtype == "raising-numel-i32"
        assert raising_numel_items.values("raising-numel") == (35, 36)

        explicit_items = coop.ThreadData.from_vector(
            ExplicitVector(),
            items_per_thread=StaticInt(1),
            dtype="override-f32",
        )
        assert explicit_items.items_per_thread == 1
        assert explicit_items.dtype == "override-f32"
        assert explicit_items.values("explicit") == (40,)

        cast_items = coop.ThreadData.from_vector(
            ElementTypeVector(),
            dtype=FakeDslUint32,
        )
        assert cast_items.items_per_thread == 2
        assert cast_items.dtype is FakeDslUint32
        assert [value.value for value in cast_items.values("cast")] == [30, 31]
        assert all(
            isinstance(value, FakeDslUint32)
            for value in cast_items.values("cast")
        )

        raising_dtype_items = coop.ThreadData.from_vector(RaisingDtypeVector())
        assert raising_dtype_items.items_per_thread == 2
        assert raising_dtype_items.dtype is None
        assert raising_dtype_items.values("raising-dtype") == (50, 51)

        empty_items = coop.ThreadData(StaticInt(2), dtype="i8")
        assert empty_items.items_per_thread == 2
        assert empty_items.dtype == "i8"
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_vector_reports_bad_memory_payloads():
    script = textwrap.dedent(
        """
        import sys
        import types

        import cuda.coop.cutlass as coop

        class MissingShapeVector:
            def __getitem__(self, idx):
                return idx

        class NonIndexableVector:
            shape = (2,)

        class ShapedVector:
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class RaisingShapeVector:
            @property
            def shape(self):
                raise RuntimeError("shape should not be probed")

            def __getitem__(self, idx):
                return idx

        class RaisingMetadataVector:
            @property
            def shape(self):
                raise RuntimeError("shape probe failed")

            @property
            def _shape(self):
                raise RuntimeError("_shape probe failed")

            def __getitem__(self, idx):
                return idx

        class MemoryTensor:
            memspace = object()
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class MemoryArray:
            space = object()
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class CutlassArray:
            shape = (2,)
            dtype = "i32"

            def __getitem__(self, idx):
                return idx

            def load(self, *args, **kwargs):
                return None

            def store(self, *args, **kwargs):
                return None

        CutlassArray.__module__ = "cutlass.base_dsl.array"
        CutlassArray.__qualname__ = "Array"
        cutlass_module = types.ModuleType("cutlass")
        cutlass_module.Array = CutlassArray
        sys.modules["cutlass"] = cutlass_module

        class RaisingMemorySpaceVector:
            shape = (2,)

            @property
            def memspace(self):
                raise RuntimeError("memspace probe failed")

            def __getitem__(self, idx):
                return idx

        class RaisingSpaceVector:
            shape = (2,)

            @property
            def space(self):
                raise RuntimeError("space probe failed")

            def __getitem__(self, idx):
                return idx

        class HostArrayProtocol:
            __array_interface__ = {
                "shape": (2,),
                "typestr": "<i4",
                "data": (0, False),
                "version": 3,
            }
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class CudaArrayProtocol:
            __cuda_array_interface__ = {
                "shape": (2,),
                "typestr": "<i4",
                "data": (0, False),
                "version": 3,
            }
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class DlpackTensorProtocol:
            shape = (2,)

            def __dlpack__(self):
                raise RuntimeError("not called")

            def __getitem__(self, idx):
                return idx

        class RaisingCudaArrayProtocol:
            shape = (2,)

            @property
            def __cuda_array_interface__(self):
                raise RuntimeError("not called")

            def __getitem__(self, idx):
                return idx

        class RaisingDlpackDeviceProtocol:
            shape = (2,)

            @property
            def __dlpack_device__(self):
                raise RuntimeError("not called")

            def __getitem__(self, idx):
                return idx

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        assert_raises(
            ValueError,
            "could not infer items_per_thread",
            lambda: coop.ThreadData.from_vector(MissingShapeVector()),
        )
        assert_raises(
            ValueError,
            "could not infer items_per_thread",
            lambda: coop.ThreadData.from_vector(RaisingMetadataVector()),
        )
        assert_raises(
            TypeError,
            "requires integer-indexable vector items",
            lambda: coop.ThreadData.from_vector(NonIndexableVector()),
        )
        assert_raises(
            TypeError,
            "requires a CUTLASS vector-like per-thread payload",
            lambda: coop.ThreadData.from_vector(MemoryTensor()),
        )
        assert_raises(
            TypeError,
            "requires a CUTLASS vector-like per-thread payload",
            lambda: coop.ThreadData.from_vector(MemoryArray()),
        )
        for memory_payload in (
            HostArrayProtocol(),
            CudaArrayProtocol(),
            DlpackTensorProtocol(),
            CutlassArray(),
            RaisingMemorySpaceVector(),
            RaisingSpaceVector(),
            RaisingCudaArrayProtocol(),
            RaisingDlpackDeviceProtocol(),
        ):
            assert_raises(
                TypeError,
                "requires a CUTLASS vector-like per-thread payload",
                lambda memory_payload=memory_payload: coop.ThreadData.from_vector(
                    memory_payload
                ),
            )
        assert_raises(
            TypeError,
            "items_per_thread must be an integer",
            lambda: coop.ThreadData.from_vector(
                MissingShapeVector(),
                items_per_thread=True,
            ),
        )
        assert_raises(
            TypeError,
            "items_per_thread must be an integer",
            lambda: coop.ThreadData.from_vector(
                RaisingShapeVector(),
                items_per_thread=True,
            ),
        )
        assert_raises(
            ValueError,
            "items_per_thread must be a positive integer",
            lambda: coop.ThreadData.from_vector(
                MissingShapeVector(),
                items_per_thread=0,
            ),
        )
        assert_raises(
            ValueError,
            "items_per_thread does not match payload item count",
            lambda: coop.ThreadData.from_vector(
                ShapedVector(),
                items_per_thread=1,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_register_tensor_builds_cute_payload():
    script = textwrap.dedent(
        """
        import sys
        import types

        class AddressSpace:
            rmem = object()
            gmem = object()

        for module_name in (
            "cutlass",
            "cutlass._mlir",
            "cutlass._mlir.dialects",
        ):
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module
        cute_module = types.ModuleType("cutlass._mlir.dialects.cute")
        cute_module.AddressSpace = AddressSpace
        sys.modules["cutlass._mlir.dialects.cute"] = cute_module

        import cuda.coop.cutlass as coop

        class StaticInt:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class FakeDslUint32:
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        class RegisterTensor:
            memspace = AddressSpace.rmem
            shape = (StaticInt(3),)
            element_type = "i32"

            def __getitem__(self, idx):
                return f"r{idx}"

        class Layout:
            shape = (2,)

        class LayoutTensor:
            memspace = AddressSpace.rmem
            layout = Layout()
            element_type = "layout-i32"

            def __getitem__(self, idx):
                return f"layout{idx}"

        class ExplicitRegisterTensor:
            memspace = AddressSpace.rmem

            def __getitem__(self, idx):
                return f"explicit{idx}"

        class CastRegisterTensor:
            memspace = AddressSpace.rmem
            shape = (2,)

            def __getitem__(self, idx):
                return idx + 70

        class RaisingElementTypeTensor:
            memspace = AddressSpace.rmem
            shape = (2,)

            @property
            def element_type(self):
                raise RuntimeError("element_type probe failed")

            def __getitem__(self, idx):
                return f"raising{idx}"

        shape_items = coop.ThreadData.from_register_tensor(RegisterTensor())
        assert shape_items.items_per_thread == 3
        assert shape_items.dtype == "i32"
        assert shape_items.values("register") == ("r0", "r1", "r2")

        layout_items = coop.ThreadData.from_register_tensor(LayoutTensor())
        assert layout_items.items_per_thread == 2
        assert layout_items.dtype == "layout-i32"
        assert layout_items.values("layout") == ("layout0", "layout1")

        explicit_items = coop.ThreadData.from_register_tensor(
            ExplicitRegisterTensor(),
            items_per_thread=StaticInt(1),
            dtype="override-f32",
        )
        assert explicit_items.items_per_thread == 1
        assert explicit_items.dtype == "override-f32"
        assert explicit_items.values("explicit") == ("explicit0",)

        cast_items = coop.ThreadData.from_register_tensor(
            CastRegisterTensor(),
            dtype=FakeDslUint32,
        )
        assert cast_items.items_per_thread == 2
        assert cast_items.dtype is FakeDslUint32
        assert [
            value.value for value in cast_items.values("cast-register")
        ] == [70, 71]

        raising_element_items = coop.ThreadData.from_register_tensor(
            RaisingElementTypeTensor()
        )
        assert raising_element_items.items_per_thread == 2
        assert raising_element_items.dtype is None
        assert raising_element_items.values("raising-element") == (
            "raising0",
            "raising1",
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_payload_adapts_backend_payloads():
    script = textwrap.dedent(
        """
        import sys
        import types

        class AddressSpace:
            rmem = object()
            gmem = object()

        for module_name in (
            "cutlass",
            "cutlass._mlir",
            "cutlass._mlir.dialects",
        ):
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module
        cute_module = types.ModuleType("cutlass._mlir.dialects.cute")
        cute_module.AddressSpace = AddressSpace
        sys.modules["cutlass._mlir.dialects.cute"] = cute_module

        import cuda.coop.cutlass as coop

        class StaticInt:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        class RegisterTensor:
            memspace = AddressSpace.rmem
            shape = (2,)
            element_type = "i32"

            def __getitem__(self, idx):
                return f"r{idx}"

        class Vector:
            shape = (3,)
            dtype = "i64"

            def __getitem__(self, idx):
                return f"item{idx}"

        class CutlassArray:
            shape = (2,)
            dtype = "i32"

            def __getitem__(self, idx):
                return f"a{idx}"

            def load(self, *args, **kwargs):
                return None

            def store(self, *args, **kwargs):
                return None

        CutlassArray.__module__ = "cutlass.base_dsl.array"
        CutlassArray.__qualname__ = "Array"
        sys.modules["cutlass"].Array = CutlassArray

        class RaisingDtypeVector:
            shape = (2,)

            @property
            def dtype(self):
                raise RuntimeError("dtype probe failed")

            @property
            def _dtype(self):
                raise RuntimeError("_dtype probe failed")

            @property
            def element_type(self):
                raise RuntimeError("element_type probe failed")

            def __getitem__(self, idx):
                return f"vd{idx}"

        class FakeDslUint32:
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        from_register = coop.ThreadData.from_payload(RegisterTensor())
        assert from_register.items_per_thread == 2
        assert from_register.dtype == "i32"
        assert from_register.values("register") == ("r0", "r1")

        from_vector = coop.ThreadData.from_payload(Vector())
        assert from_vector.items_per_thread == 3
        assert from_vector.dtype == "i64"
        assert from_vector.values("vector") == ("item0", "item1", "item2")

        from_raising_dtype_vector = coop.ThreadData.from_payload(RaisingDtypeVector())
        assert from_raising_dtype_vector.items_per_thread == 2
        assert from_raising_dtype_vector.dtype is None
        assert from_raising_dtype_vector.values("raising-dtype-vector") == (
            "vd0",
            "vd1",
        )

        existing = coop.ThreadData.from_values(1, 2)
        assert coop.ThreadData.from_payload(existing) is existing
        typed_existing = coop.ThreadData.from_payload(existing, dtype="i32")
        assert typed_existing is not existing
        assert typed_existing.dtype == "i32"
        assert typed_existing.values("existing") == (1, 2)
        dsl_typed_existing = coop.ThreadData.from_payload(
            existing,
            dtype=FakeDslUint32,
        )
        assert dsl_typed_existing.dtype is FakeDslUint32
        assert [
            value.value for value in dsl_typed_existing.values("dsl-existing")
        ] == [1, 2]
        assert coop.ThreadData.from_payload(
            existing,
            items_per_thread=StaticInt(2),
        ) is existing

        typed_payload = coop.ThreadData.from_values(3, 4, dtype="i16")

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        assert_raises(
            TypeError,
            "items_per_thread must be an integer",
            lambda: coop.ThreadData.from_payload(existing, items_per_thread=True),
        )
        assert_raises(
            ValueError,
            "items_per_thread must be a positive integer",
            lambda: coop.ThreadData.from_payload(existing, items_per_thread=0),
        )
        assert_raises(
            ValueError,
            "items_per_thread does not match payload.items_per_thread",
            lambda: coop.ThreadData.from_payload(existing, items_per_thread=1),
        )
        assert_raises(
            TypeError,
            "dtype does not match payload",
            lambda: coop.ThreadData.from_payload(typed_payload, dtype="i32"),
        )
        assert_raises(
            TypeError,
            "a group-first load for cutlass.Array values",
            lambda: coop.ThreadData.from_payload(CutlassArray()),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_load_uses_explicit_producer_capability():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class RegisterVector:
            shape = (3,)
            dtype = "f32"

            def __getitem__(self, idx):
                return f"r{idx}"

        class ProducerSelectedLoad:
            # A producer wrapper may describe a memory-resident accumulator.
            # The explicit hook, rather than its address space, authorizes and
            # implements the tracing-time load.
            memspace = object()

            def __init__(self):
                self.calls = 0

            def __cuda_coop_thread_data_load__(self):
                self.calls += 1
                return RegisterVector()

        class ExistingPayload:
            def __init__(self, payload):
                self.payload = payload

            def __cuda_coop_thread_data_load__(self):
                return self.payload

        source = ProducerSelectedLoad()
        loaded = coop.ThreadData.load(
            source,
            items_per_thread=3,
            dtype="f32",
        )
        assert source.calls == 1
        assert loaded.items_per_thread == 3
        assert loaded.dtype == "f32"
        assert loaded.values("producer-load") == ("r0", "r1", "r2")

        existing = coop.ThreadData.from_values(10, 11, dtype="i32")
        assert coop.ThreadData.load(ExistingPayload(existing)) is existing
        assert hasattr(coop.ThreadDataLoadSource, "__cuda_coop_thread_data_load__")
        assert hasattr(coop.ThreadDataSource, "__cuda_coop_thread_data_load__")
        assert hasattr(coop.ThreadDataTensorMetadata, "shape")
        assert hasattr(coop.ThreadDataTensorMetadata, "dtype")
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_load_rejects_implicit_or_unsafe_sources():
    script = textwrap.dedent(
        """
        import cuda.coop.cutlass as coop

        class BareTmemTensor:
            memspace = object()
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class DynamicHook:
            def __getattr__(self, name):
                if name == "__cuda_coop_thread_data_load__":
                    return lambda: coop.ThreadData.from_values(1)
                raise AttributeError(name)

        class NonCallableHook:
            __cuda_coop_thread_data_load__ = None

        class RaisingHookProperty:
            @property
            def __cuda_coop_thread_data_load__(self):
                raise RuntimeError("hook access failed")

        class FailingHook:
            def __cuda_coop_thread_data_load__(self):
                raise RuntimeError("producer load failed")

        class MissingShapeVector:
            def __getitem__(self, idx):
                return idx

        class DynamicExtentHook:
            def __cuda_coop_thread_data_load__(self):
                return MissingShapeVector()

        class ReturnedMemoryHook:
            def __cuda_coop_thread_data_load__(self):
                return BareTmemTensor()

        class StaticVector:
            shape = (3,)
            dtype = "f32"

            def __getitem__(self, idx):
                return idx

        class CountingHook:
            def __init__(self):
                self.calls = 0

            def __cuda_coop_thread_data_load__(self):
                self.calls += 1
                return StaticVector()

        class TypedThreadDataHook:
            def __cuda_coop_thread_data_load__(self):
                return coop.ThreadData.from_values(1, 2, dtype="i32")

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        bare_tmem = BareTmemTensor()
        assert_raises(
            TypeError,
            "bare memory-backed tensors including TMEM",
            lambda: coop.ThreadData.load(bare_tmem),
        )
        # The existing generic adapter must remain strict as well.
        assert_raises(
            TypeError,
            "requires a per-thread register payload",
            lambda: coop.ThreadData.from_payload(bare_tmem),
        )
        assert_raises(
            TypeError,
            "requires source to define",
            lambda: coop.ThreadData.load(DynamicHook()),
        )
        assert_raises(
            TypeError,
            "hook to be callable",
            lambda: coop.ThreadData.load(NonCallableHook()),
        )
        assert_raises(
            TypeError,
            "could not access",
            lambda: coop.ThreadData.load(RaisingHookProperty()),
        )
        assert_raises(
            RuntimeError,
            "producer load failed",
            lambda: coop.ThreadData.load(FailingHook()),
        )
        assert_raises(
            ValueError,
            "statically sized per-thread register payload",
            lambda: coop.ThreadData.load(DynamicExtentHook()),
        )
        assert_raises(
            TypeError,
            "statically sized per-thread register payload",
            lambda: coop.ThreadData.load(ReturnedMemoryHook()),
        )
        unconsumed = CountingHook()
        assert_raises(
            ValueError,
            "items_per_thread must be a positive integer",
            lambda: coop.ThreadData.load(unconsumed, items_per_thread=0),
        )
        assert unconsumed.calls == 0
        assert_raises(
            ValueError,
            "CountingHook returned StaticVector",
            lambda: coop.ThreadData.load(
                CountingHook(),
                items_per_thread=2,
            ),
        )
        assert_raises(
            TypeError,
            "TypedThreadDataHook returned ThreadData",
            lambda: coop.ThreadData.load(
                TypedThreadDataHook(),
                dtype="f32",
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_thread_data_from_register_tensor_reports_bad_cute_payloads():
    script = textwrap.dedent(
        """
        import sys
        import types

        class AddressSpace:
            rmem = object()
            gmem = object()

        for module_name in (
            "cutlass",
            "cutlass._mlir",
            "cutlass._mlir.dialects",
        ):
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module
        cute_module = types.ModuleType("cutlass._mlir.dialects.cute")
        cute_module.AddressSpace = AddressSpace
        sys.modules["cutlass._mlir.dialects.cute"] = cute_module

        import cuda.coop.cutlass as coop

        class GlobalTensor:
            memspace = AddressSpace.gmem
            shape = (1,)

            def __getitem__(self, idx):
                return idx

        class MissingShapeTensor:
            memspace = AddressSpace.rmem

            def __getitem__(self, idx):
                return idx

        class ShapedTensor:
            memspace = AddressSpace.rmem
            shape = (2,)

            def __getitem__(self, idx):
                return idx

        class RaisingShapeTensor:
            memspace = AddressSpace.rmem

            @property
            def shape(self):
                raise RuntimeError("shape should not be probed")

            def __getitem__(self, idx):
                return idx

        class RaisingMetadataTensor:
            memspace = AddressSpace.rmem

            @property
            def shape(self):
                raise RuntimeError("shape probe failed")

            @property
            def layout(self):
                raise RuntimeError("layout probe failed")

            @property
            def type(self):
                raise RuntimeError("type probe failed")

            def __getitem__(self, idx):
                return idx

        class RaisingMemorySpaceTensor:
            @property
            def memspace(self):
                raise RuntimeError("memspace probe failed")

            def __getitem__(self, idx):
                return idx

        def assert_raises(exc_type, message, fn):
            try:
                fn()
            except exc_type as exc:
                assert message in str(exc), str(exc)
            else:
                raise AssertionError(f"{exc_type.__name__} was not raised")

        assert_raises(
            TypeError,
            "requires a register-memory",
            lambda: coop.ThreadData.from_register_tensor(GlobalTensor()),
        )
        assert_raises(
            TypeError,
            "requires a register-memory",
            lambda: coop.ThreadData.from_register_tensor(RaisingMemorySpaceTensor()),
        )
        assert_raises(
            ValueError,
            "could not infer items_per_thread",
            lambda: coop.ThreadData.from_register_tensor(MissingShapeTensor()),
        )
        assert_raises(
            ValueError,
            "could not infer items_per_thread",
            lambda: coop.ThreadData.from_register_tensor(RaisingMetadataTensor()),
        )
        assert_raises(
            TypeError,
            "items_per_thread must be an integer",
            lambda: coop.ThreadData.from_register_tensor(
                MissingShapeTensor(),
                items_per_thread=True,
            ),
        )
        assert_raises(
            TypeError,
            "items_per_thread must be an integer",
            lambda: coop.ThreadData.from_register_tensor(
                RaisingShapeTensor(),
                items_per_thread=True,
            ),
        )
        assert_raises(
            ValueError,
            "items_per_thread must be a positive integer",
            lambda: coop.ThreadData.from_register_tensor(
                MissingShapeTensor(),
                items_per_thread=0,
            ),
        )
        assert_raises(
            ValueError,
            "items_per_thread does not match payload item count",
            lambda: coop.ThreadData.from_register_tensor(
                ShapedTensor(),
                items_per_thread=1,
            ),
        )
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
