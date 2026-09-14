# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import gc
import importlib
import importlib.machinery
import importlib.util
import sys
import weakref
from importlib import metadata
from pathlib import Path

import pytest


def _is_extension_module(path):
    return any(
        path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def _find_built_extension():
    source_root = Path(__file__).resolve().parents[2]
    build_root = source_root / "build"
    if not build_root.exists():
        return None

    candidates = [
        path
        for path in build_root.rglob("_device_copy_impl*")
        if _is_extension_module(path)
    ]
    candidates.sort(key=lambda path: path.as_posix())
    return candidates[0] if candidates else None


def _find_installed_extension():
    for distribution_name in ("cuda-cccl", "cuda_cccl"):
        try:
            distribution = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            continue

        for package_file in distribution.files or ():
            package_path = Path(str(package_file))
            if (
                "cuda" in package_path.parts
                and "compute" in package_path.parts
                and any(part.startswith("cu") for part in package_path.parts)
                and package_path.name.startswith("_device_copy_impl")
                and _is_extension_module(package_path)
            ):
                return Path(distribution.locate_file(package_file))
    return None


def _load_extension_from_path(path):
    cuda_version_dir = next(
        (part for part in path.parts if part.startswith("cu") and part[2:].isdigit()),
        "cu",
    )
    module_name = f"cuda.compute.{cuda_version_dir}._device_copy_impl"
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture(scope="module")
def device_copy_impl():
    try:
        return importlib.import_module("cuda.compute._device_copy_impl")
    except (ImportError, RuntimeError) as import_error:
        extension_path = _find_built_extension() or _find_installed_extension()
        if extension_path is None:
            pytest.skip(
                f"cuda.compute v2 device copy extension is unavailable: {import_error}"
            )
        return _load_extension_from_path(extension_path)


DLPACK_MAJOR_VERSION = 1
DLPACK_MINOR_VERSION = 3
DLPACK_FLAG_BITMASK_READ_ONLY = 1
K_DLCUDA = 2
K_DLCUDA_MANAGED = 13


class DLPackVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.c_void_p),
]


class DLManagedTensorVersioned(ctypes.Structure):
    pass


DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.c_void_p),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]


class DLPackExchangeAPIHeader(ctypes.Structure):
    pass


DLPackExchangeAPIHeader._fields_ = [
    ("version", DLPackVersion),
    ("prev_api", ctypes.POINTER(DLPackExchangeAPIHeader)),
]


DLPACK_DLTENSOR_FROM_PY_OBJECT_NO_SYNC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(DLTensor),
)


DLPACK_MANAGED_TENSOR_FROM_PY_OBJECT_NO_SYNC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(DLManagedTensorVersioned)),
)


DLPACK_CURRENT_WORK_STREAM = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
)


class DLPackExchangeAPI(ctypes.Structure):
    _fields_ = [
        ("header", DLPackExchangeAPIHeader),
        ("managed_tensor_allocator", ctypes.c_void_p),
        ("managed_tensor_from_py_object_no_sync", ctypes.c_void_p),
        ("managed_tensor_to_py_object_no_sync", ctypes.c_void_p),
        ("dltensor_from_py_object_no_sync", ctypes.c_void_p),
        ("current_work_stream", ctypes.c_void_p),
    ]


_pycapsule_new = ctypes.pythonapi.PyCapsule_New
_pycapsule_new.restype = ctypes.py_object
_pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _capsule(pointer, name):
    return _pycapsule_new(ctypes.c_void_p(pointer), name, None)


def _int64_array(values):
    return (ctypes.c_int64 * len(values))(*values)


def _dltensor(
    shape,
    strides,
    *,
    data=0x1000,
    byte_offset=0,
    device_type=K_DLCUDA,
    device_id=0,
):
    shape_array = _int64_array(shape) if shape is not None else None
    strides_array = _int64_array(strides) if strides is not None else None
    tensor = DLTensor(
        ctypes.c_void_p(data),
        DLDevice(device_type, device_id),
        0 if shape is None else len(shape),
        DLDataType(0, 32, 1),
        shape_array,
        strides_array,
        byte_offset,
    )
    return tensor, shape_array, strides_array


def test_prepare_dlpack_view_from_versioned_capsule(device_copy_impl):
    tensor, shape, strides = _dltensor([2, 3], [3, -1], byte_offset=24)
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        7,
        tensor,
    )
    capsule = _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    view = device_copy_impl._prepare_dlpack_view(capsule)

    assert view.data_ptr == 0x1000
    assert view.byte_offset == 24
    assert view.device_type == K_DLCUDA
    assert view.device_id == 0
    assert view.flags == 7
    assert view.flags_known is True
    assert view.read_only is True
    assert view.rank == 2
    assert view.shape == (2, 3)
    assert view.strides == (3, -1)
    assert shape is not None
    assert strides is not None


def test_prepare_dlpack_view_from_legacy_capsule_with_compact_strides(
    device_copy_impl,
):
    tensor, shape, strides = _dltensor([2, 4, 5], None)
    managed = DLManagedTensor(tensor, None, None)
    capsule = _capsule(ctypes.addressof(managed), b"dltensor")

    view = device_copy_impl._prepare_dlpack_view(capsule)

    assert view.rank == 3
    assert view.flags == 0
    assert view.flags_known is False
    assert view.read_only is False
    assert view.shape == (2, 4, 5)
    assert view.strides == (20, 5, 1)
    assert shape is not None
    assert strides is None


def test_prepare_dlpack_view_from_zero_rank_tensor(device_copy_impl):
    tensor, shape, strides = _dltensor(None, None, data=0x2000)
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        0,
        tensor,
    )
    capsule = _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    view = device_copy_impl._prepare_dlpack_view(capsule)

    assert view.data_ptr == 0x2000
    assert view.rank == 1
    assert view.shape == (1,)
    assert view.strides == (1,)
    assert shape is None
    assert strides is None


def test_prepare_dlpack_view_calls_dlpack_with_latest_supported_version(
    device_copy_impl,
):
    tensor, shape, strides = _dltensor([4], [1])
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        0,
        tensor,
    )
    capsule = _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    class DLPackProducer:
        def __init__(self):
            self.calls = []

        def __dlpack__(self, *, stream=None, max_version=None):
            self.calls.append((stream, max_version))
            return capsule

    producer = DLPackProducer()

    view = device_copy_impl._prepare_dlpack_view(producer, stream=13)

    assert view.shape == (4,)
    assert producer.calls == [(13, (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))]
    assert shape is not None
    assert strides is not None


def test_prepare_dlpack_view_uses_c_exchange_api(device_copy_impl):
    tensor, shape, strides = _dltensor([4, 5], [5, 1], data=0x1234, byte_offset=8)
    calls = []
    stream_calls = []

    @DLPACK_DLTENSOR_FROM_PY_OBJECT_NO_SYNC
    def export_tensor(_py_object, out):
        calls.append(True)
        out[0] = tensor
        return 0

    @DLPACK_CURRENT_WORK_STREAM
    def current_work_stream(device_type, device_id, out):
        stream_calls.append((device_type, device_id))
        out[0] = None
        return 0

    api = DLPackExchangeAPI(
        DLPackExchangeAPIHeader(
            DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
            ctypes.POINTER(DLPackExchangeAPIHeader)(),
        ),
        None,
        None,
        None,
        ctypes.cast(export_tensor, ctypes.c_void_p),
        ctypes.cast(current_work_stream, ctypes.c_void_p),
    )

    class CExchangeProducer:
        __dlpack_c_exchange_api__ = _capsule(
            ctypes.addressof(api),
            b"dlpack_exchange_api",
        )

        def __dlpack__(self, *args, **kwargs):
            raise AssertionError("__dlpack__ should not be used")

    producer = CExchangeProducer()
    producer.api = api
    producer.export_tensor = export_tensor
    producer.current_work_stream = current_work_stream
    producer.shape = shape
    producer.strides = strides
    producer.tensor = tensor

    view = device_copy_impl._prepare_dlpack_view(producer)

    assert calls == [True]
    assert stream_calls == [(K_DLCUDA, 0)]
    assert view.data_ptr == 0x1234
    assert view.byte_offset == 8
    assert view.rank == 2
    assert view.shape == (4, 5)
    assert view.strides == (5, 1)


def test_prepare_dlpack_destination_uses_versioned_c_exchange_flags(
    device_copy_impl,
):
    tensor, shape, strides = _dltensor(
        [4],
        [1],
        data=0x1234,
        device_id=3,
    )
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        DLPACK_FLAG_BITMASK_READ_ONLY,
        tensor,
    )
    direct_calls = []
    managed_calls = []
    stream_calls = []

    @DLPACK_DLTENSOR_FROM_PY_OBJECT_NO_SYNC
    def export_tensor(_py_object, out):
        direct_calls.append(True)
        out[0] = tensor
        return 0

    @DLPACK_MANAGED_TENSOR_FROM_PY_OBJECT_NO_SYNC
    def export_managed_tensor(_py_object, out):
        managed_calls.append(True)
        out[0] = ctypes.pointer(managed)
        return 0

    @DLPACK_CURRENT_WORK_STREAM
    def current_work_stream(device_type, device_id, out):
        stream_calls.append((device_type, device_id))
        out[0] = None
        return 0

    api = DLPackExchangeAPI(
        DLPackExchangeAPIHeader(
            DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
            ctypes.POINTER(DLPackExchangeAPIHeader)(),
        ),
        None,
        ctypes.cast(export_managed_tensor, ctypes.c_void_p),
        None,
        ctypes.cast(export_tensor, ctypes.c_void_p),
        ctypes.cast(current_work_stream, ctypes.c_void_p),
    )

    class CExchangeProducer:
        __dlpack_c_exchange_api__ = _capsule(
            ctypes.addressof(api),
            b"dlpack_exchange_api",
        )

    producer = CExchangeProducer()
    producer.api = api
    producer.export_tensor = export_tensor
    producer.export_managed_tensor = export_managed_tensor
    producer.current_work_stream = current_work_stream
    producer.managed = managed
    producer.shape = shape
    producer.strides = strides

    view = device_copy_impl._prepare_dlpack_view(producer, require_flags=True)

    assert direct_calls == []
    assert managed_calls == [True]
    assert stream_calls == [(K_DLCUDA, 3)]
    assert view.device_id == 3
    assert view.flags == DLPACK_FLAG_BITMASK_READ_ONLY
    assert view.flags_known is True
    assert view.read_only is True


def test_prepare_dlpack_view_falls_back_when_c_exchange_stream_differs(
    device_copy_impl,
):
    tensor, shape, strides = _dltensor([4], [1])
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        0,
        tensor,
    )
    direct_calls = []
    dlpack_calls = []

    @DLPACK_DLTENSOR_FROM_PY_OBJECT_NO_SYNC
    def export_tensor(_py_object, out):
        direct_calls.append(True)
        out[0] = tensor
        return 0

    @DLPACK_CURRENT_WORK_STREAM
    def current_work_stream(_device_type, _device_id, out):
        out[0] = 0x55
        return 0

    api = DLPackExchangeAPI(
        DLPackExchangeAPIHeader(
            DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
            ctypes.POINTER(DLPackExchangeAPIHeader)(),
        ),
        None,
        None,
        None,
        ctypes.cast(export_tensor, ctypes.c_void_p),
        ctypes.cast(current_work_stream, ctypes.c_void_p),
    )

    class CExchangeProducer:
        __dlpack_c_exchange_api__ = _capsule(
            ctypes.addressof(api),
            b"dlpack_exchange_api",
        )

        def __dlpack__(self, **kwargs):
            dlpack_calls.append(kwargs)
            return _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    producer = CExchangeProducer()
    producer.api = api
    producer.export_tensor = export_tensor
    producer.current_work_stream = current_work_stream
    producer.managed = managed
    producer.shape = shape
    producer.strides = strides

    view = device_copy_impl._prepare_dlpack_view(producer)

    assert direct_calls == [True]
    assert dlpack_calls == [
        {"max_version": (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)}
    ]
    assert view.shape == (4,)


def test_prepare_dlpack_destination_preserves_flags_across_stream_fallback(
    device_copy_impl,
):
    tensor, shape, strides = _dltensor([4], [1])
    versioned = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        DLPACK_FLAG_BITMASK_READ_ONLY,
        tensor,
    )
    legacy = DLManagedTensor(tensor, None, None)

    @DLPACK_MANAGED_TENSOR_FROM_PY_OBJECT_NO_SYNC
    def export_managed_tensor(_py_object, out):
        out[0] = ctypes.pointer(versioned)
        return 0

    @DLPACK_CURRENT_WORK_STREAM
    def current_work_stream(_device_type, _device_id, out):
        out[0] = 0x55
        return 0

    api = DLPackExchangeAPI(
        DLPackExchangeAPIHeader(
            DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
            ctypes.POINTER(DLPackExchangeAPIHeader)(),
        ),
        None,
        ctypes.cast(export_managed_tensor, ctypes.c_void_p),
        None,
        None,
        ctypes.cast(current_work_stream, ctypes.c_void_p),
    )

    class CExchangeProducer:
        __dlpack_c_exchange_api__ = _capsule(
            ctypes.addressof(api),
            b"dlpack_exchange_api",
        )

        def __dlpack__(self, **_kwargs):
            return _capsule(ctypes.addressof(legacy), b"dltensor")

    producer = CExchangeProducer()
    producer.api = api
    producer.export_managed_tensor = export_managed_tensor
    producer.current_work_stream = current_work_stream
    producer.versioned = versioned
    producer.legacy = legacy
    producer.shape = shape
    producer.strides = strides

    view = device_copy_impl._prepare_dlpack_view(producer, require_flags=True)

    assert view.flags == DLPACK_FLAG_BITMASK_READ_ONLY
    assert view.flags_known is True
    assert view.read_only is True


def test_prepare_dlpack_view_rejects_non_cuda_tensor(device_copy_impl):
    tensor, shape, strides = _dltensor([1], [1], device_type=1)
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        0,
        tensor,
    )
    capsule = _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    with pytest.raises(BufferError, match="CUDA DLPack tensor"):
        device_copy_impl._prepare_dlpack_view(capsule)

    assert shape is not None
    assert strides is not None


def test_prepare_dlpack_view_rejects_negative_device_id(device_copy_impl):
    tensor, shape, strides = _dltensor([1], [1], device_id=-1)
    managed = DLManagedTensorVersioned(
        DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION),
        None,
        None,
        0,
        tensor,
    )
    capsule = _capsule(ctypes.addressof(managed), b"dltensor_versioned")

    with pytest.raises(BufferError, match="device id must be non-negative"):
        device_copy_impl._prepare_dlpack_view(capsule)

    assert shape is not None
    assert strides is not None


class Indexable:
    def __init__(self, value):
        self._value = value

    def __index__(self):
        return self._value


class Owner:
    pass


class CudaArrayInterfaceProducer:
    def __init__(
        self,
        data_ptr,
        shape,
        strides,
        *,
        device_type=K_DLCUDA,
        device_id=0,
        read_only=False,
    ):
        self.dtype = "int32"
        self.shape = shape
        self.device_type = device_type
        self.device_id = device_id
        self.__cuda_array_interface__ = {
            "data": (data_ptr, read_only),
            "shape": shape,
            "strides": strides,
            "typestr": "<i4",
            "version": 3,
        }

    def __dlpack_device__(self):
        return self.device_type, self.device_id


def test_DeviceCopy_rejects_read_only_destination(device_copy_impl):
    source = CudaArrayInterfaceProducer(0x1000, (0,), (4,))
    destination = CudaArrayInterfaceProducer(
        0x2000,
        (0,),
        (4,),
        read_only=True,
    )

    with pytest.raises(ValueError, match="destination is read-only"):
        device_copy_impl._make_device_copy(source, destination)


def test_DeviceCopy_rejects_different_cuda_devices(device_copy_impl):
    source = CudaArrayInterfaceProducer(0x1000, (0,), (4,), device_id=0)
    destination = CudaArrayInterfaceProducer(0x2000, (0,), (4,), device_id=1)

    with pytest.raises(ValueError, match="same device"):
        device_copy_impl._make_device_copy(source, destination)


def test_DeviceCopy_allows_managed_memory_with_a_different_device_id(
    device_copy_impl,
):
    source = CudaArrayInterfaceProducer(0, (0,), (4,), device_id=0)
    destination = CudaArrayInterfaceProducer(
        0,
        (0,),
        (4,),
        device_type=K_DLCUDA_MANAGED,
        device_id=1,
    )

    device_copy = device_copy_impl._make_device_copy(source, destination)
    device_copy.close()


def test_DeviceCopy_rejects_overlapping_memory_ranges(device_copy_impl):
    source = CudaArrayInterfaceProducer(0x1000, (4,), (4,))
    destination = CudaArrayInterfaceProducer(0x1008, (4,), (4,))

    with pytest.raises(ValueError, match="bounding memory spans overlap"):
        device_copy_impl._make_device_copy(source, destination)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type_info_key": ()},
        {"rank": 1},
    ],
)
def test_DeviceCopyBuild_rejects_partial_specialization(device_copy_impl, kwargs):
    with pytest.raises(TypeError, match="must either both be provided or both be None"):
        device_copy_impl._DeviceCopyBuild(**kwargs)


def test_runtime_axis_metadata(device_copy_impl):
    metadata = device_copy_impl._make_runtime_axis_metadata(3)

    assert metadata.rank == 3
    assert metadata._metadata_pointer() != 0
    assert metadata._as_tuple() == ((0, 0), (0, 0), (0, 0))


def test_zero_rank_metadata_maps_to_single_axis(device_copy_impl):
    metadata = device_copy_impl._make_runtime_axis_metadata(0)

    assert metadata.rank == 1
    assert metadata._as_tuple() == ((0, 0),)


def test_prepare_runtime_strided_view(device_copy_impl):
    owner = Owner()
    view = device_copy_impl._prepare_runtime_strided_view(
        owner,
        Indexable(0x1000),
        Indexable(32),
        [2, 3, 4],
        [12, -4, 1],
    )

    assert view.data_ptr == 0x1000
    assert view.byte_offset == 32
    assert view.device_type == K_DLCUDA
    assert view.device_id == 0
    assert view.flags == 0
    assert view.flags_known is False
    assert view.read_only is False
    assert view.rank == 3
    assert view.shape == (2, 3, 4)
    assert view.strides == (12, -4, 1)

    pointers = view._descriptor_pointers()
    assert pointers["shape"] != 0
    assert pointers["strides"] != 0


def test_zero_rank_view_maps_to_single_element(device_copy_impl):
    view = device_copy_impl._prepare_runtime_strided_view(Owner(), 0x1000, 0, (), None)

    assert view.rank == 1
    assert view.shape == (1,)
    assert view.strides == (1,)


def test_prepare_runtime_strided_view_keeps_owner_alive(device_copy_impl):
    owner = Owner()
    owner_ref = weakref.ref(owner)
    view = device_copy_impl._prepare_runtime_strided_view(owner, 0x1000, 0, [1], [1])

    del owner
    gc.collect()
    assert owner_ref() is not None

    del view
    gc.collect()
    assert owner_ref() is None


def test_as_device_array_view_materializes_and_reuses_native_view(
    device_copy_impl,
):
    producer = CudaArrayInterfaceProducer(
        0x1000,
        (2, 3),
        (12, 4),
        device_id=3,
        read_only=True,
    )

    view = device_copy_impl._as_device_array_view(producer)

    assert view.data_ptr == 0x1000
    assert view.byte_offset == 0
    assert view.device_type == K_DLCUDA
    assert view.device_id == 3
    assert view.read_only is True
    assert view.shape == (2, 3)
    assert view.strides == (3, 1)
    assert view.itemsize == 4
    assert view.alignment == 4
    assert device_copy_impl._as_device_array_view(view) is view


def test_as_device_array_view_keeps_protocol_owner_alive(device_copy_impl):
    producer = CudaArrayInterfaceProducer(0x1000, (1,), (4,))
    producer_ref = weakref.ref(producer)
    view = device_copy_impl._as_device_array_view(producer)

    del producer
    gc.collect()
    assert producer_ref() is not None

    del view
    gc.collect()
    assert producer_ref() is None


def test_as_device_array_view_rejects_native_view_without_dtype_metadata(
    device_copy_impl,
):
    view = device_copy_impl._prepare_runtime_strided_view(
        Owner(),
        0x1000,
        0,
        (1,),
        (1,),
    )

    with pytest.raises(TypeError, match="does not have dtype metadata"):
        device_copy_impl._as_device_array_view(view)


@pytest.mark.parametrize(
    "shape, strides",
    [
        ([-1], [1]),
        ([2, 3], [1]),
        ((), [1]),
    ],
)
def test_prepare_runtime_strided_view_rejects_invalid_shape_and_strides(
    device_copy_impl, shape, strides
):
    with pytest.raises(ValueError):
        device_copy_impl._prepare_runtime_strided_view(
            Owner(),
            0x1000,
            0,
            shape,
            strides,
        )


@pytest.mark.parametrize(
    "data_ptr, byte_offset, expected_error",
    [
        (-1, 0, ValueError),
        (0x1000, -1, ValueError),
        (2**64, 0, OverflowError),
        (0x1000, 2**64, OverflowError),
    ],
)
def test_prepare_runtime_strided_view_rejects_invalid_addresses(
    device_copy_impl, data_ptr, byte_offset, expected_error
):
    with pytest.raises(expected_error):
        device_copy_impl._prepare_runtime_strided_view(
            Owner(),
            data_ptr,
            byte_offset,
            [1],
            [1],
        )
