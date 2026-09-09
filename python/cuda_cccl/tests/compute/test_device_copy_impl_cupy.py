import importlib

import pytest


def _import_or_skip(module_name):
    try:
        return importlib.import_module(module_name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"{module_name} is unavailable: {exc}")


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CuPy did not find a CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.fixture(scope="module")
def device_copy_impl():
    return _import_or_skip("cuda.compute._device_copy_impl")


def test_DeviceCopyImpl_copy_into_cupy_contiguous(cp, device_copy_impl):
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    device_copy_impl._copy_into(source, destination)
    cp.cuda.Device().synchronize()

    cp.testing.assert_array_equal(destination, source)


def test_DeviceCopyImpl_copy_into_cupy_strided_views(cp, device_copy_impl):
    source_base = cp.arange(6 * 8, dtype=cp.int32).reshape(6, 8)
    source = source_base[1:5, ::2]

    destination_base = cp.full((7, 10), -1, dtype=cp.int32)
    destination = destination_base[2:6, 1:9:2]

    expected = cp.full((7, 10), -1, dtype=cp.int32)
    expected[2:6, 1:9:2] = source

    device_copy_impl._copy_into(source, destination)
    cp.cuda.Device().synchronize()

    cp.testing.assert_array_equal(destination_base, expected)


def test_DeviceCopyImpl_copy_into_rejects_same_size_different_dtype(
    cp, device_copy_impl
):
    source = cp.arange(16, dtype=cp.uint64)
    destination = cp.empty(16, dtype=cp.float64)

    with pytest.raises(TypeError, match="dtypes must match"):
        device_copy_impl._copy_into(source, destination)


@pytest.mark.parametrize("prepared", [False, True])
def test_DeviceCopyImpl_assume_non_overlapping_allows_disjoint_interleaved_views(
    cp, device_copy_impl, prepared
):
    values = cp.arange(32, dtype=cp.int32)
    source = values[::2]
    destination = values[1::2]
    expected = source.copy()

    with pytest.raises(ValueError, match="bounding memory spans overlap"):
        if prepared:
            device_copy_impl._make_device_copy(source, destination)
        else:
            device_copy_impl._copy_into(source, destination)

    if prepared:
        with device_copy_impl._make_device_copy(
            source,
            destination,
            assume_non_overlapping=True,
        ) as device_copy:
            device_copy(source, destination)
    else:
        device_copy_impl._copy_into(
            source,
            destination,
            assume_non_overlapping=True,
        )

    cp.cuda.Device().synchronize()
    cp.testing.assert_array_equal(destination, expected)


class _CountingDLPackArray:
    def __init__(self, array):
        self.array = array
        self.dlpack_calls = []

    def __dlpack__(self, *args, **kwargs):
        self.dlpack_calls.append((args, kwargs))
        return self.array.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()


def test_DeviceCopyImpl_reuses_native_views_without_reacquiring_dlpack(
    cp, device_copy_impl
):
    source = cp.arange(24, dtype=cp.int32).reshape(4, 6)[:, ::2]
    destination = cp.empty_like(source)
    source_producer = _CountingDLPackArray(source)
    destination_producer = _CountingDLPackArray(destination)

    source_view = device_copy_impl._as_device_array_view(source_producer)
    destination_view = device_copy_impl._as_device_array_view(destination_producer)
    source_calls = len(source_producer.dlpack_calls)
    destination_calls = len(destination_producer.dlpack_calls)

    device_copy_impl._copy_into(source_view, destination_view)
    with device_copy_impl._make_device_copy(
        source_view,
        destination_view,
    ) as device_copy:
        source *= 2
        device_copy(source_view, destination_view)
    cp.cuda.Device().synchronize()

    assert len(source_producer.dlpack_calls) == source_calls
    assert len(destination_producer.dlpack_calls) == destination_calls
    cp.testing.assert_array_equal(destination, source)
