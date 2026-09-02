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
