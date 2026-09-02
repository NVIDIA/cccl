import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


def test_DeviceCopyImpl_get_cubin_returns_generated_artifact():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    device_copy = impl._make_device_copy(source, destination)
    try:
        cubin = device_copy._get_cubin()
    finally:
        device_copy.close()

    assert isinstance(cubin, bytes)
    assert len(cubin) > 0
