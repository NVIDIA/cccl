import pytest

from cuda.compute import _device_copy_impl as impl
from cuda.compute import clear_all_caches

cp = pytest.importorskip("cupy")


def test_DeviceCopyCache_copy_into_handles_multiple_shapes():
    clear_all_caches()

    source = cp.arange(12, dtype=cp.int32).reshape(3, 4)
    destination = cp.empty_like(source)
    impl._copy_into(source, destination)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_array_equal(destination, source)

    source = cp.arange(20, dtype=cp.int32).reshape(4, 5)
    destination = cp.empty_like(source)
    impl._copy_into(source, destination)
    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_array_equal(destination, source)


def test_DeviceCopyCache_prepared_object_accepts_lower_simplified_rank():
    clear_all_caches()

    source_for_build = cp.arange(30, dtype=cp.int32).reshape(2, 3, 5)[:, :, ::2]
    destination_for_build = cp.empty(
        source_for_build.shape, dtype=source_for_build.dtype
    )
    device_copy = impl._make_device_copy(source_for_build, destination_for_build)

    try:
        source = cp.arange(12, dtype=cp.int32).reshape(3, 4)
        destination = cp.empty_like(source)
        device_copy(source, destination)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_array_equal(destination, source)
    finally:
        device_copy.close()


def test_DeviceCopyCache_prepared_object_rejects_higher_input_rank():
    clear_all_caches()

    source_for_build = cp.arange(6, dtype=cp.int32).reshape(2, 3)
    destination_for_build = cp.empty_like(source_for_build)
    device_copy = impl._make_device_copy(source_for_build, destination_for_build)

    try:
        source = cp.arange(16, dtype=cp.int32).reshape(2, 2, 2, 2)
        destination = cp.empty_like(source)
        with pytest.raises(ValueError, match="runtime rank exceeds prepared rank"):
            device_copy(source, destination)
    finally:
        device_copy.close()


def test_DeviceCopyCache_precompile_all_policy_is_accepted():
    clear_all_caches()

    source = cp.arange(30, dtype=cp.int32).reshape(2, 3, 5)[:, :, ::2]
    destination = cp.empty(source.shape, dtype=source.dtype)
    device_copy = impl._make_device_copy(source, destination, precompile="all")

    try:
        device_copy(source, destination)
        cp.cuda.get_current_stream().synchronize()
        cp.testing.assert_array_equal(destination, source)
    finally:
        device_copy.close()


def test_DeviceCopyCache_rejects_unknown_precompile_policy():
    source = cp.arange(6, dtype=cp.int32).reshape(2, 3)
    destination = cp.empty_like(source)

    with pytest.raises(ValueError, match="precompile"):
        impl._make_device_copy(source, destination, precompile="eager")
