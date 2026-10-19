import pytest

from cuda.compute import _device_copy_impl as impl


def test_DeviceCopyCompileSpec_default_extents_are_dynamic():
    spec = impl._device_copy_compile_spec()

    assert spec.all_static_extents is False
    assert spec.static_extent_axes == ()


def test_DeviceCopyCompileSpec_accepts_all_static_extent_token():
    spec = impl._device_copy_compile_spec(extents="static")

    assert spec.all_static_extents is True
    assert spec.static_extent_axes == ()


def test_DeviceCopyCompileSpec_accepts_sparse_static_extent_kinds():
    spec = impl._device_copy_compile_spec(
        extents=("dynamic", "static", None, "runtime")
    )

    assert spec.all_static_extents is False
    assert spec.static_extent_axes == (1,)


def test_DeviceCopyCompileSpec_accepts_sparse_static_extent_axes():
    spec = impl._device_copy_compile_spec(static_extents=(3, 1))

    assert spec.all_static_extents is False
    assert set(spec.static_extent_axes) == {1, 3}


def test_DeviceCopyCompileSpec_rejects_duplicate_static_extent_axes():
    with pytest.raises(ValueError):
        impl._device_copy_compile_spec(static_extents=(1, 1))


def test_DeviceCopyCompileSpec_rejects_negative_static_extent_axis():
    with pytest.raises(ValueError):
        impl._device_copy_compile_spec(static_extents=(-1,))
