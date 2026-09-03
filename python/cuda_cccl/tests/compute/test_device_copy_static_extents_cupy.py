import pytest

from cuda.compute import _device_copy_impl as impl

cp = pytest.importorskip("cupy")


def test_DeviceCopy_default_compile_spec_keeps_extents_dynamic():
    source = cp.arange(8, dtype=cp.int32)
    destination = cp.empty_like(source)

    with impl._make_device_copy(source, destination) as device_copy:
        generated_source = device_copy._get_source()

        assert (
            "std::extents<index_type, ::cuda::std::dynamic_extent>" in generated_source
        )
        assert (
            "const extents_type extents{static_cast<index_type>(source_shape[0])};"
            in generated_source
        )

        device_copy(source, destination)
        cp.cuda.runtime.deviceSynchronize()
        cp.testing.assert_array_equal(destination, source)

        next_source = cp.arange(11, dtype=cp.int32)
        next_destination = cp.empty_like(next_source)

        device_copy(next_source, next_destination)
        cp.cuda.runtime.deviceSynchronize()
        cp.testing.assert_array_equal(next_destination, next_source)


def test_DeviceCopy_all_static_extents_reject_changed_runtime_extent():
    source = cp.arange(8, dtype=cp.int32)
    destination = cp.empty_like(source)
    compile_spec = impl._device_copy_compile_spec(extents="static")

    with impl._make_device_copy(
        source,
        destination,
        compile_spec=compile_spec,
    ) as device_copy:
        generated_source = device_copy._get_source()

        assert "std::extents<index_type, 8>" in generated_source
        assert "const extents_type extents{};" in generated_source

        device_copy(source, destination)
        cp.cuda.runtime.deviceSynchronize()
        cp.testing.assert_array_equal(destination, source)

        next_source = cp.arange(9, dtype=cp.int32)
        next_destination = cp.empty_like(next_source)

        with pytest.raises(RuntimeError, match="cccl_device_copy failed"):
            device_copy(next_source, next_destination)


def test_DeviceCopy_sparse_static_extents_allow_dynamic_axes_only():
    source_base = cp.arange(21, dtype=cp.int32).reshape(3, 7)
    source = source_base[:, :4]
    destination = cp.empty((3, 4), dtype=cp.int32)
    compile_spec = impl._device_copy_compile_spec(static_extents=(1,))

    with impl._make_device_copy(
        source,
        destination,
        compile_spec=compile_spec,
    ) as device_copy:
        generated_source = device_copy._get_source()

        assert (
            "std::extents<index_type, ::cuda::std::dynamic_extent, 4>"
            in generated_source
        )
        assert (
            "const extents_type extents{static_cast<index_type>(source_shape[0])};"
            in generated_source
        )

        device_copy(source, destination)
        cp.cuda.runtime.deviceSynchronize()
        cp.testing.assert_array_equal(destination, source)

        dynamic_source_base = cp.arange(35, dtype=cp.int32).reshape(5, 7)
        dynamic_source = dynamic_source_base[:, :4]
        dynamic_destination = cp.empty((5, 4), dtype=cp.int32)

        device_copy(dynamic_source, dynamic_destination)
        cp.cuda.runtime.deviceSynchronize()
        cp.testing.assert_array_equal(dynamic_destination, dynamic_source)

        static_mismatch_source_base = cp.arange(40, dtype=cp.int32).reshape(5, 8)
        static_mismatch_source = static_mismatch_source_base[:, :5]
        static_mismatch_destination = cp.empty((5, 5), dtype=cp.int32)

        with pytest.raises(RuntimeError, match="cccl_device_copy failed"):
            device_copy(static_mismatch_source, static_mismatch_destination)


def test_DeviceCopy_rejects_changed_simplified_rank_before_calling_c_api():
    source_base = cp.arange(21, dtype=cp.int32).reshape(3, 7)
    source = source_base[:, :4]
    destination = cp.empty((3, 4), dtype=cp.int32)

    with impl._make_device_copy(source, destination) as device_copy:
        rank_one_source = cp.arange(12, dtype=cp.int32)
        rank_one_destination = cp.empty_like(rank_one_source)

        with pytest.raises(ValueError, match="different simplified rank"):
            device_copy(rank_one_source, rank_one_destination)
