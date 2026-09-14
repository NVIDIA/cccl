# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import importlib
import weakref

import numpy as np
import pytest
from cuda.core import Device

import cuda.compute._caching as _caching
from cuda.compute import clear_all_caches


def _import_or_skip(module_name):
    try:
        return importlib.import_module(module_name)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(
            f"{module_name} is unavailable: {exc}",
            allow_module_level=True,
        )


cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CuPy did not find a CUDA device", allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError as exc:
    pytest.skip(
        f"CuPy CUDA runtime is unavailable: {exc}",
        allow_module_level=True,
    )

impl = _import_or_skip("cuda.compute._device_copy_impl")


# Execution and reusable views


def test_DeviceCopyImpl_copy_into_cupy_contiguous():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    impl._copy_into(source, destination)
    cp.cuda.Device().synchronize()

    cp.testing.assert_array_equal(destination, source)


def test_DeviceCopyImpl_copy_into_cupy_strided_views():
    source_base = cp.arange(6 * 8, dtype=cp.int32).reshape(6, 8)
    source = source_base[1:5, ::2]

    destination_base = cp.full((7, 10), -1, dtype=cp.int32)
    destination = destination_base[2:6, 1:9:2]

    expected = cp.full((7, 10), -1, dtype=cp.int32)
    expected[2:6, 1:9:2] = source

    impl._copy_into(source, destination)
    cp.cuda.Device().synchronize()

    cp.testing.assert_array_equal(destination_base, expected)


def test_DeviceCopyImpl_copy_into_rejects_same_size_different_dtype():
    source = cp.arange(16, dtype=cp.uint64)
    destination = cp.empty(16, dtype=cp.float64)

    with pytest.raises(TypeError, match="dtypes must match"):
        impl._copy_into(source, destination)


@pytest.mark.parametrize("prepared", [False, True])
def test_DeviceCopyImpl_assume_non_overlapping_allows_disjoint_interleaved_views(
    prepared,
):
    values = cp.arange(32, dtype=cp.int32)
    source = values[::2]
    destination = values[1::2]
    expected = source.copy()

    with pytest.raises(ValueError, match="bounding memory spans overlap"):
        if prepared:
            impl._make_device_copy(source, destination)
        else:
            impl._copy_into(source, destination)

    if prepared:
        with impl._make_device_copy(
            source,
            destination,
            assume_non_overlapping=True,
        ) as device_copy:
            device_copy(source, destination)
    else:
        impl._copy_into(
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


def test_DeviceCopyImpl_reuses_native_views_without_reacquiring_dlpack():
    source = cp.arange(24, dtype=cp.int32).reshape(4, 6)[:, ::2]
    destination = cp.empty_like(source)
    source_producer = _CountingDLPackArray(source)
    destination_producer = _CountingDLPackArray(destination)

    source_view = impl._as_device_array_view(source_producer)
    destination_view = impl._as_device_array_view(destination_producer)
    source_calls = len(source_producer.dlpack_calls)
    destination_calls = len(destination_producer.dlpack_calls)

    impl._copy_into(source_view, destination_view)
    with impl._make_device_copy(
        source_view,
        destination_view,
    ) as device_copy:
        source *= 2
        device_copy(source_view, destination_view)
    cp.cuda.Device().synchronize()

    assert len(source_producer.dlpack_calls) == source_calls
    assert len(destination_producer.dlpack_calls) == destination_calls
    cp.testing.assert_array_equal(destination, source)


# Protocol fallback


class _NoDLPackArray:
    def __init__(self, array, *, device_id=None):
        self.array = array
        self.dtype = array.dtype
        self.device_id = cp.cuda.runtime.getDevice() if device_id is None else device_id

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("test array intentionally does not expose DLPack")

    def __dlpack_device__(self):
        return 2, self.device_id


def test_DeviceCopyImpl_protocol_fallback_when_dlpack_is_unavailable():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    impl._copy_into(_NoDLPackArray(source), _NoDLPackArray(destination))

    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(destination, source)


def test_DeviceCopyImpl_protocol_fallback_rejects_non_execution_device():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)
    reported_device = cp.cuda.runtime.getDevice() + 1

    with pytest.raises(ValueError, match="source is not on the execution device"):
        impl._make_device_copy(
            _NoDLPackArray(source, device_id=reported_device),
            _NoDLPackArray(destination, device_id=reported_device),
        )


# Generated artifacts and specialization


def test_DeviceCopyImpl_get_cubin_returns_generated_artifact():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    with impl._make_device_copy(source, destination) as device_copy:
        cubin = device_copy._get_cubin()

    assert isinstance(cubin, bytes)
    assert len(cubin) > 0


def test_DeviceCopy_get_source_returns_generated_source():
    source = cp.arange(16, dtype=cp.int32)
    destination = cp.empty_like(source)

    with impl._make_device_copy(source, destination) as device_copy:
        generated_source = device_copy._get_source()

    assert isinstance(generated_source, str)
    assert (
        'extern "C" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy'
        in generated_source
    )
    assert "cub::DeviceCopy::Copy" in generated_source


def test_DeviceCopy_uses_layout_stride_for_positive_runtime_strides():
    source_base = cp.arange(24, dtype=cp.int32).reshape(4, 6)
    source = source_base[:, ::2]
    destination_base = cp.empty((3, 4), dtype=cp.int32)
    destination = destination_base.T

    with impl._make_device_copy(source, destination) as device_copy:
        generated_source = device_copy._get_source()
        device_copy(source, destination)

    assert "using input_type = ::cuda::std::mdspan" in generated_source
    assert "using output_type = ::cuda::std::mdspan" in generated_source
    assert (
        "using input_type_layout_type = "
        "cccl_device_copy_layout_stride_relaxed<runtime_strides_type, offset_type>"
        in generated_source
    )
    assert (
        "using output_type_layout_type = ::cuda::std::layout_stride;"
        in generated_source
    )
    assert (
        "destination_view_mapping_type{extents, destination_view_strides}"
        in generated_source
    )
    cp.testing.assert_array_equal(destination, source)


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

    compile_spec = impl._device_copy_compile_spec()

    with impl._make_device_copy(
        source, destination, compile_spec=compile_spec
    ) as device_copy:
        rank_one_source = cp.arange(12, dtype=cp.int32)
        rank_one_destination = cp.empty_like(rank_one_source)

        with pytest.raises(ValueError, match="different simplified rank"):
            device_copy(rank_one_source, rank_one_destination)


# Caching


def _device_copy_build_cache_entries():
    return [
        value
        for key, value in _caching._process_wide_build_results_cache.items()
        if key[0].__name__ == "_DeviceCopyBuild"
    ]


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


def test_DeviceCopyCache_reuses_process_build_after_wrapper_cache_clear():
    clear_all_caches()

    source = cp.arange(27, dtype=cp.int32).reshape(3, 9)[:, :8:2]
    destination = cp.empty(source.shape, dtype=source.dtype)
    impl._copy_into(source, destination)
    cp.cuda.get_current_stream().synchronize()

    [first_build_results] = _device_copy_build_cache_entries()

    # Prepared and one-shot calls with the default target share the same key.
    with impl._make_device_copy(source, destination):
        [prepared_build_results] = _device_copy_build_cache_entries()
        assert prepared_build_results is first_build_results

    # Force construction of a new executable wrapper. Its native build still
    # comes from the process-wide build-results cache.
    impl._clear_device_copy_cache()
    with impl._make_device_copy(source, destination):
        [rebuilt_wrapper_results] = _device_copy_build_cache_entries()
        assert rebuilt_wrapper_results is first_build_results


def test_DeviceCopyCache_explicit_compute_capability_uses_cached_build_result():
    clear_all_caches()

    source = cp.arange(12, dtype=cp.int32)
    destination = cp.empty_like(source)
    compute_capability = tuple(Device().compute_capability)

    with impl._make_device_copy(
        source,
        destination,
        compute_capability=compute_capability,
    ) as device_copy:
        device_copy(source, destination)

    cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_array_equal(destination, source)
    assert len(_device_copy_build_cache_entries()) == 1


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


# Asynchronous ownership


class _StreamProtocol:
    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return 0, self._stream.ptr


_delay_kernel = cp.RawKernel(
    r"""
    extern "C" __global__ void delay(unsigned long long cycles)
    {
      const unsigned long long start = clock64();
      while (clock64() - start < cycles)
      {
        __nanosleep(1000);
      }
    }
    """,
    "delay",
)


@pytest.mark.parametrize("view_kind", ["protocol", "prepared"])
def test_DeviceCopy_keeps_view_owners_alive_until_stream_completion(view_kind):
    """Verify protocol and prepared-view owners outlive asynchronous copies.

    The test performs the following sequence:

    1. Warm up DeviceCopy, synchronize, and drain completed owner retentions.
    2. Launch a finite delay kernel on a blocker stream.
    3. Record an event after the delay kernel on the blocker stream.
    4. Make the copy stream wait for that event.
    5. Submit DeviceCopy work to the copy stream.
    6. Verify that the blocker event is incomplete, so the copy is pending.
    7. Retain weak references, drop the ordinary references, and collect garbage.
    8. Verify that DeviceCopy still retains the source and destination owners.
    9. Synchronize the blocker and copy streams.
    10. Directly drain completed owner retentions and collect garbage again.
    11. Verify that the source and destination owners have been destroyed.

    ``_drain_device_copy_owner_releases()`` directly invokes the C++ drain routine,
    which atomically detaches the list of owner-retention records whose stream-ordered
    CUDA host callbacks have completed, decrements their source and destination
    references on this Python thread, and frees the records. It cannot release
    owners whose CUDA callbacks have not run. A drain callback already scheduled
    through ``Py_AddPendingCall()`` may run later, find the list empty, and do
    nothing.

    The finite delay kernel terminates without action by this test thread. A
    blocking CUDA host callback would instead create a cycle if it waited for
    this thread while DeviceCopy was blocked performing subsequent CUDA API
    work.

    The warm-up call is made before starting the delay kernel because HostJIT
    compilation and lazy loading can outlast the delay. Without the warm-up,
    compilation could consume the pending-work window and make the lifetime
    assertions race with completion rather than test asynchronous retention.
    """
    source = cp.arange(256, dtype=cp.int32)
    destination = cp.empty_like(source)
    if view_kind == "prepared":
        source_arg = impl._as_device_array_view(source)
        destination_arg = impl._as_device_array_view(destination)
    else:
        source_arg = _NoDLPackArray(source)
        destination_arg = _NoDLPackArray(destination)
    device_copy = impl._make_device_copy(source_arg, destination_arg)
    device_copy(source_arg, destination_arg)
    cp.cuda.get_current_stream().synchronize()
    impl._drain_device_copy_owner_releases()

    blocker_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream = cp.cuda.Stream(non_blocking=True)
    stream_protocol = _StreamProtocol(copy_stream)
    blocker_finished = cp.cuda.Event()
    delay_cycles = np.uint64(cp.cuda.Device().attributes["ClockRate"] * 5500)

    try:
        # launch a delay kernel on blocker_stream
        _delay_kernel((1,), (1,), (delay_cycles,), stream=blocker_stream)
        # order copy_stream after blocker_stream
        blocker_finished.record(blocker_stream)
        copy_stream.wait_event(blocker_finished)

        # submit the copy to copy_stream and verify that it is pending
        device_copy(source_arg, destination_arg, stream=stream_protocol)
        assert not blocker_finished.done
        source_ref = weakref.ref(source)
        destination_ref = weakref.ref(destination)
        del source
        del destination
        del source_arg
        del destination_arg
        gc.collect()

        assert not copy_stream.done, "copy completed before the ownership check"
        # verify that DeviceCopy still retains the source and destination owners
        assert source_ref() is not None
        assert destination_ref() is not None

        blocker_stream.synchronize()
        copy_stream.synchronize()
        # ensure release of owners whose callbacks have completed
        impl._drain_device_copy_owner_releases()
        gc.collect()

        assert source_ref() is None
        assert destination_ref() is None
    finally:
        blocker_stream.synchronize()
        copy_stream.synchronize()
        impl._drain_device_copy_owner_releases()
        device_copy.close()
