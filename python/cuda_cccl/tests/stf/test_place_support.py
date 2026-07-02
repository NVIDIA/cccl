# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.bindings import runtime as cudart  # noqa: E402


def _require_green_context_helper(sm_count=1, dev_id=0):
    if not hasattr(stf, "green_context_helper"):
        pytest.skip("green context STF bindings are not available")
    try:
        return stf.green_context_helper(sm_count, dev_id)
    except Exception as exc:
        pytest.skip(f"green context support unavailable: {exc}")


def test_scope_context_manager():
    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        pass


def test_scope_nested():
    stf.machine_init()
    outer = stf.exec_place.device(0)
    inner = stf.exec_place.device(0)
    with outer:
        with inner:
            pass


def test_pick_stream_standalone():
    """Places work without an STF context: caller owns the registry."""
    stf.machine_init()
    resources = stf.exec_place_resources()
    place = stf.exec_place.device(0)
    with place:
        s = place.pick_stream(resources)
        assert isinstance(s, stf.CudaStream)
        assert isinstance(s, int)
        assert s != 0


def test_pick_stream_borrowed_from_context():
    """STF users borrow the context's registry and share its pools."""
    stf.machine_init()
    place = stf.exec_place.device(0)
    with stf.context() as ctx, place:
        s = place.pick_stream(ctx.place_resources)
        assert isinstance(s, stf.CudaStream)
        assert s != 0


def test_pick_stream_requires_resources():
    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        with pytest.raises(TypeError):
            place.pick_stream(None)


def test_two_resources_handles_isolated():
    """Independent registries hand out independent streams for the same place."""
    stf.machine_init()
    r1 = stf.exec_place_resources()
    r2 = stf.exec_place_resources()
    place = stf.exec_place.device(0)
    with place:
        s1 = place.pick_stream(r1)
        s2 = place.pick_stream(r2)
        assert int(s1) != int(s2)


def test_affine_data_place():
    place = stf.exec_place.device(0)
    dp = place.affine_data_place
    assert dp.device_id == 0


def test_grid_getitem():
    grid = stf.exec_place_grid.from_devices([0, 0])
    sub = grid[0]
    assert sub.kind == "device"


def test_grid_iteration():
    grid = stf.exec_place_grid.from_devices([0, 0])
    for i in range(grid.size):
        sub = grid[i]
        assert sub.kind == "device"
        assert sub.affine_data_place.device_id == 0


def test_getitem_out_of_bounds():
    place = stf.exec_place.device(0)
    with pytest.raises(IndexError):
        place[1]

    grid = stf.exec_place_grid.from_devices([0, 0])
    with pytest.raises(IndexError):
        grid[grid.size]


def test_machine_init_idempotent():
    stf.machine_init()
    stf.machine_init()


def test_green_context_helper_view():
    helper = _require_green_context_helper()
    assert helper.get_count() >= 1
    assert len(helper) == helper.get_count()

    view = helper.get_view(0)
    assert view.helper is helper
    assert view.index == 0
    assert view.device_id == helper.device_id


def test_green_context_exec_and_data_places():
    stf.machine_init()
    helper = _require_green_context_helper()
    view = helper.get_view(0)

    place = stf.exec_place.green_ctx(view)
    assert place.kind == "device"
    assert place.affine_data_place.device_id == helper.device_id

    resources = stf.exec_place_resources()
    with place:
        stream = place.pick_stream(resources)
        assert isinstance(stream, stf.CudaStream)
        assert stream != 0

    green_affine_place = stf.exec_place.green_ctx(view, use_green_ctx_data_place=True)
    assert "green_ctx" in green_affine_place.affine_data_place.kind

    dplace = stf.data_place.green_ctx(view)
    assert dplace.device_id == helper.device_id
    assert "green_ctx" in dplace.kind


def test_scope_with_cuda_compute():
    """Activate place, pick_stream, run cuda.compute.reduce_into -- no STF tasks."""
    try:
        import cuda.compute
        from cuda.compute import OpKind
    except ImportError:
        pytest.skip("cuda.compute not available")

    import numpy as np

    from cuda.stf._experimental._stf_bindings import stf_cai

    stf.machine_init()
    place = stf.exec_place.device(0)
    resources = stf.exec_place_resources()

    with place:
        stream = place.pick_stream(resources)

        n = 1024
        h_input = np.arange(n, dtype=np.float32)

        import numba.cuda

        d_input = numba.cuda.to_device(h_input)
        d_output = numba.cuda.device_array(1, dtype=np.float32)

        input_cai = stf_cai(
            d_input.device_ctypes_pointer.value, (n,), np.float32, stream=stream
        )
        output_cai = stf_cai(
            d_output.device_ctypes_pointer.value, (1,), np.float32, stream=stream
        )

        h_init = np.array([0.0], dtype=np.float32)
        cuda.compute.reduce_into(
            d_in=input_cai,
            d_out=output_cai,
            op=OpKind.PLUS,
            num_items=n,
            h_init=h_init,
            stream=stream,
        )

        numba.cuda.current_context().synchronize()

        result = d_output.copy_to_host()
        expected = h_input.sum()
        assert abs(result[0] - expected) < 1e-2, f"got {result[0]}, expected {expected}"


# ---------------------------------------------------------------------------
# data_place.allocate / deallocate / allocation_is_stream_ordered
# ---------------------------------------------------------------------------


def test_data_place_allocate_deallocate():
    """Allocate on a device data_place, verify non-zero pointer, deallocate."""
    stf.machine_init()
    place = stf.exec_place.device(0)
    resources = stf.exec_place_resources()
    with place:
        dp = place.affine_data_place
        stream = place.pick_stream(resources)
        ptr = dp.allocate(1024, stream)
        assert ptr != 0
        dp.deallocate(ptr, 1024, stream)


def test_data_place_host_allocate():
    """Allocate on the host data_place, write/read, deallocate."""
    import ctypes

    dp = stf.data_place.host()
    ptr = dp.allocate(256)
    assert ptr != 0
    ctypes.memset(ptr, 0x42, 4)
    buf = (ctypes.c_uint8 * 4).from_address(ptr)
    assert buf[0] == 0x42
    dp.deallocate(ptr, 256)


def test_allocation_is_stream_ordered():
    dp_dev = stf.data_place.device(0)
    assert dp_dev.allocation_is_stream_ordered is True

    dp_host = stf.data_place.host()
    assert dp_host.allocation_is_stream_ordered is False

    dp_mgd = stf.data_place.managed()
    assert dp_mgd.allocation_is_stream_ordered is False


# ---------------------------------------------------------------------------
# DeviceArray
# ---------------------------------------------------------------------------


def test_device_array_rejects_negative_size():
    """DeviceArray size must describe a real 1-D allocation."""
    import numpy as np

    with pytest.raises(ValueError, match="non-negative"):
        stf.DeviceArray(-1, np.float32, stf.data_place.host())


def test_device_array_from_host_rejects_non_1d_input():
    """from_host keeps DeviceArray semantics intentionally 1-D."""
    import numpy as np

    with pytest.raises(ValueError, match="1-D"):
        stf.DeviceArray.from_host(
            np.zeros((2, 2), dtype=np.float32), stf.data_place.host()
        )


def test_device_array_roundtrip():
    """Create DeviceArray from host, copy back, verify."""
    import numpy as np

    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        dp = place.affine_data_place
        h = np.arange(128, dtype=np.float32)
        d = stf.DeviceArray.from_host(h, dp)
        assert d.size == 128
        assert d.dtype == np.float32
        result = d.copy_to_host()
        np.testing.assert_array_equal(result, h)


def test_device_array_with_cuda_compute():
    """Use DeviceArray as input to cuda.compute.reduce_into."""
    try:
        import cuda.compute
        from cuda.compute import OpKind
    except ImportError:
        pytest.skip("cuda.compute not available")

    import numpy as np

    stf.machine_init()
    place = stf.exec_place.device(0)
    resources = stf.exec_place_resources()
    with place:
        dp = place.affine_data_place
        stream = place.pick_stream(resources)

        h_in = np.arange(256, dtype=np.float64)
        d_in = stf.DeviceArray.from_host(h_in, dp)
        d_out = stf.DeviceArray(1, np.float64, dp)
        h_init = np.array([0.0], dtype=np.float64)

        cuda.compute.reduce_into(
            d_in=d_in,
            d_out=d_out,
            op=OpKind.PLUS,
            num_items=256,
            h_init=h_init,
            stream=stream,
        )

        (err,) = cudart.cudaStreamSynchronize(cudart.cudaStream_t(int(stream)))
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaStreamSynchronize failed with error code {int(err)}"
            )

        result = d_out.copy_to_host()
        expected = h_in.sum()
        assert abs(result[0] - expected) < 1e-6, f"got {result[0]}, expected {expected}"


def test_device_array_slice_view():
    """Verify slicing returns a view with correct offset and data."""
    import numpy as np

    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        dp = place.affine_data_place
        h = np.arange(100, dtype=np.int32)
        d = stf.DeviceArray.from_host(h, dp)

        view = d[10:20]
        assert view.size == 10
        assert view.dtype == np.int32
        result = view.copy_to_host()
        np.testing.assert_array_equal(result, h[10:20])


def test_device_array_empty():
    """Zero-size DeviceArray should work without errors."""
    import numpy as np

    stf.machine_init()
    place = stf.exec_place.device(0)
    with place:
        dp = place.affine_data_place
        d = stf.DeviceArray(0, np.float32, dp)
        assert d.size == 0
        result = d.copy_to_host()
        assert result.shape == (0,)
