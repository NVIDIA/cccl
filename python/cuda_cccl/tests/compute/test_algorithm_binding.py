# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public-API tests that binding preserves an algorithm's computation."""

import concurrent.futures
import gc
import threading

import numpy as np
import pytest
from _utils.device_array import DeviceArray
from cuda.core import Device, Program, ProgramOptions

import cuda.compute as compute
import cuda.compute._cccl_interop as cccl_interop
from cuda.compute import OpKind
from cuda.compute._caching import _PerCCBuildResults
from cuda.compute.op import RawOp

# These tests bind deserialized algorithms; serialization is unsupported on v2.
# Setting this special pytest variable result in this mark being applied to every
# test in this file
pytestmark = pytest.mark.serialization


"""Test "Binding preserves the computation." invariant"""


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("num_items", [0, 1, 257])
def test_bound_merge_sort_preserves_computation(dtype, num_items, cuda_stream):
    rng = np.random.default_rng(42)
    # Keep allocated buffers even for an empty logical range (num_items=0).
    h_in = rng.integers(-10, 11, size=max(1, num_items)).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)

    algorithm = compute.make_merge_sort(
        d_in_keys=d_in, d_out_keys=d_out, op=OpKind.LESS
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)

    arguments = dict(
        d_in_keys=d_in,
        d_in_values=None,
        d_out_keys=d_out,
        d_out_values=None,
        num_items=num_items,
        op=OpKind.LESS,
        stream=cuda_stream,
    )
    workspace_size = executable(temp_storage=None, **arguments)
    workspace = DeviceArray.empty(workspace_size, np.uint8, stream=cuda_stream)
    executable(temp_storage=workspace, **arguments)

    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream)[:num_items], np.sort(h_in[:num_items])
    )


@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_bound_merge_sort_queries_and_executes_without_device_lookup(
    monkeypatch, roundtrip
):
    h = np.asarray([4, 1, 3, 2], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h)
    d_out = DeviceArray.empty(h.shape, h.dtype)
    portable = compute.make_merge_sort(d_in_keys=d_in, d_out_keys=d_out, op=OpKind.LESS)
    if roundtrip:
        portable = compute.deserialize(compute.serialize(portable))
        assert portable._bound_build_result is None
    original_binding = portable._bound_build_result
    bound = portable.bind(device_id=Device().device_id)
    assert portable._bound_build_result is original_binding

    # Serialization omits the binding, including for an already bound object.
    restored = compute.deserialize(compute.serialize(bound))
    assert restored._bound_build_result is None

    def unexpected_lookup(*args):
        raise AssertionError("a bound call must not query or resolve a device")

    monkeypatch.setattr(cccl_interop, "current_device_id", unexpected_lookup)
    monkeypatch.setattr(cccl_interop, "current_device_info", unexpected_lookup)
    monkeypatch.setattr(_PerCCBuildResults, "resolve", unexpected_lookup)
    args = dict(
        d_in_keys=d_in,
        d_in_values=None,
        d_out_keys=d_out,
        d_out_values=None,
        num_items=h.size,
        op=OpKind.LESS,
    )
    size = bound(temp_storage=None, **args)
    assert bound(temp_storage=None, **args) == size
    workspace = DeviceArray.empty(size, np.uint8)
    bound(temp_storage=workspace, **args)
    Device().sync()
    np.testing.assert_array_equal(d_out.copy_to_host(), np.sort(h))


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("num_items", [0, 1, 257])
def test_bound_reduce_preserves_computation(dtype, num_items, cuda_stream):
    rng = np.random.default_rng(42)
    h_in = rng.integers(-10, 11, size=num_items).astype(dtype)
    h_init = np.asarray([37], dtype=dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)

    algorithm = compute.make_reduce_into(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)

    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=OpKind.PLUS,
        h_init=h_init,
        stream=cuda_stream,
    )
    workspace_size = executable(temp_storage=None, **arguments)
    workspace = DeviceArray.empty(workspace_size, np.uint8, stream=cuda_stream)
    executable(temp_storage=workspace, **arguments)

    expected = np.add.reduce(h_in, dtype=dtype, initial=h_init[0])
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream), np.asarray([expected], dtype=dtype)
    )


def _make_is_positive_op(dtype, device):
    value_type = {np.int32: "int", np.float64: "double"}[dtype]
    name = f"is_positive_{np.dtype(dtype).name}"
    source = f"""
extern "C" __device__ void {name}(void* input, void* result) {{
    *static_cast<unsigned char*>(result) =
        *static_cast<{value_type}*>(input) > 0;
}}
"""
    major, minor = device.compute_capability
    options = ProgramOptions(
        arch=f"sm_{major}{minor}",
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    ltoir = Program(source, "c++", options=options).compile("ltoir").code
    return RawOp(ltoir=ltoir, name=name)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("num_items", [0, 1, 257])
def test_bound_select_preserves_computation(dtype, num_items, cuda_stream):
    positive_op = _make_is_positive_op(dtype, cuda_stream.device)
    rng = np.random.default_rng(42)
    h_in = rng.integers(-10, 11, size=num_items).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)
    d_num_selected = DeviceArray.empty(2, np.uint64, stream=cuda_stream)

    algorithm = compute.make_select(
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected,
        cond=positive_op,
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)

    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_num_selected,
        cond=positive_op,
        num_items=num_items,
        stream=cuda_stream,
    )
    workspace_size = executable(temp_storage=None, **arguments)
    workspace = DeviceArray.empty(workspace_size, np.uint8, stream=cuda_stream)
    executable(temp_storage=workspace, **arguments)

    expected = h_in[h_in > 0]
    num_selected = int(d_num_selected.copy_to_host(stream=cuda_stream)[0])
    assert num_selected == expected.size
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream)[:num_selected], expected
    )


"""Test "Binding leaves the original usable." invariant"""


def _execute_with_workspace(algorithm, arguments, stream):
    workspace_size = algorithm(temp_storage=None, **arguments)
    workspace = DeviceArray.empty(workspace_size, np.uint8, stream=stream)
    algorithm(temp_storage=workspace, **arguments)
    # Finish execution before releasing this call's workspace.
    stream.sync()


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_merge_sort_original_remains_usable(dtype, roundtrip, cuda_stream):
    rng = np.random.default_rng(42)
    h_original = rng.integers(-50, 51, size=257).astype(dtype)
    h_bound = rng.integers(-100, 101, size=129).astype(dtype)
    d_original_in = DeviceArray.from_numpy(h_original, stream=cuda_stream)
    d_original_out = DeviceArray.empty(h_original.shape, dtype, stream=cuda_stream)
    d_bound_in = DeviceArray.from_numpy(h_bound, stream=cuda_stream)
    d_bound_out = DeviceArray.empty(h_bound.shape, dtype, stream=cuda_stream)

    original = compute.make_merge_sort(
        d_in_keys=d_original_in, d_out_keys=d_original_out, op=OpKind.LESS
    )
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    original_arguments = dict(
        d_in_keys=d_original_in,
        d_in_values=None,
        d_out_keys=d_original_out,
        d_out_values=None,
        num_items=h_original.size,
        op=OpKind.LESS,
        stream=cuda_stream,
    )
    bound_arguments = dict(
        d_in_keys=d_bound_in,
        d_in_values=None,
        d_out_keys=d_bound_out,
        d_out_values=None,
        num_items=h_bound.size,
        op=OpKind.LESS,
        stream=cuda_stream,
    )

    expected_original = np.sort(h_original)
    _execute_with_workspace(original, original_arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream), expected_original
    )

    executable = original.bind(device_id=cuda_stream.device.device_id)
    _execute_with_workspace(executable, bound_arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_bound_out.copy_to_host(stream=cuda_stream), np.sort(h_bound)
    )

    # Reinitialize the output so a skipped execution cannot reuse the old result.
    d_original_out.copy_from_host(np.full_like(h_original, 999), stream=cuda_stream)
    _execute_with_workspace(original, original_arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream), expected_original
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_reduce_original_remains_usable(dtype, roundtrip, cuda_stream):
    rng = np.random.default_rng(42)
    h_original = rng.integers(-50, 51, size=257).astype(dtype)
    h_bound = rng.integers(-100, 101, size=129).astype(dtype)
    h_original_init = np.asarray([37], dtype=dtype)
    h_bound_init = np.asarray([-11], dtype=dtype)
    d_original_in = DeviceArray.from_numpy(h_original, stream=cuda_stream)
    d_original_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    d_bound_in = DeviceArray.from_numpy(h_bound, stream=cuda_stream)
    d_bound_out = DeviceArray.empty(1, dtype, stream=cuda_stream)

    original = compute.make_reduce_into(
        d_in=d_original_in,
        d_out=d_original_out,
        op=OpKind.PLUS,
        h_init=h_original_init,
    )
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    original_arguments = dict(
        d_in=d_original_in,
        d_out=d_original_out,
        num_items=h_original.size,
        op=OpKind.PLUS,
        h_init=h_original_init,
        stream=cuda_stream,
    )
    bound_arguments = dict(
        d_in=d_bound_in,
        d_out=d_bound_out,
        num_items=h_bound.size,
        op=OpKind.PLUS,
        h_init=h_bound_init,
        stream=cuda_stream,
    )

    expected_original = np.asarray(
        [np.add.reduce(h_original, dtype=dtype, initial=h_original_init[0])],
        dtype=dtype,
    )
    _execute_with_workspace(original, original_arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream), expected_original
    )

    executable = original.bind(device_id=cuda_stream.device.device_id)
    _execute_with_workspace(executable, bound_arguments, cuda_stream)
    expected_bound = np.asarray(
        [np.add.reduce(h_bound, dtype=dtype, initial=h_bound_init[0])], dtype=dtype
    )
    np.testing.assert_array_equal(
        d_bound_out.copy_to_host(stream=cuda_stream), expected_bound
    )

    d_original_out.copy_from_host(expected_original + 1, stream=cuda_stream)
    _execute_with_workspace(original, original_arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream), expected_original
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_select_original_remains_usable(dtype, roundtrip, cuda_stream):
    positive_op = _make_is_positive_op(dtype, cuda_stream.device)
    rng = np.random.default_rng(42)
    h_original = rng.integers(-50, 51, size=257).astype(dtype)
    h_bound = rng.integers(-100, 101, size=129).astype(dtype)
    d_original_in = DeviceArray.from_numpy(h_original, stream=cuda_stream)
    d_original_out = DeviceArray.empty(h_original.shape, dtype, stream=cuda_stream)
    d_original_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    d_bound_in = DeviceArray.from_numpy(h_bound, stream=cuda_stream)
    d_bound_out = DeviceArray.empty(h_bound.shape, dtype, stream=cuda_stream)
    d_bound_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)

    original = compute.make_select(
        d_in=d_original_in,
        d_out=d_original_out,
        d_num_selected_out=d_original_count,
        cond=positive_op,
    )
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    original_arguments = dict(
        d_in=d_original_in,
        d_out=d_original_out,
        d_num_selected_out=d_original_count,
        cond=positive_op,
        num_items=h_original.size,
        stream=cuda_stream,
    )
    bound_arguments = dict(
        d_in=d_bound_in,
        d_out=d_bound_out,
        d_num_selected_out=d_bound_count,
        cond=positive_op,
        num_items=h_bound.size,
        stream=cuda_stream,
    )

    expected_original = h_original[h_original > 0]
    _execute_with_workspace(original, original_arguments, cuda_stream)
    original_count = int(d_original_count.copy_to_host(stream=cuda_stream)[0])
    assert original_count == expected_original.size
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream)[:original_count],
        expected_original,
    )

    executable = original.bind(device_id=cuda_stream.device.device_id)
    _execute_with_workspace(executable, bound_arguments, cuda_stream)
    expected_bound = h_bound[h_bound > 0]
    bound_count = int(d_bound_count.copy_to_host(stream=cuda_stream)[0])
    assert bound_count == expected_bound.size
    np.testing.assert_array_equal(
        d_bound_out.copy_to_host(stream=cuda_stream)[:bound_count], expected_bound
    )

    d_original_out.copy_from_host(np.full_like(h_original, 999), stream=cuda_stream)
    d_original_count.copy_from_host(np.zeros(2, dtype=np.uint64), stream=cuda_stream)
    _execute_with_workspace(original, original_arguments, cuda_stream)
    original_count = int(d_original_count.copy_to_host(stream=cuda_stream)[0])
    assert original_count == expected_original.size
    np.testing.assert_array_equal(
        d_original_out.copy_to_host(stream=cuda_stream)[:original_count],
        expected_original,
    )


"""Test "Independent executables." invariant"""


def _run_independent_executables(original, device_id, run):
    executables = [original.bind(device_id=device_id) for _ in range(2)]
    assert executables[0] is not executables[1]
    assert all(executable is not original for executable in executables)
    barrier = threading.Barrier(2, timeout=60)

    def worker(worker_id):
        try:
            device = Device(device_id)
            device.set_current()
            stream = device.create_stream()
            try:
                run(executables[worker_id], worker_id, stream, barrier)
            finally:
                stream.close()
        except BaseException:
            # Release the peer if setup, execution, or validation fails.
            barrier.abort()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(2)]
        errors = [future.exception() for future in futures]
    # Prefer the original failure over the peer's consequent broken barrier.
    for error in errors:
        if error is not None and not isinstance(error, threading.BrokenBarrierError):
            raise error
    for error in errors:
        if error is not None:
            raise error


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_merge_sort_independent_executables(dtype, roundtrip, cuda_stream):
    d_in = DeviceArray.empty(1, dtype, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    original = compute.make_merge_sort(d_in_keys=d_in, d_out_keys=d_out, op=OpKind.LESS)
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    def run(executable, worker_id, stream, barrier):
        rng = np.random.default_rng(42 + worker_id)
        num_items = 257 + 256 * worker_id
        d_in = DeviceArray.empty(num_items, dtype, stream=stream)
        d_out = DeviceArray.empty(num_items, dtype, stream=stream)
        arguments = dict(
            d_in_keys=d_in,
            d_in_values=None,
            d_out_keys=d_out,
            d_out_values=None,
            num_items=num_items,
            op=OpKind.LESS,
            stream=stream,
        )
        workspace_size = executable(temp_storage=None, **arguments)
        workspace = DeviceArray.empty(workspace_size, np.uint8, stream=stream)
        for _ in range(16):
            h_in = rng.integers(-100, 101, size=num_items).astype(dtype)
            d_in.copy_from_host(h_in, stream=stream)
            d_out.copy_from_host(np.full_like(h_in, 999), stream=stream)
            barrier.wait()
            executable(temp_storage=workspace, **arguments)
            np.testing.assert_array_equal(
                d_out.copy_to_host(stream=stream), np.sort(h_in)
            )

    _run_independent_executables(original, cuda_stream.device.device_id, run)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_reduce_independent_executables(dtype, roundtrip, cuda_stream):
    d_in = DeviceArray.empty(1, dtype, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    original = compute.make_reduce_into(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=np.zeros(1, dtype=dtype)
    )
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    def run(executable, worker_id, stream, barrier):
        rng = np.random.default_rng(42 + worker_id)
        num_items = 257 + 256 * worker_id
        d_in = DeviceArray.empty(num_items, dtype, stream=stream)
        d_out = DeviceArray.empty(1, dtype, stream=stream)
        h_init = np.zeros(1, dtype=dtype)
        arguments = dict(
            d_in=d_in,
            d_out=d_out,
            num_items=num_items,
            op=OpKind.PLUS,
            h_init=h_init,
            stream=stream,
        )
        workspace_size = executable(temp_storage=None, **arguments)
        workspace = DeviceArray.empty(workspace_size, np.uint8, stream=stream)
        for iteration in range(16):
            h_in = rng.integers(-100, 101, size=num_items).astype(dtype)
            h_init[0] = 1000 * worker_id + iteration
            expected = np.asarray(
                [np.add.reduce(h_in, dtype=dtype, initial=h_init[0])], dtype=dtype
            )
            d_in.copy_from_host(h_in, stream=stream)
            d_out.copy_from_host(expected + 1, stream=stream)
            barrier.wait()
            executable(temp_storage=workspace, **arguments)
            np.testing.assert_array_equal(d_out.copy_to_host(stream=stream), expected)

    _run_independent_executables(original, cuda_stream.device.device_id, run)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize("roundtrip", [False, True], ids=["factory", "deserialized"])
def test_select_independent_executables(dtype, roundtrip, cuda_stream):
    value_type = {np.int32: "int", np.float64: "double"}[dtype]
    name = f"greater_than_threshold_{np.dtype(dtype).name}"
    source = f"""
extern "C" __device__ void {name}(void* state, void* input, void* result) {{
    *static_cast<unsigned char*>(result) =
        *static_cast<{value_type}*>(input) > *static_cast<{value_type}*>(state);
}}
"""
    major, minor = cuda_stream.device.compute_capability
    options = ProgramOptions(
        arch=f"sm_{major}{minor}",
        relocatable_device_code=True,
        link_time_optimization=True,
    )
    ltoir = Program(source, "c++", options=options).compile("ltoir").code

    def predicate(threshold):
        return RawOp(
            ltoir=ltoir,
            name=name,
            state=np.asarray([threshold], dtype=dtype).tobytes(),
            state_alignment=np.dtype(dtype).alignment,
        )

    d_in = DeviceArray.empty(1, dtype, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    d_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    original = compute.make_select(
        d_in=d_in, d_out=d_out, d_num_selected_out=d_count, cond=predicate(0)
    )
    if roundtrip:
        original = compute.deserialize(compute.serialize(original))

    def run(executable, worker_id, stream, barrier):
        rng = np.random.default_rng(42 + worker_id)
        num_items = 257 + 256 * worker_id
        d_in = DeviceArray.empty(num_items, dtype, stream=stream)
        d_out = DeviceArray.empty(num_items, dtype, stream=stream)
        d_count = DeviceArray.empty(2, np.uint64, stream=stream)
        arguments = dict(
            d_in=d_in,
            d_out=d_out,
            d_num_selected_out=d_count,
            cond=predicate(0),
            num_items=num_items,
            stream=stream,
        )
        workspace_size = executable(temp_storage=None, **arguments)
        workspace = DeviceArray.empty(workspace_size, np.uint8, stream=stream)
        for iteration in range(16):
            h_in = rng.integers(-100, 101, size=num_items).astype(dtype)
            threshold = 40 * worker_id - 20 + iteration
            arguments["cond"] = predicate(threshold)
            d_in.copy_from_host(h_in, stream=stream)
            d_out.copy_from_host(np.full_like(h_in, 999), stream=stream)
            d_count.copy_from_host(np.zeros(2, dtype=np.uint64), stream=stream)
            barrier.wait()
            executable(temp_storage=workspace, **arguments)
            count = int(d_count.copy_to_host(stream=stream)[0])
            expected = h_in[h_in > threshold]
            assert count == expected.size
            np.testing.assert_array_equal(
                d_out.copy_to_host(stream=stream)[:count], expected
            )

    _run_independent_executables(original, cuda_stream.device.device_id, run)


"""Test "Serialization round-trip." invariant"""


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize(
    "execute_before_serialization", [False, True], ids=["unexecuted", "executed"]
)
def test_bound_merge_sort_serialization_roundtrip(
    dtype, execute_before_serialization, cuda_stream
):
    rng = np.random.default_rng(42)
    h_before = rng.integers(-50, 51, size=257).astype(dtype)
    d_before_in = DeviceArray.from_numpy(h_before, stream=cuda_stream)
    d_before_out = DeviceArray.empty(h_before.shape, dtype, stream=cuda_stream)
    algorithm = compute.make_merge_sort(
        d_in_keys=d_before_in, d_out_keys=d_before_out, op=OpKind.LESS
    )
    bound = algorithm.bind(device_id=cuda_stream.device.device_id)
    if execute_before_serialization:
        arguments = dict(
            d_in_keys=d_before_in,
            d_in_values=None,
            d_out_keys=d_before_out,
            d_out_values=None,
            num_items=h_before.size,
            op=OpKind.LESS,
            stream=cuda_stream,
        )
        _execute_with_workspace(bound, arguments, cuda_stream)
        np.testing.assert_array_equal(
            d_before_out.copy_to_host(stream=cuda_stream), np.sort(h_before)
        )

    blob = compute.serialize(bound)
    portable = compute.deserialize(blob)
    rebound = portable.bind(device_id=cuda_stream.device.device_id)

    # Use new buffers and a different input size after reconstruction.
    h_after = rng.integers(-100, 101, size=129).astype(dtype)
    d_after_in = DeviceArray.from_numpy(h_after, stream=cuda_stream)
    d_after_out = DeviceArray.empty(h_after.shape, dtype, stream=cuda_stream)
    arguments = dict(
        d_in_keys=d_after_in,
        d_in_values=None,
        d_out_keys=d_after_out,
        d_out_values=None,
        num_items=h_after.size,
        op=OpKind.LESS,
        stream=cuda_stream,
    )
    _execute_with_workspace(rebound, arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_after_out.copy_to_host(stream=cuda_stream), np.sort(h_after)
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize(
    "execute_before_serialization", [False, True], ids=["unexecuted", "executed"]
)
def test_bound_reduce_serialization_roundtrip(
    dtype, execute_before_serialization, cuda_stream
):
    rng = np.random.default_rng(42)
    h_before = rng.integers(-50, 51, size=257).astype(dtype)
    h_before_init = np.asarray([37], dtype=dtype)
    d_before_in = DeviceArray.from_numpy(h_before, stream=cuda_stream)
    d_before_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    algorithm = compute.make_reduce_into(
        d_in=d_before_in, d_out=d_before_out, op=OpKind.PLUS, h_init=h_before_init
    )
    bound = algorithm.bind(device_id=cuda_stream.device.device_id)
    if execute_before_serialization:
        arguments = dict(
            d_in=d_before_in,
            d_out=d_before_out,
            num_items=h_before.size,
            op=OpKind.PLUS,
            h_init=h_before_init,
            stream=cuda_stream,
        )
        _execute_with_workspace(bound, arguments, cuda_stream)
        expected_before = np.asarray(
            [np.add.reduce(h_before, dtype=dtype, initial=h_before_init[0])],
            dtype=dtype,
        )
        np.testing.assert_array_equal(
            d_before_out.copy_to_host(stream=cuda_stream), expected_before
        )

    blob = compute.serialize(bound)
    portable = compute.deserialize(blob)
    rebound = portable.bind(device_id=cuda_stream.device.device_id)

    h_after = rng.integers(-100, 101, size=129).astype(dtype)
    h_after_init = np.asarray([-11], dtype=dtype)
    d_after_in = DeviceArray.from_numpy(h_after, stream=cuda_stream)
    d_after_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    arguments = dict(
        d_in=d_after_in,
        d_out=d_after_out,
        num_items=h_after.size,
        op=OpKind.PLUS,
        h_init=h_after_init,
        stream=cuda_stream,
    )
    _execute_with_workspace(rebound, arguments, cuda_stream)
    expected_after = np.asarray(
        [np.add.reduce(h_after, dtype=dtype, initial=h_after_init[0])], dtype=dtype
    )
    np.testing.assert_array_equal(
        d_after_out.copy_to_host(stream=cuda_stream), expected_after
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
@pytest.mark.parametrize(
    "execute_before_serialization", [False, True], ids=["unexecuted", "executed"]
)
def test_bound_select_serialization_roundtrip(
    dtype, execute_before_serialization, cuda_stream
):
    positive_op = _make_is_positive_op(dtype, cuda_stream.device)
    rng = np.random.default_rng(42)
    h_before = rng.integers(-50, 51, size=257).astype(dtype)
    d_before_in = DeviceArray.from_numpy(h_before, stream=cuda_stream)
    d_before_out = DeviceArray.empty(h_before.shape, dtype, stream=cuda_stream)
    d_before_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    algorithm = compute.make_select(
        d_in=d_before_in,
        d_out=d_before_out,
        d_num_selected_out=d_before_count,
        cond=positive_op,
    )
    bound = algorithm.bind(device_id=cuda_stream.device.device_id)
    if execute_before_serialization:
        arguments = dict(
            d_in=d_before_in,
            d_out=d_before_out,
            d_num_selected_out=d_before_count,
            cond=positive_op,
            num_items=h_before.size,
            stream=cuda_stream,
        )
        _execute_with_workspace(bound, arguments, cuda_stream)
        expected_before = h_before[h_before > 0]
        count_before = int(d_before_count.copy_to_host(stream=cuda_stream)[0])
        assert count_before == expected_before.size
        np.testing.assert_array_equal(
            d_before_out.copy_to_host(stream=cuda_stream)[:count_before],
            expected_before,
        )

    blob = compute.serialize(bound)
    portable = compute.deserialize(blob)
    rebound = portable.bind(device_id=cuda_stream.device.device_id)

    h_after = rng.integers(-100, 101, size=129).astype(dtype)
    d_after_in = DeviceArray.from_numpy(h_after, stream=cuda_stream)
    d_after_out = DeviceArray.empty(h_after.shape, dtype, stream=cuda_stream)
    d_after_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    arguments = dict(
        d_in=d_after_in,
        d_out=d_after_out,
        d_num_selected_out=d_after_count,
        cond=positive_op,
        num_items=h_after.size,
        stream=cuda_stream,
    )
    _execute_with_workspace(rebound, arguments, cuda_stream)
    expected_after = h_after[h_after > 0]
    count_after = int(d_after_count.copy_to_host(stream=cuda_stream)[0])
    assert count_after == expected_after.size
    np.testing.assert_array_equal(
        d_after_out.copy_to_host(stream=cuda_stream)[:count_after], expected_after
    )


"""Test "Bound executables outlive the portable object." invariant"""


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_merge_sort_outlives_portable(dtype, cuda_stream):
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)
    algorithm = compute.make_merge_sort(
        d_in_keys=d_in, d_out_keys=d_out, op=OpKind.LESS
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    del portable
    gc.collect()

    arguments = dict(
        d_in_keys=d_in,
        d_in_values=None,
        d_out_keys=d_out,
        d_out_values=None,
        num_items=h_in.size,
        op=OpKind.LESS,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    np.testing.assert_array_equal(d_out.copy_to_host(stream=cuda_stream), np.sort(h_in))


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_reduce_outlives_portable(dtype, cuda_stream):
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    h_init = np.asarray([37], dtype=dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    algorithm = compute.make_reduce_into(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    del portable
    gc.collect()

    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        num_items=h_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    expected = np.asarray(
        [np.add.reduce(h_in, dtype=dtype, initial=h_init[0])], dtype=dtype
    )
    np.testing.assert_array_equal(d_out.copy_to_host(stream=cuda_stream), expected)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_select_outlives_portable(dtype, cuda_stream):
    positive_op = _make_is_positive_op(dtype, cuda_stream.device)
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)
    d_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    algorithm = compute.make_select(
        d_in=d_in, d_out=d_out, d_num_selected_out=d_count, cond=positive_op
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    del portable
    gc.collect()

    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_count,
        cond=positive_op,
        num_items=h_in.size,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    expected = h_in[h_in > 0]
    count = int(d_count.copy_to_host(stream=cuda_stream)[0])
    assert count == expected.size
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream)[:count], expected
    )


"""Test "Repeated binding preserves computation." invariant"""


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_merge_sort_can_bind_again(dtype, cuda_stream):
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)
    algorithm = compute.make_merge_sort(
        d_in_keys=d_in, d_out_keys=d_out, op=OpKind.LESS
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    arguments = dict(
        d_in_keys=d_in,
        d_in_values=None,
        d_out_keys=d_out,
        d_out_values=None,
        num_items=h_in.size,
        op=OpKind.LESS,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    np.testing.assert_array_equal(d_out.copy_to_host(stream=cuda_stream), np.sort(h_in))

    rebound = executable.bind(device_id=cuda_stream.device.device_id)
    h_after = rng.integers(-100, 101, size=h_in.size).astype(dtype)
    d_in.copy_from_host(h_after, stream=cuda_stream)
    d_out.copy_from_host(np.full_like(h_after, 999), stream=cuda_stream)
    _execute_with_workspace(rebound, arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream), np.sort(h_after)
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_reduce_can_bind_again(dtype, cuda_stream):
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    h_init = np.asarray([37], dtype=dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(1, dtype, stream=cuda_stream)
    algorithm = compute.make_reduce_into(
        d_in=d_in, d_out=d_out, op=OpKind.PLUS, h_init=h_init
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        num_items=h_in.size,
        op=OpKind.PLUS,
        h_init=h_init,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    expected = np.asarray(
        [np.add.reduce(h_in, dtype=dtype, initial=h_init[0])], dtype=dtype
    )
    np.testing.assert_array_equal(d_out.copy_to_host(stream=cuda_stream), expected)

    rebound = executable.bind(device_id=cuda_stream.device.device_id)
    h_after = rng.integers(-100, 101, size=h_in.size).astype(dtype)
    h_init[0] = -11
    expected_after = np.asarray(
        [np.add.reduce(h_after, dtype=dtype, initial=h_init[0])], dtype=dtype
    )
    d_in.copy_from_host(h_after, stream=cuda_stream)
    d_out.copy_from_host(expected_after + 1, stream=cuda_stream)
    _execute_with_workspace(rebound, arguments, cuda_stream)
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream), expected_after
    )


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_bound_select_can_bind_again(dtype, cuda_stream):
    positive_op = _make_is_positive_op(dtype, cuda_stream.device)
    rng = np.random.default_rng(42)
    h_in = rng.integers(-50, 51, size=257).astype(dtype)
    d_in = DeviceArray.from_numpy(h_in, stream=cuda_stream)
    d_out = DeviceArray.empty(h_in.shape, dtype, stream=cuda_stream)
    d_count = DeviceArray.empty(2, np.uint64, stream=cuda_stream)
    algorithm = compute.make_select(
        d_in=d_in, d_out=d_out, d_num_selected_out=d_count, cond=positive_op
    )
    portable = compute.deserialize(compute.serialize(algorithm))
    executable = portable.bind(device_id=cuda_stream.device.device_id)
    arguments = dict(
        d_in=d_in,
        d_out=d_out,
        d_num_selected_out=d_count,
        cond=positive_op,
        num_items=h_in.size,
        stream=cuda_stream,
    )
    _execute_with_workspace(executable, arguments, cuda_stream)
    expected = h_in[h_in > 0]
    count = int(d_count.copy_to_host(stream=cuda_stream)[0])
    assert count == expected.size
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream)[:count], expected
    )

    rebound = executable.bind(device_id=cuda_stream.device.device_id)
    h_after = rng.integers(-100, 101, size=h_in.size).astype(dtype)
    d_in.copy_from_host(h_after, stream=cuda_stream)
    d_out.copy_from_host(np.full_like(h_after, 999), stream=cuda_stream)
    d_count.copy_from_host(np.zeros(2, dtype=np.uint64), stream=cuda_stream)
    _execute_with_workspace(rebound, arguments, cuda_stream)
    expected_after = h_after[h_after > 0]
    count_after = int(d_count.copy_to_host(stream=cuda_stream)[0])
    assert count_after == expected_after.size
    np.testing.assert_array_equal(
        d_out.copy_to_host(stream=cuda_stream)[:count_after], expected_after
    )
