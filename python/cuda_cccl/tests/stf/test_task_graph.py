# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda  # noqa: E402

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.numba import numba_arguments  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_low_occupancy_warnings(monkeypatch):
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)


@cuda.jit
def add_kernel(x, val):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = x[i] + val


def _record_add_graph(n=256, value=1.0):
    x_host = np.zeros(n, dtype=np.float32)
    graph = stf.task_graph()
    ctx = graph.context
    lx = ctx.logical_data(x_host, name="X")

    tpb = 128
    bpg = (n + tpb - 1) // tpb

    with graph:
        with ctx.task(lx.rw()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dx = numba_arguments(t)
            add_kernel[bpg, tpb, nb_stream](dx, value)

    return graph, x_host


def test_task_graph_relaunch():
    graph, x_host = _record_add_graph(value=1.0)

    for _ in range(5):
        graph.launch()

    graph.finalize()
    assert np.allclose(x_host, 5.0), f"Expected 5.0, got {x_host[0]}"


def test_task_graph_accessors_after_recording():
    graph, _ = _record_add_graph()

    assert graph.raw.valid
    assert graph.graph != 0
    assert graph.exec_graph != 0
    assert graph.stream != 0

    graph.finalize()


def test_task_graph_context_data_declarations_outside_recording():
    graph = stf.task_graph()
    ctx = graph.context

    x_host = np.zeros(16, dtype=np.float32)
    lx = ctx.logical_data(x_host, name="X")
    token = ctx.token()

    assert lx is not None
    assert token is not None

    graph.finalize()


def test_task_graph_reset_then_finalize():
    graph, _ = _record_add_graph()

    graph.reset()
    graph.reset()
    graph.finalize()
    graph.finalize()


def test_task_graph_launch_before_recording_raises():
    graph = stf.task_graph()
    try:
        with pytest.raises(RuntimeError):
            graph.launch()
    finally:
        graph.finalize()


def test_task_graph_task_outside_recording_raises():
    graph = stf.task_graph()
    ctx = graph.context
    x_host = np.zeros(16, dtype=np.float32)
    lx = ctx.logical_data(x_host, name="X")

    try:
        with pytest.raises(RuntimeError):
            ctx.task(lx.rw())
    finally:
        graph.finalize()


def test_task_graph_nested_enter_raises():
    graph = stf.task_graph()
    try:
        with graph:
            with pytest.raises(RuntimeError):
                with graph:
                    pass
    finally:
        graph.finalize()


def test_task_graph_second_recording_raises():
    graph, _ = _record_add_graph()
    try:
        with pytest.raises(RuntimeError):
            with graph:
                pass
    finally:
        graph.finalize()


def test_task_graph_enter_after_reset_raises():
    graph, _ = _record_add_graph()
    graph.reset()

    try:
        with pytest.raises(RuntimeError):
            with graph:
                pass
    finally:
        graph.finalize()


def test_task_graph_enter_after_finalize_raises():
    graph = stf.task_graph()
    graph.finalize()

    with pytest.raises(RuntimeError):
        with graph:
            pass


def test_task_graph_failed_recording_locks_graph():
    graph = stf.task_graph()

    with pytest.raises(ValueError):
        with graph:
            raise ValueError("record failed")

    with pytest.raises(RuntimeError):
        graph.launch()
    with pytest.raises(RuntimeError):
        with graph:
            pass

    graph.finalize()


def test_task_graph_launch_after_reset_raises():
    graph, _ = _record_add_graph()
    graph.reset()

    try:
        with pytest.raises(RuntimeError):
            graph.launch()
    finally:
        graph.finalize()


def test_task_graph_launch_after_finalize_raises():
    graph, _ = _record_add_graph()
    graph.finalize()

    with pytest.raises(RuntimeError):
        graph.launch()


def test_task_graph_accessors_before_recording_raise():
    graph = stf.task_graph()
    try:
        with pytest.raises(RuntimeError):
            _ = graph.raw
        with pytest.raises(RuntimeError):
            _ = graph.graph
        with pytest.raises(RuntimeError):
            _ = graph.exec_graph
        with pytest.raises(RuntimeError):
            _ = graph.stream
    finally:
        graph.finalize()


def test_task_graph_accessors_after_reset_raise():
    graph, _ = _record_add_graph()
    graph.reset()

    try:
        with pytest.raises(RuntimeError):
            _ = graph.raw
        with pytest.raises(RuntimeError):
            _ = graph.graph
        with pytest.raises(RuntimeError):
            _ = graph.exec_graph
        with pytest.raises(RuntimeError):
            _ = graph.stream
    finally:
        graph.finalize()


def test_task_graph_finalize_while_recording_raises():
    graph = stf.task_graph()

    with graph:
        with pytest.raises(RuntimeError):
            graph.finalize()

    graph.finalize()
