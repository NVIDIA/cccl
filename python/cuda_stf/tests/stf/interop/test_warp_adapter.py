# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for ``cuda.stf._experimental.interop.warp``."""

import functools

import numpy as np
import pytest

wp = pytest.importorskip("warp")

import cuda.stf._experimental as stf
from cuda.stf._experimental.interop.warp import (
    WarpTaskGraph,
    array_arg,
    local_context,
    warp_task,
    wrap_stream,
)


@wp.kernel
def axpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * x[i] + y[i]


@wp.kernel
def scale(a: float, y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = a * y[i]


@pytest.fixture(scope="module", autouse=True)
def _warm():
    wp.init()
    x = wp.zeros(8, dtype=float)
    wp.launch(axpy, dim=8, inputs=[0.0, x, x])
    wp.launch(scale, dim=8, inputs=[1.0, x])
    wp.synchronize()


def test_wrap_stream_is_cached():
    ctx = stf.context()
    with ctx.task(ctx.token().rw()) as t:
        a = wrap_stream(t.stream_ptr())
        b = wrap_stream(t.stream_ptr())
    assert a is b
    ctx.finalize()


def test_warp_task_logical_data_arrays():
    n = 1024
    ctx = stf.context()
    lx = ctx.logical_data(np.full(n, 2.0, dtype=np.float32))
    ly = ctx.logical_data(np.zeros(n, dtype=np.float32))
    with warp_task(ctx, lx.read(), ly.rw()) as (s, (x, y)):
        assert isinstance(x, wp.array) and x.dtype == wp.float32 and x.shape == (n,)
        wp.launch(axpy, dim=n, inputs=[3.0, x, y])  # current stream is the task stream
        assert wp.get_stream() is s
    ctx.wait(ly)
    ctx.finalize()


def test_warp_task_fork_join_record_replay():
    n = 1 << 16
    frames = 4
    x = wp.full(n, 1.0, dtype=float)
    ya, yb, out = (wp.zeros(n, dtype=float) for _ in range(3))
    g = WarpTaskGraph()
    A, B, OUT = (g.tokens[k] for k in ("a", "b", "out"))

    def frame():
        with g.task(A.rw(), symbol="left"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, ya])
        with g.task(B.rw(), symbol="right"):
            wp.launch(axpy, dim=n, inputs=[2.0, x, yb])
            wp.launch(scale, dim=n, inputs=[0.5, yb])
        with g.task(A.read(), B.read(), OUT.rw(), symbol="join"):
            wp.copy(out, ya)  # a Warp builtin inside a task
            wp.launch(axpy, dim=n, inputs=[1.0, yb, out])

    graph = g.record(frame, frame_task=False)
    for _ in range(frames):
        graph.launch()
    wp.synchronize()
    ref_a, ref_b = 0.0, 0.0
    for _ in range(frames):
        ref_a += 1.0
        ref_b = 0.5 * (ref_b + 2.0)
    assert out.numpy()[0] == pytest.approx(ref_a + ref_b)
    del graph
    g.finalize()


def test_local_context_inside_warp_capture_parallel_branches():
    """A library routine opens a local context inside the caller's own
    wp.ScopedCapture: same result, k2 and k3 become parallel branches."""
    from cuda.bindings import runtime as rt

    n = 1 << 14

    def lib_stream_only(x, y, z):
        wp.launch(scale, dim=n, inputs=[2.0, x])
        wp.launch(axpy, dim=n, inputs=[1.0, x, y])
        wp.launch(axpy, dim=n, inputs=[1.0, x, z])
        wp.launch(axpy, dim=n, inputs=[1.0, y, z])

    def lib_tasks(x, y, z):
        with local_context() as lc:
            tx, ty, tz = (lc.tokens[k] for k in "xyz")
            with lc.task(tx.rw(), symbol="k1"):
                wp.launch(scale, dim=n, inputs=[2.0, x])
            with lc.task(tx.read(), ty.rw(), symbol="k2"):
                wp.launch(axpy, dim=n, inputs=[1.0, x, y])
            with lc.task(tx.read(), tz.rw(), symbol="k3"):
                wp.launch(axpy, dim=n, inputs=[1.0, x, z])
            with lc.task(ty.read(), tz.rw(), symbol="k4"):
                wp.launch(axpy, dim=n, inputs=[1.0, y, z])

    def longest_path(graph):
        h = int(getattr(graph.graph, "value", graph.graph))
        nodes = rt.cudaGraphGetNodes(h, rt.cudaGraphGetNodes(h, 0)[-1])[1]
        res = rt.cudaGraphGetEdges(h, rt.cudaGraphGetEdges(h, 0)[-1])
        idx = {int(a): i for i, a in enumerate(nodes)}
        succ = [[] for _ in nodes]
        indeg = [0] * len(nodes)
        for a, b in zip(res[1], res[2]):
            succ[idx[int(a)]].append(idx[int(b)])
            indeg[idx[int(b)]] += 1
        depth = [1] * len(nodes)
        order = [i for i, d in enumerate(indeg) if d == 0]
        k = 0
        while k < len(order):
            u = order[k]
            k += 1
            for v in succ[u]:
                depth[v] = max(depth[v], depth[u] + 1)
                indeg[v] -= 1
                if indeg[v] == 0:
                    order.append(v)
        return len(nodes), max(depth)

    results = {}
    for name, lib in (("stream", lib_stream_only), ("tasks", lib_tasks)):
        x, y, z = (
            wp.full(n, 1.0, dtype=float),
            wp.zeros(n, dtype=float),
            wp.zeros(n, dtype=float),
        )
        with wp.ScopedCapture() as cap:
            wp.launch(axpy, dim=n, inputs=[0.5, x, y])
            lib(x, y, z)
            wp.launch(axpy, dim=n, inputs=[0.5, x, z])
        for _ in range(3):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        results[name] = (float(z.numpy()[0]), *longest_path(cap.graph))
    assert results["stream"][0] == results["tasks"][0]
    assert results["stream"][1] == results["tasks"][1] == 6  # same kernel count
    assert results["tasks"][2] < results["stream"][2]  # k2 || k3


def test_frame_task_suspend_resume_bit_identical():
    """Library code adds graph-level tasks while a frame is being recorded."""
    n = 1 << 14

    class Solver:
        def __init__(self):
            self.a = wp.zeros(n, dtype=float)
            self.b = wp.zeros(n, dtype=float)
            self.f = wp.full(n, 1.0, dtype=float)

        def step(self, tasks):
            from cuda.stf._experimental.interop.warp import current

            g = current()
            if not tasks or g is None:
                wp.launch(axpy, dim=n, inputs=[1.0, self.f, self.a])
                wp.launch(axpy, dim=n, inputs=[2.0, self.f, self.b])
                wp.launch(scale, dim=n, inputs=[0.5, self.f])
                return
            was_open = g.suspend_frame()
            try:
                ta, tb, tf = g.tokens["a"], g.tokens["b"], g.frame_token
                with g.task(ta.rw(), tf.read(), symbol="a"):
                    wp.launch(axpy, dim=n, inputs=[1.0, self.f, self.a])
                with g.task(tb.rw(), tf.read(), symbol="b"):
                    wp.launch(axpy, dim=n, inputs=[2.0, self.f, self.b])
                with g.task(ta.read(), tb.read(), tf.rw(), symbol="barrier"):
                    wp.launch(scale, dim=n, inputs=[0.5, self.f])
            finally:
                g.resume_frame(was_open)

    def frame(solver, tasks):
        wp.launch(scale, dim=n, inputs=[1.5, solver.f])  # frame work outside tasks
        solver.step(tasks)
        wp.launch(axpy, dim=n, inputs=[1.0, solver.b, solver.a])  # and after

    fps = {}
    for tasks in (False, True):
        s = Solver()
        g = WarpTaskGraph()
        graph = g.record(functools.partial(frame, s, tasks))
        for _ in range(3):
            graph.launch()
        wp.synchronize()
        fps[tasks] = (
            float(s.a.numpy().sum()),
            float(s.b.numpy().sum()),
            float(s.f.numpy().sum()),
        )
        del graph
        g.finalize()
    assert fps[False] == fps[True]
    assert fps[True][0] != 0.0


def test_array_arg_dtype_override():
    ctx = stf.context()
    lx = ctx.logical_data(np.zeros(12, dtype=np.float32))
    with ctx.task(lx.rw()) as t:
        v = array_arg(t, 0, dtype=wp.vec3, shape=(4,))
        assert v.dtype == wp.vec3 and v.shape == (4,)
    ctx.finalize()
