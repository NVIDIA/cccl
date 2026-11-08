# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for ``cuda.stf._experimental.interop.warp``: scopes, tasks, native blocks."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental import AccessMode  # noqa: E402
from cuda.stf._experimental.interop.warp import (  # noqa: E402
    array_arg,
    release_graph_handles,
    scope,
    token,
    warp_native,
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


@wp.kernel
def count_down(counter: wp.array(dtype=int), acc: wp.array(dtype=float)):
    i = wp.tid()
    acc[i] = acc[i] + 1.0
    if i == 0:
        counter[0] = counter[0] - 1


@pytest.fixture(scope="module", autouse=True)
def _warm():
    wp.init()
    x = wp.zeros(8, dtype=float)
    c = wp.zeros(1, dtype=int)
    wp.launch(axpy, dim=8, inputs=[0.0, x, x])
    wp.launch(scale, dim=8, inputs=[1.0, x])
    wp.launch(count_down, dim=8, inputs=[c, x])
    wp.synchronize()


def graph_topology(graph_handle):
    """(kernel nodes, longest path in kernels) with child graphs (nested scopes) expanded."""
    from cuda.bindings import runtime as rt

    def walk(h):
        h = int(h)
        n = rt.cudaGraphGetNodes(h, 0)[-1]
        nodes = rt.cudaGraphGetNodes(h, n)[1]
        m = rt.cudaGraphGetEdges(h, 0)[-1]
        res = rt.cudaGraphGetEdges(h, m)
        idx = {int(x): i for i, x in enumerate(nodes)}
        succ = [[] for _ in nodes]
        indeg = [0] * len(nodes)
        for a, b in zip(res[1], res[2]):
            succ[idx[int(a)]].append(idx[int(b)])
            indeg[idx[int(b)]] += 1
        weight, count = [0] * len(nodes), 0
        for i, x in enumerate(nodes):
            t = rt.cudaGraphNodeGetType(x)[1]
            if t == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
                c, w = walk(rt.cudaGraphChildGraphNodeGetGraph(x)[1])
                weight[i], count = w, count + c
            elif t == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                weight[i], count = 1, count + 1
        depth = weight[:]
        order = [i for i, d in enumerate(indeg) if d == 0]
        k = 0
        while k < len(order):
            u = order[k]
            k += 1
            for v in succ[u]:
                depth[v] = max(depth[v], depth[u] + weight[v])
                indeg[v] -= 1
                if indeg[v] == 0:
                    order.append(v)
        return count, (max(depth) if nodes else 0)

    return walk(graph_handle)


def _record_and_replay(frame, replays):
    """Record ``frame(ctx)`` in a launchable scope, replay it; returns the
    (kernel nodes, longest path) topology."""
    ctx = stf.stackable_context()
    try:
        with scope(ctx, launchable=True) as g:
            frame(ctx)
            topo = graph_topology(g.graph)
            for _ in range(replays):
                g.launch()
        wp.synchronize()
    finally:
        ctx.finalize()
        release_graph_handles()
    return topo


# ---------------------------------------------------------------------------
# streams, arrays
# ---------------------------------------------------------------------------


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


def test_array_arg_dtype_override():
    ctx = stf.context()
    lx = ctx.logical_data(np.zeros(12, dtype=np.float32))
    with ctx.task(lx.rw()) as t:
        v = array_arg(t, 0, dtype=wp.vec3, shape=(4,))
        assert v.dtype == wp.vec3 and v.shape == (4,)
    ctx.finalize()


# ---------------------------------------------------------------------------
# scopes, tasks, native blocks
# ---------------------------------------------------------------------------


def test_fork_join_record_replay():
    n = 1 << 16
    frames = 4
    x = wp.full(n, 1.0, dtype=float)
    ya, yb, out = (wp.zeros(n, dtype=float) for _ in range(3))

    def frame(ctx):
        A, B, OUT = (token(ctx, k) for k in ("a", "b", "out"))
        with warp_task(ctx, A.rw(), symbol="left"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, ya])
        with warp_task(ctx, B.rw(), symbol="right"):
            wp.launch(axpy, dim=n, inputs=[2.0, x, yb])
            wp.launch(scale, dim=n, inputs=[0.5, yb])
        with warp_task(ctx, A.read(), B.read(), OUT.rw(), symbol="join"):
            wp.copy(out, ya)  # a Warp builtin inside a task
            wp.launch(axpy, dim=n, inputs=[1.0, yb, out])

    kernels, longest = _record_and_replay(frame, frames)
    assert kernels == 4 and longest == 3  # left || right(2) -> join
    ref_a, ref_b = 0.0, 0.0
    for _ in range(frames):
        ref_a += 1.0
        ref_b = 0.5 * (ref_b + 2.0)
    assert out.numpy()[0] == pytest.approx(ref_a + ref_b)


def test_native_blocks_ordered_with_tasks():
    """native ; task(A) ; task(B) ; native: A and B are siblings, the native
    blocks are barriers; result equals the sequential one."""
    n = 1 << 14

    def body(ctx, x, y, z):
        with warp_native(ctx, "k1"):
            wp.launch(scale, dim=n, inputs=[2.0, x])
        ty, tz = token(ctx, "y"), token(ctx, "z")
        with warp_task(ctx, ty.rw(), symbol="A"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, y])
        with warp_task(ctx, tz.rw(), symbol="B"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, z])
        with warp_native(ctx, "k4"):
            wp.launch(axpy, dim=n, inputs=[1.0, y, z])

    results = {}
    for arm in ("sequential", "tasks"):
        x, y, z = (
            wp.full(n, 1.0, dtype=float),
            wp.zeros(n, dtype=float),
            wp.zeros(n, dtype=float),
        )
        if arm == "sequential":
            for _ in range(3):
                body(None, x, y, z)  # no context: every name is a no-op
            wp.synchronize()
            topo = None
        else:
            topo = _record_and_replay(lambda ctx: body(ctx, x, y, z), 3)
        results[arm] = float(z.numpy()[0])
    assert results["sequential"] == results["tasks"]
    assert topo == (4, 3)  # k1 -> {A, B} -> k4


def test_back_to_back_tasks_stay_siblings():
    n = 1 << 12
    a, b = wp.zeros(n, dtype=float), wp.zeros(n, dtype=float)

    def frame(ctx):
        ta, tb = token(ctx), token(ctx)
        with warp_task(ctx, ta.rw()):
            wp.launch(scale, dim=n, inputs=[2.0, a])
        with warp_task(ctx, tb.rw()):
            wp.launch(scale, dim=n, inputs=[2.0, b])

    assert _record_and_replay(frame, 1) == (2, 1)  # no edge between them


def test_nested_scope_inner_fork_bit_identical():
    """A library called from a task opens its own scope and forks inside:
    the parent sees one node, results are bit-identical, the path is shorter."""
    n = 1 << 14

    class Entry:
        def __init__(self, ctx, fork):
            self.ctx, self.fork = ctx, fork
            self.v = wp.zeros(n, dtype=float)
            self.mass = wp.ones(n, dtype=float)
            self.rows = wp.ones(n, dtype=float)
            self.t_mass, self.t_rows = token(ctx), token(ctx)

        def step(self, forcing):
            with scope(self.ctx):
                if self.fork:
                    with warp_task(self.ctx, self.t_mass.rw()):
                        wp.launch(scale, dim=n, inputs=[0.999, self.mass])
                    with warp_task(self.ctx, self.t_rows.rw()):
                        wp.launch(axpy, dim=n, inputs=[0.001, forcing, self.rows])
                else:
                    with warp_native(self.ctx):
                        wp.launch(scale, dim=n, inputs=[0.999, self.mass])
                        wp.launch(axpy, dim=n, inputs=[0.001, forcing, self.rows])
                with warp_native(self.ctx):
                    wp.launch(axpy, dim=n, inputs=[0.5, self.rows, self.v])
                    wp.launch(axpy, dim=n, inputs=[0.5, self.mass, self.v])

    out = {}
    for fork in (False, True):
        ctx = stf.stackable_context()
        e = Entry(ctx, fork)
        forcing = wp.ones(n, dtype=float)
        t_e = token(ctx)
        with scope(ctx, launchable=True) as g:
            with warp_native(ctx, "pre"):
                wp.launch(scale, dim=n, inputs=[1.5, forcing])
            with warp_task(ctx, t_e.rw(), symbol="solve"):
                e.step(forcing)
            with warp_native(ctx, "post"):
                wp.launch(axpy, dim=n, inputs=[1.0, e.v, forcing])
            topo = graph_topology(g.graph)
            for _ in range(3):
                g.launch()
        wp.synchronize()
        out[fork] = (float(e.v.numpy().sum()), float(forcing.numpy().sum()), topo)
        ctx.finalize()
        release_graph_handles()
    assert out[False][:2] == out[True][:2]
    assert out[False][2] == (6, 6)
    assert out[True][2] == (6, 5)


def test_sibling_scopes_overlap_with_read_push():
    """A root token used inside sibling scopes is auto-imported read-write
    (serialising them); ``push(READ)`` at entry keeps them siblings."""
    n = 1 << 12

    def run(push_read):
        x = wp.ones(n, dtype=float)
        a, b = wp.zeros(n, dtype=float), wp.zeros(n, dtype=float)

        def frame(ctx):
            tx, ta, tb = token(ctx), token(ctx), token(ctx)
            for tok, arr in ((ta, a), (tb, b)):
                with scope(ctx):
                    if push_read:
                        tx.push(AccessMode.READ)
                    with warp_task(ctx, tx.read(), tok.rw()):
                        wp.launch(axpy, dim=n, inputs=[1.0, x, arr])

        topo = _record_and_replay(frame, 2)
        return topo, float(a.numpy()[0]) + float(b.numpy()[0])

    serial, v1 = run(False)
    sibl, v2 = run(True)
    assert v1 == v2 == 4.0
    assert serial == (2, 2) and sibl == (2, 1)


def test_capture_while_inside_nested_scope_task():
    """Warp's conditional capture needs a capture-origin stream: task streams
    of a (nested) graph scope are such streams."""
    n = 64
    acc = wp.zeros(n, dtype=float)
    counter = wp.zeros(1, dtype=int)
    reset = wp.full(1, 3, dtype=int)

    def frame(ctx):
        t = token(ctx)
        with warp_task(ctx, t.rw()):
            with scope(ctx):
                with warp_native(ctx):
                    wp.copy(counter, reset)
                    wp.capture_while(
                        counter,
                        lambda: wp.launch(count_down, dim=n, inputs=[counter, acc]),
                    )

    _record_and_replay(frame, 2)
    assert acc.numpy()[0] == 6.0 and counter.numpy()[0] == 0


def test_poison_raises_outside_blocks():
    n = 8
    y = wp.zeros(n, dtype=float)
    ctx = stf.stackable_context()
    try:
        with pytest.raises(RuntimeError, match="outside warp_task"):
            with scope(ctx, launchable=True):
                wp.launch(scale, dim=n, inputs=[2.0, y])  # forgot warp_native()
    finally:
        ctx.finalize()
    # the current stream is restored after the scope
    wp.launch(scale, dim=n, inputs=[2.0, y])
    wp.synchronize()


def test_warp_native_requires_scope():
    ctx = stf.stackable_context()
    try:
        with pytest.raises(RuntimeError, match="open scope"):
            with warp_native(ctx):
                pass
    finally:
        ctx.finalize()


def test_no_context_is_a_no_op():
    n = 8
    y = wp.ones(n, dtype=float)
    t = token(None)
    assert t.rw() is None and t.read() is None
    with scope(None) as g:
        assert g is None
        with warp_task(None, t.rw()) as (s, arrays):
            assert s is wp.get_stream() and arrays == ()
            wp.launch(scale, dim=n, inputs=[2.0, y])
        with warp_native(None):
            wp.launch(scale, dim=n, inputs=[2.0, y])
    wp.synchronize()
    assert y.numpy()[0] == 4.0


def test_scope_between_blocks_is_ordered_inside_task_is_not():
    """A scope opened between tasks / native blocks is ordered by its
    dependencies; one opened inside a running task is a sibling of that
    task's own launches (documented limitation)."""
    n = 1 << 16
    expected = (1.01**20 + 1.0) * 1.01**20

    def run(inside):
        ctx = stf.stackable_context()
        t = token(ctx)
        y = wp.ones(n, dtype=float)
        with scope(ctx, launchable=True) as g:
            if inside:
                with warp_task(ctx, t.rw()):
                    for _ in range(20):
                        wp.launch(scale, dim=n, inputs=[1.01, y])
                    with scope(ctx):
                        with warp_native(ctx):
                            wp.launch(
                                axpy, dim=n, inputs=[1.0, wp.ones(n, dtype=float), y]
                            )
                    for _ in range(20):
                        wp.launch(scale, dim=n, inputs=[1.01, y])
            else:
                with warp_task(ctx, t.rw()):
                    for _ in range(20):
                        wp.launch(scale, dim=n, inputs=[1.01, y])
                with scope(ctx, writes=(t,)):
                    with warp_native(ctx):
                        wp.launch(axpy, dim=n, inputs=[1.0, wp.ones(n, dtype=float), y])
                with warp_task(ctx, t.rw()):
                    for _ in range(20):
                        wp.launch(scale, dim=n, inputs=[1.01, y])
            g.launch()
        wp.synchronize()
        v = float(y.numpy()[0])
        ctx.finalize()
        release_graph_handles()
        return v

    assert run(False) == pytest.approx(expected, rel=1e-5)
    assert run(True) != pytest.approx(expected, rel=1e-5)


def test_scope_allocations_belong_to_the_level():
    """Warp arrays created inside a nested scope are served by the level's STF
    allocator (no allocation node in the child graph); one dropped after the
    scope exits must not raise; results survive the replays."""
    from cuda.bindings import runtime as rt

    n = 1 << 12
    ctx = stf.stackable_context()
    t = token(ctx)
    keep = []
    with scope(ctx, launchable=True) as g:
        with warp_task(ctx, t.rw()):
            pass
        with scope(ctx, writes=(t,)):
            with warp_native(ctx):
                tmp = wp.zeros(
                    n, dtype=float
                )  # per-step temporary: dropped inside the scope
                wp.launch(scale, dim=n, inputs=[2.0, tmp])
                keep.append(wp.full(n, 1.0, dtype=float))  # outlives the scope
                wp.launch(scale, dim=n, inputs=[2.0, keep[0]])
                del tmp
        # no MemAlloc node anywhere
        h = int(g.graph)
        nodes = rt.cudaGraphGetNodes(h, rt.cudaGraphGetNodes(h, 0)[-1])[1]
        for x in nodes:
            tt = rt.cudaGraphNodeGetType(x)[1]
            assert tt != rt.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc
            if tt == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
                c = rt.cudaGraphChildGraphNodeGetGraph(x)[1]
                for y in rt.cudaGraphGetNodes(c, rt.cudaGraphGetNodes(c, 0)[-1])[1]:
                    assert (
                        rt.cudaGraphNodeGetType(y)[1]
                        != rt.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc
                    )
        for _ in range(3):
            g.launch()
        wp.synchronize()
        assert keep[0].numpy()[0] == 2.0  # the fill is part of the recorded graph: every replay resets then scales
    keep.clear()  # Warp frees after the scope: ignored by the adapter
    ctx.finalize()
    release_graph_handles()
