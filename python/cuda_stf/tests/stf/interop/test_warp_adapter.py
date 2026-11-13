# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for ``cuda.stf._experimental.interop.warp``: scopes, tasks, tokens."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

import cuda.stf._experimental as stf
from cuda.stf._experimental import AccessMode
from cuda.stf._experimental.interop.warp import (
    current_context,
    get_arg_warp,
    release_graph_handles,
    scope,
    scope_token,
    token,
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


def test_get_arg_warp_dtype_override():
    ctx = stf.context()
    lx = ctx.logical_data(np.zeros(12, dtype=np.float32))
    with ctx.task(lx.rw()) as t:
        v = get_arg_warp(t, 0, dtype=wp.vec3, shape=(4,))
        assert v.dtype == wp.vec3 and v.shape == (4,)
    ctx.finalize()


# ---------------------------------------------------------------------------
# scopes, tasks, tokens
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


def test_scope_token_orders_tasks():
    """k1 ; A ; B ; k4 written with the scope token: k1 and k4 take it
    read-write, A and B read it and overlap; the result equals the sequential
    one. The same function runs unrecorded, where every name is a no-op."""
    n = 1 << 14

    def body(x, y, z):
        s = scope_token()
        assert scope_token() is s or current_context() is None  # stable in a scope
        with warp_task(s.rw(), symbol="k1"):
            wp.launch(scale, dim=n, inputs=[2.0, x])
        with warp_task(s.read(), symbol="A"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, y])
        with warp_task(s.read(), symbol="B"):
            wp.launch(axpy, dim=n, inputs=[1.0, x, z])
        with warp_task(s.rw(), symbol="k4"):
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
                body(x, y, z)  # nothing is recorded
            wp.synchronize()
            topo = None
        else:
            topo = _record_and_replay(lambda ctx: body(x, y, z), 3)
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


def test_library_in_its_own_scope_forks_bit_identical():
    """A library that records its own tasks is given a scope by its caller:
    the parent sees one node, ordered by what the scope is given; inside, the
    library's sequence is its own scope token. Results are bit-identical with
    and without the fork, the path is shorter with it."""
    n = 1 << 14

    class Entry:
        def __init__(self, fork):
            self.fork = fork
            self.v = wp.zeros(n, dtype=float)
            self.mass = wp.ones(n, dtype=float)
            self.rows = wp.ones(n, dtype=float)

        def step(self, forcing):
            s = scope_token()
            if self.fork:
                with warp_task(s.read(), symbol="mass"):
                    wp.launch(scale, dim=n, inputs=[0.999, self.mass])
                with warp_task(s.read(), symbol="rows"):
                    wp.launch(axpy, dim=n, inputs=[0.001, forcing, self.rows])
            else:
                with warp_task(s.rw()):
                    wp.launch(scale, dim=n, inputs=[0.999, self.mass])
                    wp.launch(axpy, dim=n, inputs=[0.001, forcing, self.rows])
            with warp_task(s.rw()):
                wp.launch(axpy, dim=n, inputs=[0.5, self.rows, self.v])
                wp.launch(axpy, dim=n, inputs=[0.5, self.mass, self.v])

    out = {}
    for fork in (False, True):
        ctx = stf.stackable_context()
        e = Entry(fork)
        forcing = wp.ones(n, dtype=float)
        with scope(ctx, launchable=True) as g:
            s = scope_token()
            with warp_task(s.rw(), symbol="pre"):
                wp.launch(scale, dim=n, inputs=[1.5, forcing])
            with scope(writes=(s,), symbol="solve"):
                assert scope_token() is not s  # the library's own
                e.step(forcing)
            assert scope_token() is s
            with warp_task(s.rw(), symbol="post"):
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
        with scope(ctx), warp_task(ctx):
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
                wp.launch(scale, dim=n, inputs=[2.0, y])  # forgot warp_task()
    finally:
        ctx.finalize()
    # the current stream is restored after the scope
    wp.launch(scale, dim=n, inputs=[2.0, y])
    wp.synchronize()


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
        with warp_task(scope_token().rw()):  # no scope: an inert token, no task
            wp.launch(scale, dim=n, inputs=[2.0, y])
    wp.synchronize()
    assert y.numpy()[0] == 4.0


def test_library_runs_from_a_scope_from_a_task_and_unrecorded():
    """The same library code (no context argument) records its own tasks when
    called from a scope, runs as part of the task when its caller wrapped the
    call in one, and runs as plain Warp code when nothing is recorded. With an
    explicit context, a task or a scope inside a task is refused."""
    n = 1 << 16
    expected = (1.01**20 + 1.0) * 1.01**20

    def library(y, one):
        s = scope_token()
        with warp_task(s.rw(), symbol="first"):
            for _ in range(20):
                wp.launch(scale, dim=n, inputs=[1.01, y])
        with warp_task(s.read(), symbol="add"):
            wp.launch(axpy, dim=n, inputs=[1.0, one, y])
        with warp_task(s.read(), symbol="idle"):
            pass
        with scope(symbol="unused"):  # a no-op inside a task, a scope otherwise
            pass
        with warp_task(s.rw(), symbol="second"):
            for _ in range(20):
                wp.launch(scale, dim=n, inputs=[1.01, y])

    def run(form):
        y = wp.ones(n, dtype=float)
        one = wp.ones(n, dtype=float)
        assert current_context() is None
        if form == "unrecorded":
            library(y, one)
        else:
            ctx = stf.stackable_context()
            with scope(ctx, launchable=True) as g:
                assert current_context() is ctx
                if form == "scope":
                    library(y, one)
                    assert graph_topology(g.graph) == (41, 41)
                else:
                    with warp_task(symbol="caller"):
                        library(y, one)
                        with pytest.raises(RuntimeError, match="inside task"):
                            warp_task(ctx, symbol="nested")
                        with pytest.raises(RuntimeError, match="inside task"):
                            with scope(ctx):
                                pass
                g.launch()
            ctx.finalize()
            release_graph_handles()
        assert current_context() is None
        wp.synchronize()
        return float(y.numpy()[0])

    for form in ("unrecorded", "scope", "task"):
        assert run(form) == pytest.approx(expected, rel=1e-5), form


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
        with scope(ctx, writes=(t,)), warp_task(ctx):
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
        assert (
            keep[0].numpy()[0] == 2.0
        )  # the fill is part of the recorded graph: every replay resets then scales
    keep.clear()  # Warp frees after the scope: ignored by the adapter
    ctx.finalize()
    release_graph_handles()


def test_scope_allocator_only_under_capture():
    """Inside a scope but outside any task nothing is capturing: an allocation
    there is Warp's own (lives as long as the Python object), not the level's."""
    from cuda.stf._experimental.interop import warp as adapter

    n = 1 << 10
    ctx = stf.stackable_context()
    t = token(ctx)
    with scope(ctx, launchable=True) as g:
        a = wp.empty(
            n, dtype=float
        )  # no stream work: the poison stays quiet, no capture is active
        assert not adapter._ScopeAllocator._owner  # not level-owned
        with warp_task(ctx, t.rw()):
            b = wp.zeros(n, dtype=float)  # under capture: the level's
            assert len(adapter._ScopeAllocator._owner) == 1
            wp.launch(scale, dim=n, inputs=[2.0, b])
        g.launch()
    ctx.finalize()
    release_graph_handles()
    a.fill_(3.0)  # eager use after the scope: fine, Warp owns it
    wp.synchronize()
    assert a.numpy()[0] == 3.0
