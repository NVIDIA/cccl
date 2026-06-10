# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Demo: a DAG of captured tasks, each with a *local* STF context inside
that exposes intra-task concurrency. Two complementary STF patterns
composed inside a single unified ``cudaGraph_t``.

Two complementary uses of CUDASTF compose here:

* OUTER ("STF on the outside"): a ``stf.stackable_context`` drives a DAG
  of captured tasks that share one ``cudaGraph_t``. Sibling tasks with
  independent tokens become parallel branches in the unified graph.

* INNER ("STF on the inside"): inside each captured task, a local
  ``stf.context(stream=...)`` expresses fine-grain fork-join parallelism
  on that task's stream. The local context emits its sub-DAG directly
  into the surrounding capture; ``ctx.finalize()`` runs while the outer
  capture is still open.

The two compose: the resulting unified graph carries both *inter-task*
parallelism (sibling captured tasks A ‖ B) and *intra-task* parallelism
(fork-join interior of each task). One ``g.launch()`` per frame replays
the whole thing.

Compute scheme (illustrative, two parallel outer tasks each with a
fork-join interior, then a join task that consumes both)::

    graph = stf.task_graph()
    outer_ctx = graph.context
    with graph:
        ...
    ┌──────────────────────────────────────────────────────────────┐
    │                       unified cudaGraph_t                    │
    │                                                              │
    │  ┌─ outer A (tok_a.write) ─┐   ┌─ outer B (tok_b.write) ──┐  │
    │  │  local stf.context:     │   │  local stf.context:      │  │
    │  │    fill a1 ┐            │   │    fill b1 ┐             │  │
    │  │            ├─ reduce a  │   │            ├─ reduce b   │  │
    │  │    fill a2 ┘            │   │    fill b2 ┘             │  │
    │  └─────────────────────────┘   └──────────────────────────┘  │
    │                                                              │
    │  ┌─ outer C (tok_a.read, tok_b.read, tok_c.write) ─────────┐ │
    │  │   c[i] = a[i] + b[i]                                    │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────┘
    for _ in range(FRAMES):
        graph.launch()

Capture-mode note: the outer stackable graph_tasks already use
``cudaStreamCaptureModeRelaxed`` internally (see ``graph_task.cuh``),
so the secondary ``stf.context`` opened inside a captured task gets
that Relaxed-mode tolerance "for free": no extra setup is needed for
its first-touch capture-unsafe runtime calls.

API note: every Warp+STF task in this file is opened via
``warp.stf_experimental.task(...)``, which fuses the four
boilerplate steps that every Warp+STF integration needs:

  * caches one ``wp.Stream`` per raw ``cudaStream_t``;
  * pushes the task stream as Warp's active stream via
    ``wp.ScopedStream(s, sync_enter=False)``, so ``wp.empty()`` /
    ``wp.zeros()`` / ``wp.launch()`` calls without an explicit
    ``stream=`` land on the task stream;
  * auto-detects via ``cudaStreamIsCapturing`` whether the task's
    stream is part of an active CUDA graph capture; if so, wraps the
    body in ``wp.capture_begin(stream=s, external=True)`` /
    ``wp.capture_end`` so Warp's allocator bookkeeping tracks each
    alloc and the matching ``MEM_FREE`` is emitted with the task's
    tail as its predecessor (not the whole graph's leaves);
  * exposes any non-token deps as zero-copy ``wp.array`` views.

Without that capture-bookkeeping on the outer tasks, ``wp.empty()``
would run on Warp's default (uncaptured) stream: allocations miss the
graph entirely, the pool reuses the same physical address for A and
B, and the two parallel siblings race on shared scratch memory. This
is the same pattern ``example_mpm_anymal_stf.py::_record_task`` uses
to wrap solver calls inside captured STF tasks; ``wp_stf.task`` makes
it the default.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

wp = pytest.importorskip("warp")
wp_stf = pytest.importorskip("warp.stf_experimental")

N = 1 << 14
FRAMES = 3

INIT_A1 = 1
INIT_A2 = 2
INIT_B1 = 4
INIT_B2 = 8


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------


@wp.kernel
def fill_kernel(arr: wp.array(dtype=wp.int32), value: wp.int32):
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    arr[i] = value


@wp.kernel
def add_kernel(
    out: wp.array(dtype=wp.int32),
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= out.shape[0]:
        return
    out[i] = a[i] + b[i]


# ---------------------------------------------------------------------------
# Inner sub-DAG, executed inside one outer captured task.
#
# Structure:
#   fill v1 (tok_v1.write)  ┐
#                            ├──▶ reduce dst = v1 + v2 (tok_v1.read,
#   fill v2 (tok_v2.write)  ┘                          tok_v2.read,
#                                                      tok_dst.write)
#
# fill_v1 and fill_v2 have no shared input: STF emits them as parallel
# branches inside the surrounding capture.
# ---------------------------------------------------------------------------


def _inner_fork_join(
    outer_stream: wp.Stream,
    device,
    *,
    dst: wp.array,
    val1: int,
    val2: int,
):
    # Per-task scratchpads allocated on the capturing stream (via the
    # outer ``wp_stf.task(..., capture=True)`` ScopedStream). Warp sees
    # the external capture, so the MEM_ALLOC/FREE nodes land inside the
    # graph and sibling tasks end up with distinct, non-aliasing addrs.
    v1 = wp.empty(N, dtype=wp.int32, device=device)
    v2 = wp.empty(N, dtype=wp.int32, device=device)

    inner_ctx = stf.context(stream=int(outer_stream.cuda_stream))
    tok_v1 = inner_ctx.token()
    tok_v2 = inner_ctx.token()
    tok_dst = inner_ctx.token()

    with wp_stf.task(inner_ctx, tok_v1.write()) as (s,):
        wp.launch(fill_kernel, dim=N, inputs=[v1, val1], stream=s)

    with wp_stf.task(inner_ctx, tok_v2.write()) as (s,):
        wp.launch(fill_kernel, dim=N, inputs=[v2, val2], stream=s)

    with wp_stf.task(inner_ctx, tok_v1.read(), tok_v2.read(), tok_dst.write()) as (s,):
        wp.launch(add_kernel, dim=N, inputs=[dst, v1, v2], stream=s)

    # finalize the inner DAG while the outer capture is still open
    inner_ctx.finalize()


# ---------------------------------------------------------------------------
# Path 1: pure-eager reference. No capture, no STF. Used as a numerical
# oracle for the unified-graph result.
# ---------------------------------------------------------------------------


def run_eager(device, frames: int = FRAMES) -> np.ndarray:
    a = wp.empty(N, dtype=wp.int32, device=device)
    b = wp.empty(N, dtype=wp.int32, device=device)
    c = wp.empty(N, dtype=wp.int32, device=device)
    v1 = wp.empty(N, dtype=wp.int32, device=device)
    v2 = wp.empty(N, dtype=wp.int32, device=device)

    for _ in range(frames):
        wp.launch(fill_kernel, dim=N, inputs=[v1, INIT_A1], device=device)
        wp.launch(fill_kernel, dim=N, inputs=[v2, INIT_A2], device=device)
        wp.launch(add_kernel, dim=N, inputs=[a, v1, v2], device=device)

        wp.launch(fill_kernel, dim=N, inputs=[v1, INIT_B1], device=device)
        wp.launch(fill_kernel, dim=N, inputs=[v2, INIT_B2], device=device)
        wp.launch(add_kernel, dim=N, inputs=[b, v1, v2], device=device)

        wp.launch(add_kernel, dim=N, inputs=[c, a, b], device=device)

    wp.synchronize_device(device)
    return c.numpy()


# ---------------------------------------------------------------------------
# Path 2: outer stackable_context with three captured tasks. The first
# two are parallel siblings (independent tokens); each opens a local
# stf.context inside to expose intra-task fork-join concurrency. The
# third joins them.
# ---------------------------------------------------------------------------


def run_unified_with_local_stf(device, frames: int = FRAMES) -> np.ndarray:
    a = wp.empty(N, dtype=wp.int32, device=device)
    b = wp.empty(N, dtype=wp.int32, device=device)
    c = wp.empty(N, dtype=wp.int32, device=device)

    graph = stf.task_graph()
    outer_ctx = graph.context

    tok_a = outer_ctx.token()
    tok_b = outer_ctx.token()
    tok_c = outer_ctx.token()

    with graph:
        # Parallel sibling A: fork-join inside, writes ``a``.
        # ``capture=`` is auto-detected via cudaStreamIsCapturing -- True for
        # outer tasks inside task_graph(), True for the inner-ctx tasks below
        # (their streams fork from the outer capturing stream), and False
        # for plain eager use.
        with wp_stf.task(outer_ctx, tok_a.write()) as (s,):
            _inner_fork_join(s, device, dst=a, val1=INIT_A1, val2=INIT_A2)

        # Parallel sibling B: fork-join inside, writes ``b``.
        with wp_stf.task(outer_ctx, tok_b.write()) as (s,):
            _inner_fork_join(s, device, dst=b, val1=INIT_B1, val2=INIT_B2)

        # Join: reads both, writes ``c``. Single Warp launch, no inner ctx.
        with wp_stf.task(
            outer_ctx,
            tok_a.read(),
            tok_b.read(),
            tok_c.write(),
        ) as (s,):
            wp.launch(add_kernel, dim=N, inputs=[c, a, b], stream=s)

    for _ in range(frames):
        graph.launch()

    graph.reset()
    graph.finalize()

    wp.synchronize_device(device)
    return c.numpy()


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def _assert_all_equal(arr: np.ndarray, expected: int, label: str) -> None:
    if not np.all(arr == expected):
        uniq = np.unique(arr).tolist()
        raise AssertionError(
            f"{label}: expected all == {expected}, got unique values {uniq}"
        )


def test_unified_dag_with_local_stf_matches_eager() -> None:
    """One ``g.launch()`` per frame produces the same result as the eager
    fork-join + tail dataflow.
    """
    wp.init()
    device = wp.get_device("cuda:0")

    expected = (INIT_A1 + INIT_A2) + (INIT_B1 + INIT_B2)

    c_ref = run_eager(device)
    _assert_all_equal(c_ref, expected, "eager reference")

    c_got = run_unified_with_local_stf(device)
    _assert_all_equal(c_got, expected, "unified DAG with local STF")


if __name__ == "__main__":
    test_unified_dag_with_local_stf_matches_eager()
    print("DAG of captured tasks with local STF inside           : OK")
