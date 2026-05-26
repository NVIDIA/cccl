"""Two-step simulation mockup: ``robot`` then ``sand``, Warp + cuda.stf._experimental.

Strips the Newton ``example_mpm_anymal_stf`` scenario down to its bare
dataflow so we can study how the "one unified captured graph, each
sub-step a token-connected STF task" pattern actually behaves.

Setup mirrors the real scene:

    * ``x`` : the shared "body state" buffer (analog of ``state_0.body_q``).
      Step A writes it; step B reads it.
    * ``y`` : a "sand" buffer. Step B writes it.

Three execution paths are exercised and cross-checked numerically:

    1. ``run_eager``         -- no capture, direct ``wp.launch`` per step.
    2. ``run_two_subgraphs`` -- baseline from ``example_mpm_anymal.py``:
                                 each step in its own ``wp.ScopedCapture``,
                                 two ``wp.capture_launch`` calls per frame.
    3. ``run_stf_unified``   -- what the user asked for: one
                                 ``stf.task_graph()`` with two tasks joined
                                 by a token; one graph launch per frame.

Path (3) is where all the friction lives. If it passes here on trivial
kernels but fails inside Newton, the problem is in Newton's solver
stack (``wp.capture_while`` in multi-stream capture, per-frame
allocator activity, ...). If it fails here too, the combination is
fundamentally unsupported today and Newton is just an early victim.

Run:
    python test_two_step_sim_warp.py
"""

from __future__ import annotations

import numpy as np
import warp as wp

import cuda.stf._experimental as stf

N = 1 << 16  # buffer size
FRAMES = 5  # how many times to "step" each path


# ---------------------------------------------------------------------------
# Stream cache: double-registering the same raw cudaStream_t with Warp
# corrupts its bookkeeping, so we memoize one ``wp.Stream`` per raw ptr.
# ---------------------------------------------------------------------------

_wp_stream_cache: dict[tuple[int, int], wp.Stream] = {}


def wrap_stream(raw_ptr: int, device) -> wp.Stream:
    key = (id(device), int(raw_ptr))
    s = _wp_stream_cache.get(key)
    if s is None:
        s = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = s
    return s


# ---------------------------------------------------------------------------
# Kernels. Each one does a handful of fused ops so the "step" actually
# touches memory; nothing fancy, just to be closer to a real sub-step.
# ---------------------------------------------------------------------------


@wp.kernel
def robot_step(x: wp.array(dtype=wp.float32), dt: wp.float32):
    """Advance ``x`` in place. Analog of ``simulate_robot`` (many sub-steps)."""
    i = wp.tid()
    if i >= x.shape[0]:
        return
    v = x[i]
    for _ in range(4):  # 4 "sub-steps" like anymal
        v = v + dt * (1.0 - v)
    x[i] = v


@wp.kernel
def sand_step(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    dt: wp.float32,
):
    """Read ``x``, update ``y`` in place. Analog of ``simulate_sand``."""
    i = wp.tid()
    if i >= y.shape[0]:
        return
    y[i] = y[i] + dt * x[i]


# ---------------------------------------------------------------------------
# Path 1: eager. Baseline numerics.
# ---------------------------------------------------------------------------


def run_eager(device, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    dt = wp.float32(0.1)

    for _ in range(frames):
        wp.launch(robot_step, dim=N, inputs=[x, dt], device=device)
        wp.launch(sand_step, dim=N, inputs=[x, y, dt], device=device)

    wp.synchronize_device(device)
    return x.numpy(), y.numpy()


# ---------------------------------------------------------------------------
# Path 2: two independent ``wp.ScopedCapture`` sub-graphs. Same shape as
# the current ``example_mpm_anymal.py``.
# ---------------------------------------------------------------------------


def run_two_subgraphs(device, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    dt = wp.float32(0.1)

    with wp.ScopedCapture() as cap_a:
        wp.launch(robot_step, dim=N, inputs=[x, dt], device=device)
    robot_graph = cap_a.graph

    with wp.ScopedCapture() as cap_b:
        wp.launch(sand_step, dim=N, inputs=[x, y, dt], device=device)
    sand_graph = cap_b.graph

    for _ in range(frames):
        wp.capture_launch(robot_graph)
        wp.capture_launch(sand_graph)

    wp.synchronize_device(device)
    return x.numpy(), y.numpy()


# ---------------------------------------------------------------------------
# Path 3: one unified graph via STF ``push()`` + token-connected tasks +
# ``stf.task_graph()``. This is the shape we want Newton to have.
# ---------------------------------------------------------------------------


def run_stf_unified(device, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Two tasks joined by a ``ctx.token()`` -- what the user asked for."""
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    dt = wp.float32(0.1)

    graph = stf.task_graph()
    ctx = graph.context
    tok = ctx.token()

    with graph:
        with ctx.task(tok.write()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            with wp.ScopedStream(s, sync_enter=False):
                wp.launch(robot_step, dim=N, inputs=[x, dt], device=device, stream=s)
        with ctx.task(tok.read()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            with wp.ScopedStream(s, sync_enter=False):
                wp.launch(sand_step, dim=N, inputs=[x, y, dt], device=device, stream=s)

    for _ in range(frames):
        graph.launch()

    graph.reset()
    graph.finalize()

    wp.synchronize_device(device)
    return x.numpy(), y.numpy()


def run_stf_unified_ld(device, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``run_stf_unified`` but with a 1-byte ``logical_data`` as
    the sync carrier instead of a token. Used to probe whether the
    blocker is specifically token+push or all tasks+push.
    """
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    dt = wp.float32(0.1)

    graph = stf.task_graph()
    ctx = graph.context

    dep_host = np.zeros(1, dtype=np.uint8)
    ldep = ctx.logical_data(dep_host, name="body_q_dep")

    with graph:
        with ctx.task(ldep.rw()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            with wp.ScopedStream(s, sync_enter=False):
                wp.launch(robot_step, dim=N, inputs=[x, dt], device=device, stream=s)
        with ctx.task(ldep.read()) as t:
            s = wrap_stream(t.stream_ptr(), device)
            with wp.ScopedStream(s, sync_enter=False):
                wp.launch(sand_step, dim=N, inputs=[x, y, dt], device=device, stream=s)

    for _ in range(frames):
        graph.launch()

    graph.reset()
    graph.finalize()

    wp.synchronize_device(device)
    return x.numpy(), y.numpy()


def run_stf_single_task(device, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Fallback shape: one STF task holding both kernels. No token /
    logical_data needed; both kernels run on the same task stream.
    """
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    dt = wp.float32(0.1)

    graph = stf.task_graph()
    ctx = graph.context

    with graph:
        with ctx.task() as t:
            s = wrap_stream(t.stream_ptr(), device)
            with wp.ScopedStream(s, sync_enter=False):
                wp.launch(robot_step, dim=N, inputs=[x, dt], device=device, stream=s)
                wp.launch(sand_step, dim=N, inputs=[x, y, dt], device=device, stream=s)

    for _ in range(frames):
        graph.launch()

    graph.reset()
    graph.finalize()

    wp.synchronize_device(device)
    return x.numpy(), y.numpy()


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def _assert_close(a_ref, a_got, label: str, atol: float = 1e-5) -> None:
    assert np.allclose(a_ref, a_got, atol=atol), (
        f"{label} mismatch: max abs diff {np.max(np.abs(a_ref - a_got))}"
    )


def test_eager_vs_two_subgraphs() -> None:
    """Baseline sanity: the two-graph path agrees with eager."""
    device = wp.get_device("cuda:0")
    x_ref, y_ref = run_eager(device)
    x_got, y_got = run_two_subgraphs(device)
    _assert_close(x_ref, x_got, "two-subgraphs x")
    _assert_close(y_ref, y_got, "two-subgraphs y")


def test_stf_unified_matches_eager() -> None:
    """The thing we care about: one STF-unified graph agrees with eager."""
    device = wp.get_device("cuda:0")
    x_ref, y_ref = run_eager(device)
    x_got, y_got = run_stf_unified(device)
    _assert_close(x_ref, x_got, "stf-unified x")
    _assert_close(y_ref, y_got, "stf-unified y")


def _run_case(label: str, fn, x_ref, y_ref, device) -> None:
    print(f"  {label:<32s} ... ", end="", flush=True)
    try:
        x, y = fn(device)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL\n    {type(e).__name__}: {e}")
        return
    try:
        _assert_close(x_ref, x, f"{label} x")
        _assert_close(y_ref, y, f"{label} y")
    except AssertionError as e:
        print(f"WRONG ({e})")
        return
    print("OK")


if __name__ == "__main__":
    wp.init()
    dev = wp.get_device("cuda:0")

    print("reference (eager):")
    x_ref, y_ref = run_eager(dev)
    print(f"  x[0]={float(x_ref[0]):.6f}  y[0]={float(y_ref[0]):.6f}")

    import os

    # The token path aborts at the C level (hard exit, not catchable)
    # with "void_interface vs mdspan<char>" under a stackable graph. Gated
    # behind an env var so the other paths can still run.
    run_token = os.environ.get("STF_TWOSTEP_RUN_TOKEN", "0") == "1"

    print("paths:")
    _run_case("two subgraphs (baseline)", run_two_subgraphs, x_ref, y_ref, dev)
    if run_token:
        _run_case("STF unified (token)", run_stf_unified, x_ref, y_ref, dev)
    else:
        print(
            "  STF unified (token)              ... SKIP "
            "(hard-aborts; set STF_TWOSTEP_RUN_TOKEN=1 to run)"
        )
    _run_case("STF unified (logical_data)", run_stf_unified_ld, x_ref, y_ref, dev)
    _run_case("STF unified (single task)", run_stf_single_task, x_ref, y_ref, dev)
