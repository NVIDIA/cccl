# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Neural ODE RK4 with CUDASTF: capture the integrator once, replay it many times.

This example integrates a small neural vector field ``f_theta(y)`` (a 3-layer
MLP) with a classical fixed-step RK4 loop and shows how STF turns it into a
replayable CUDA graph: the loop runs inside ``ctx.graph_scope() +
ctx.repeat(N)``, so the step body is CUDA-graph-captured once and replayed N
times instead of paying Python dispatch + kernel-launch overhead on every step.

(The companion ``neural_ode_dopri5.py`` covers the adaptive-step variant using
``ctx.while_loop`` with a device-side termination scalar.)

The STF integrator shares its compiled step body with a plain eager PyTorch
reference, and the test asserts that the STF trajectory matches that reference.
The eager loop is also the relatable "pain-point" baseline: on a small per-step
MLP its per-iteration Python overhead dominates the ~50 us of kernel time, which
is exactly the gap that graph replay recovers.

Run it directly to validate the trajectory::

    python neural_ode_rk4.py

Set ``LLM_NODE_BENCH=1`` to additionally print eager-vs-STF wall-clock timings
(informational only -- nothing is asserted on performance). The problem size can
be tuned with ``LLM_NODE_B`` / ``LLM_NODE_D`` / ``LLM_NODE_H`` / ``LLM_NODE_N``.

Why the specific shapes: B=64, D=32, H=128 sizes the per-step MLP to ~4.7
MFLOPs (~50 us of kernel time on an A100/H100), small enough that eager
PyTorch's per-iter Python dispatch overhead dominates; N=500 iterations gives
enough replays that STF's graph-capture setup is fully amortised.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeConfig:
    """Workload shape for the Neural ODE benchmark."""

    batch: int = 64
    state_dim: int = 32  # D
    hidden_dim: int = 128  # H
    n_steps: int = 500
    t0: float = 0.0
    t1: float = 1.0
    dtype: str = "float32"

    @property
    def h_step(self) -> float:
        return (self.t1 - self.t0) / float(self.n_steps)

    @property
    def np_dtype(self):
        return np.float32 if self.dtype == "float32" else np.float64

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float64


def _default_cfg() -> NodeConfig:
    return NodeConfig(
        batch=int(os.environ.get("LLM_NODE_B", "64")),
        state_dim=int(os.environ.get("LLM_NODE_D", "32")),
        hidden_dim=int(os.environ.get("LLM_NODE_H", "128")),
        n_steps=int(os.environ.get("LLM_NODE_N", "500")),
    )


SEED = 0xC0DE


# ---------------------------------------------------------------------------
# Weight factory
#
# Weights live as numpy arrays + torch CUDA tensors so PyTorch baselines and
# STF tasks see bit-identical parameters and the correctness test is
# meaningful. Shapes are stored in PRE-transposed layout (in, out) so the
# per-layer op is a plain ``addmm(b, y, W)`` without a .T transpose, which
# keeps the compiled body fusion-friendly.
# ---------------------------------------------------------------------------


@dataclass
class MLPWeights:
    W1: np.ndarray  # (D, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, H)
    b2: np.ndarray  # (H,)
    W3: np.ndarray  # (H, D)
    b3: np.ndarray  # (D,)

    def as_torch(self, device="cuda", dtype=torch.float32) -> "MLPWeightsT":
        return MLPWeightsT(
            W1=torch.as_tensor(self.W1, device=device, dtype=dtype).contiguous(),
            b1=torch.as_tensor(self.b1, device=device, dtype=dtype).contiguous(),
            W2=torch.as_tensor(self.W2, device=device, dtype=dtype).contiguous(),
            b2=torch.as_tensor(self.b2, device=device, dtype=dtype).contiguous(),
            W3=torch.as_tensor(self.W3, device=device, dtype=dtype).contiguous(),
            b3=torch.as_tensor(self.b3, device=device, dtype=dtype).contiguous(),
        )


@dataclass
class MLPWeightsT:
    W1: "torch.Tensor"
    b1: "torch.Tensor"
    W2: "torch.Tensor"
    b2: "torch.Tensor"
    W3: "torch.Tensor"
    b3: "torch.Tensor"

    def tuple(self):
        return (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)


def build_weights(cfg: NodeConfig, *, seed: int = 0) -> MLPWeights:
    rng = np.random.default_rng(seed + 1)
    D, H = cfg.state_dim, cfg.hidden_dim
    # Xavier-style scaling so tanh stays well in its linear regime. The
    # integrator is only stable when the field magnitude is bounded and
    # predictable, so keeping ||f(y)|| ~ O(1) matters for the correctness
    # test tolerance at h=1/500.
    scale_in = 1.0 / np.sqrt(D)
    scale_h = 1.0 / np.sqrt(H)
    scale_out = 0.1 / np.sqrt(H)  # small output so y stays O(1) across N steps
    return MLPWeights(
        W1=(rng.standard_normal((D, H)) * scale_in).astype(cfg.np_dtype),
        b1=(rng.standard_normal(H) * 0.01).astype(cfg.np_dtype),
        W2=(rng.standard_normal((H, H)) * scale_h).astype(cfg.np_dtype),
        b2=(rng.standard_normal(H) * 0.01).astype(cfg.np_dtype),
        W3=(rng.standard_normal((H, D)) * scale_out).astype(cfg.np_dtype),
        b3=(rng.standard_normal(D) * 0.01).astype(cfg.np_dtype),
    )


def build_y0(cfg: NodeConfig, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    return rng.standard_normal((cfg.batch, cfg.state_dim)).astype(cfg.np_dtype)


# ---------------------------------------------------------------------------
# Pure functions: vector field and one RK4 step
#
# Written so a single torch.compile call can specialise the whole RK4 body
# as one Inductor graph, producing 4 fused MLP evals + one weighted combine.
# This is the "one compiled body per STF task" contract from the plan: we
# never want to split the 4 stages into 4 separate pytorch_task calls.
# ---------------------------------------------------------------------------


def _f_theta(y, W1, b1, W2, b2, W3, b3):
    """3-layer autonomous MLP vector field: dy/dt = f_theta(y)."""
    h1 = torch.tanh(torch.addmm(b1, y, W1))
    h2 = torch.tanh(torch.addmm(b2, h1, W2))
    return torch.addmm(b3, h2, W3)


def _rk4_body(y, h_step: float, W1, b1, W2, b2, W3, b3):
    """One classical RK4 step. Returns y_next."""
    k1 = _f_theta(y, W1, b1, W2, b2, W3, b3)
    k2 = _f_theta(y + 0.5 * h_step * k1, W1, b1, W2, b2, W3, b3)
    k3 = _f_theta(y + 0.5 * h_step * k2, W1, b1, W2, b2, W3, b3)
    k4 = _f_theta(y + h_step * k3, W1, b1, W2, b2, W3, b3)
    return y + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ``mode="default"`` enables Inductor fusion but NOT the reduce-overhead
# CUDA-graph capture that would collide with STF's own graph_scope capture.
# fullgraph=True forces a single Inductor graph (no graph breaks), which is
# what we need for the body to be a clean CUDA-graph node inside ctx.repeat.
_f_compiled = torch.compile(_f_theta, mode="default", fullgraph=True)
_rk4_body_compiled = torch.compile(_rk4_body, mode="default", fullgraph=True)


# Per-shape warmup cache. torch.compile keys on input shapes; we only have
# one shape in this test, but the cache keeps the warmup idempotent across
# repeated pytest invocations.
_warmed_shapes: set[tuple[int, int, int, str]] = set()


def _warmup_compiled_bodies(cfg: NodeConfig):
    """Trigger Inductor codegen OUTSIDE any STF / CUDA-graph capture.

    Dynamo probes ``torch.cuda.get_rng_state()`` on first compile. That
    call raises "Cannot call CUDAGeneratorImpl::current_seed during CUDA
    graph capture" when first-compile happens inside ``ctx.graph_scope()``
    (where capture is active). One eager call on dummy tensors with the
    right shapes populates the compile cache so all STF replays see a
    ready-made artifact.
    """
    key = (cfg.batch, cfg.state_dim, cfg.hidden_dim, cfg.dtype)
    if key in _warmed_shapes:
        return

    device = torch.device("cuda")
    dtype = cfg.torch_dtype
    B, D, H = cfg.batch, cfg.state_dim, cfg.hidden_dim

    y = torch.zeros((B, D), dtype=dtype, device=device)
    W1 = torch.zeros((D, H), dtype=dtype, device=device)
    b1 = torch.zeros((H,), dtype=dtype, device=device)
    W2 = torch.zeros((H, H), dtype=dtype, device=device)
    b2 = torch.zeros((H,), dtype=dtype, device=device)
    W3 = torch.zeros((H, D), dtype=dtype, device=device)
    b3 = torch.zeros((D,), dtype=dtype, device=device)

    _ = _f_compiled(y, W1, b1, W2, b2, W3, b3)
    _ = _rk4_body_compiled(y, cfg.h_step, W1, b1, W2, b2, W3, b3)
    torch.cuda.synchronize()

    _warmed_shapes.add(key)


# ---------------------------------------------------------------------------
# Eager PyTorch reference integrator (the pain-point baseline)
# ---------------------------------------------------------------------------


def integrate_rk4_eager(y0: "torch.Tensor", w: MLPWeightsT, cfg: NodeConfig):
    """Plain Python for-loop over RK4 steps. No torch.compile anywhere.

    This is the pain-point baseline: every step pays Python dispatch +
    kernel-launch overhead. On this workload that overhead dominates.
    """
    y = y0.clone()
    h = cfg.h_step
    for _ in range(cfg.n_steps):
        y = _rk4_body(y, h, w.W1, w.b1, w.W2, w.b2, w.W3, w.b3)
    return y


# ---------------------------------------------------------------------------
# STF fixed-step RK4 via ctx.repeat(N)
#
# One pytorch_task per iteration, body dispatched through the compiled
# RK4 body shared with the eager reference. The whole repeat region is
# wrapped in a ctx.graph_scope() so the body is CUDA-graph-captured once
# and replayed N times (verified via CUDASTF_DOT_FILE).
# ---------------------------------------------------------------------------


def _build_stf_persistent_forward(cfg: NodeConfig, weights: MLPWeights):
    """Build STF context and logical data once; return a ``forward`` closure.

    Persistent-context timing pattern: all allocations and weight staging
    happen out of the timed path. The
    returned closure opens a fresh ``graph_scope() + repeat(N)`` each
    invocation, runs the integration, and returns without synchronising
    (the caller synchronises and times).
    """
    _warmup_compiled_bodies(cfg)

    ctx = stf.stackable_context()

    # y: host-backed so we can read it back after finalize() for the
    # correctness check. One-time H2D staging cost, paid before the
    # first timed forward.
    y_host = build_y0(cfg, seed=0)
    l_y = ctx.logical_data(y_host, name="y")

    # Weights: host-backed logical_data is fine at this size
    # (a few hundred KB total). Staged once, stays on device.
    l_W1 = ctx.logical_data(weights.W1, name="W1")
    l_b1 = ctx.logical_data(weights.b1, name="b1")
    l_W2 = ctx.logical_data(weights.W2, name="W2")
    l_b2 = ctx.logical_data(weights.b2, name="b2")
    l_W3 = ctx.logical_data(weights.W3, name="W3")
    l_b3 = ctx.logical_data(weights.b3, name="b3")
    # Weights are genuinely read-only across the whole test -- the MLP
    # parameters never get updated. Marking them read-only at the root lets
    # the stackable context auto-push them as READ into every nested scope
    # (see validate_access in stackable_ctx.cuh: push_mode = is_read_only()
    # ? read : rw), which is both simpler than pushing READ at each level
    # by hand and stronger: it also preserves the ability of sibling scopes
    # to hold concurrent read freezes at the root.
    for ld in (l_W1, l_b1, l_W2, l_b2, l_W3, l_b3):
        ld.set_read_only()

    h = cfg.h_step
    n = cfg.n_steps

    def forward():
        """One full N-step integration. Submits one graph_scope + repeat(N).

        The body inside ``with ctx.repeat(n):`` becomes a single CUDA-graph
        child-node that is replayed n times. Per-iteration host overhead
        drops from ~200 us (Python dispatch + eager kernel launch) to a
        few us of graph-replay submission cost.
        """
        with ctx.graph_scope():
            with ctx.repeat(n):
                with pytorch_task(
                    ctx,
                    l_y.rw(),
                    l_W1.read(),
                    l_b1.read(),
                    l_W2.read(),
                    l_b2.read(),
                    l_W3.read(),
                    l_b3.read(),
                ) as (tY, tW1, tb1, tW2, tb2, tW3, tb3):
                    tY.copy_(
                        _rk4_body_compiled(
                            tY,
                            h,
                            tW1,
                            tb1,
                            tW2,
                            tb2,
                            tW3,
                            tb3,
                        )
                    )

    return forward, ctx, y_host


def integrate_rk4_stf(cfg: NodeConfig, weights: MLPWeights) -> np.ndarray:
    """One-shot STF run -- used by the correctness test.

    Builds the context, runs a single forward, finalises, and returns the
    final ``y`` as a numpy array copied from the host-backed logical_data.
    """
    forward, ctx, y_host = _build_stf_persistent_forward(cfg, weights)
    torch.cuda.synchronize()
    forward()
    ctx.finalize()
    torch.cuda.synchronize()
    return y_host.copy()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _time_callable(fn, *, iters: int, warmup: int) -> float:
    """Return median wall-clock (ms) per invocation.

    Uses median rather than mean so a single cold outlier doesn't skew the
    result; relevant because torch.compile can have a staggered warmup
    even after the explicit _warmup_compiled_bodies pass.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _time_stf(forward, ctx, *, iters: int, warmup: int) -> float:
    """Specialised timer for the STF forward.

    Identical shape to ``_time_callable`` except we treat the forward +
    sync as the unit. ``ctx.finalize()`` is NOT called per-iteration
    because that would destroy the context; instead we finalise after the
    last timed iteration in the caller.
    """
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rk4_correctness():
    """STF fixed-step RK4 must match the eager PyTorch reference.

    Tolerance: 1e-4 absolute / relative. Classical RK4 at h=1/500 on this
    bounded-magnitude vector field gives ~6-7 correct digits in fp32, but
    slight reduction-order differences between Inductor's fused kernel and
    the eager pointwise ops widen the gap; 1e-4 accommodates that.
    """
    cfg = _default_cfg()
    weights = build_weights(cfg, seed=0)

    w_t = weights.as_torch(device="cuda", dtype=cfg.torch_dtype)
    y0_t = torch.as_tensor(
        build_y0(cfg, seed=0),
        device="cuda",
        dtype=cfg.torch_dtype,
    )

    y_eager = integrate_rk4_eager(y0_t, w_t, cfg).detach().cpu().numpy()
    y_stf = integrate_rk4_stf(cfg, weights)

    np.testing.assert_allclose(
        y_stf,
        y_eager,
        atol=1e-4,
        rtol=1e-4,
        err_msg=(
            "STF RK4 trajectory does not match eager reference. "
            "Likely causes: (1) compiled body and eager body diverged in "
            "reduction order, (2) non-contiguous tensor layout in the STF "
            "task view, (3) weights staged with a different dtype."
        ),
    )


def _print_timings(cfg: NodeConfig, *, iters: int, warmup: int):
    """Print eager-vs-STF wall-clock timings (informational; no assertions).

    Apples-to-apples comparison: same RK4 algorithm and step count, but the
    eager loop pays Python dispatch on every step while ``stf/repeat`` replays
    a CUDA graph captured once.
    """
    weights = build_weights(cfg, seed=0)
    w_t = weights.as_torch(device="cuda", dtype=cfg.torch_dtype)
    y0_t = torch.as_tensor(build_y0(cfg, seed=0), device="cuda", dtype=cfg.torch_dtype)

    # Trigger Inductor codegen outside any timed region or STF capture.
    _warmup_compiled_bodies(cfg)

    eager_ms = _time_callable(
        lambda: integrate_rk4_eager(y0_t, w_t, cfg), iters=iters, warmup=warmup
    )

    forward, ctx, _ = _build_stf_persistent_forward(cfg, weights)
    try:
        stf_ms = _time_stf(forward, ctx, iters=iters, warmup=warmup)
    finally:
        ctx.finalize()

    print(
        f"\n=== Neural ODE RK4 integration timings: N={cfg.n_steps}, B={cfg.batch}, "
        f"H={cfg.hidden_dim}, D={cfg.state_dim}, dtype={cfg.dtype} ==="
    )
    print(f"  {'mode':<28} {'ms / run':>12} {'speedup vs eager':>20}")
    print("  " + "-" * 62)
    for name, t in (
        ("py/eager (RK4)", eager_ms),
        ("stf/repeat (RK4)", stf_ms),
    ):
        sp = eager_ms / t if t > 0 else float("nan")
        print(f"  {name:<28} {t:>10.2f}   {sp:>18.2f}x")


def main():
    test_rk4_correctness()
    print("Correctness: PASS")
    if os.environ.get("LLM_NODE_BENCH", "0") != "0":
        cfg = _default_cfg()
        iters = int(os.environ.get("LLM_NODE_ITERS", "20"))
        warmup = int(os.environ.get("LLM_NODE_WARMUP", "5"))
        _print_timings(cfg, iters=iters, warmup=warmup)


if __name__ == "__main__":
    main()
