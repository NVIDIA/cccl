# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
STF Dopri5 as a drop-in for ``torchdiffeq.odeint`` on the canonical Neural ODE.

This example reproduces the *evaluation* forward pass of torchdiffeq's own
``examples/ode_demo.py`` (the canonical "fit a 2D spiral" Neural ODE) and shows
that a single STF entry point::

    stf_odeint(f, y0, (t0, t1), atol=..., rtol=...)  ->  y(t1)

can replace ``torchdiffeq.odeint(f, y0, [t0, t1])`` for forward integration.

(The companion ``neural_ode_rk4.py`` covers the fixed-step variant using
``ctx.graph_scope() + ctx.repeat(N)``.)

How it works
------------
The Dormand-Prince 5(4) step is written as a fixed-shape compiled body (all
accept/reject control flow expressed as ``torch.where`` masks on a scalar
signal), and the adaptive loop runs inside ``ctx.while_loop`` with a device-side
termination scalar. The whole integration is therefore one CUDA graph with a
device-driven WHILE node -- no host<->device synchronization per step to decide
whether to continue.

Scope
-----
* Forward-only drop-in: returns ``y(t1)``. No autograd/adjoint yet, and no
  dense ``t_eval`` trajectory (endpoint only) -- both are noted as followups.
* Two vector fields are exercised:
    1. ``Lambda()`` -- torchdiffeq's ground-truth dynamics ``dy/dt = y**3 @ A``.
    2. ``ODEFunc()`` -- the actual Neural ODE nn.Module from ode_demo.py.

Correctness is checked against an independent, torch-only reference that solves
the same Dopri5 body with a host-driven CUDA-graph loop
(``cudagraph_host_odeint``); both must agree on ``y(t1)``.

Run it directly::

    python neural_ode_dopri5.py

Set ``LLM_ODE_DEMO_BENCH=1`` to additionally print STF-vs-host-loop wall-clock
timings (informational only -- nothing is asserted on performance). The
integration horizon can be tuned with ``LLM_ODE_DEMO_TEND`` (default 25).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import cuda.stf._experimental as stf
from cuda.stf._experimental.interop.pytorch import pytorch_task

torch = pytest.importorskip("torch")
nn = torch.nn

# ---------------------------------------------------------------------------
# ODE models -- verbatim from torchdiffeq/examples/ode_demo.py
# ---------------------------------------------------------------------------


class Lambda(nn.Module):
    """Ground-truth dynamics from ode_demo.py: dy/dt = y**3 @ A."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "A",
            torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]),
        )

    def forward(self, t, y):
        return torch.mm(y**3, self.A)


class ODEFunc(nn.Module):
    """Same architecture as ode_demo.py's ODEFunc: (Linear-Tanh-Linear)(y**3).

    The ode_demo.py default is ``(dim, hidden) = (2, 50)`` with N(0, 0.1)
    weight init and 0 bias init. We keep that default so
    ``ODEFunc()`` is literally the torchdiffeq tutorial module, and
    parameterise the dims so the sweep test can exercise realistic
    larger shapes (latent-ODE / FFJORD regimes) without duplicating the
    class.
    """

    def __init__(self, dim: int = 2, hidden: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, y):
        return self.net(y**3)


# ---------------------------------------------------------------------------
# Dormand-Prince 5(4) tableau
# ---------------------------------------------------------------------------
#
# Coefficients duplicated (rather than imported from neural_ode_rk4) so this
# file stays standalone and can be read as a worked example.

_A21 = 1.0 / 5.0
_A31, _A32 = 3.0 / 40.0, 9.0 / 40.0
_A41, _A42, _A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_A51, _A52, _A53, _A54 = (
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
)
_A61, _A62, _A63, _A64, _A65 = (
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
)
_A71, _A73, _A74, _A75, _A76 = (
    35.0 / 384.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
)
_E1, _E3, _E4, _E5, _E6, _E7 = (
    71.0 / 57600.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
)
_C2, _C3, _C4, _C5, _C6, _C7 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0


# The Dopri5 step body *itself* is generic; we just need a fresh k_fn(t, y)
# per vector field. To sidestep torch.compile's guards on nn.Module state
# (which fire when the body is re-entered inside a CUDA graph capture and
# try to re-take an RNG snapshot), we specialize the compiled body per
# vector-field family and pass all parameters explicitly. This mirrors the
# pattern already validated in ``neural_ode_rk4.py``.


def _dopri5_step(y, t, h, t_end, atol, rtol, k_fn):
    """Core Dopri5 update -- called from the specialized compiled bodies.

    Never call this directly from user code: it's only correct when
    embedded in a module-level ``@torch.compile``-ed wrapper whose inputs
    are pure tensors (so Dynamo's guards stay stable across calls).
    """
    k1 = k_fn(t, y)
    k2 = k_fn(t + _C2 * h, y + h * (_A21 * k1))
    k3 = k_fn(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2))
    k4 = k_fn(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
    k5 = k_fn(
        t + _C5 * h,
        y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4),
    )
    k6 = k_fn(
        t + _C6 * h,
        y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5),
    )
    y_prop = y + h * (_A71 * k1 + _A73 * k3 + _A74 * k4 + _A75 * k5 + _A76 * k6)
    k7 = k_fn(t + _C7 * h, y_prop)

    err = h * (_E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
    sc = atol + rtol * torch.maximum(y.abs(), y_prop.abs())
    err_norm = ((err / sc) ** 2).mean().sqrt()

    accept = err_norm <= 1.0
    safety = 0.9
    factor = (safety * (1.0 / err_norm.clamp(min=1e-20)).clamp_max(10.0) ** 0.2).clamp(
        0.2, 10.0
    )

    y_new = torch.where(accept, y_prop, y)
    t_new = torch.where(accept, t + h, t)
    h_next = torch.minimum(h * factor, t_end - t_new).clamp(min=1e-20)
    cond = (t_new < t_end).to(y.dtype)
    return y_new, t_new, h_next, cond


# ---- specialization: Lambda (y**3 @ A) ------------------------------------


def _lambda_body(y, t, h, t_end, atol, rtol, A):
    def k_fn(t_, y_):  # noqa: ARG001 (t unused, autonomous)
        return torch.mm(y_**3, A)

    return _dopri5_step(y, t, h, t_end, atol, rtol, k_fn)


_lambda_body_compiled = torch.compile(_lambda_body, mode="default", fullgraph=True)


# ---- specialization: ODEFunc (Linear(2,50) -> Tanh -> Linear(50,2) on y**3) ----


def _odefunc_body(y, t, h, t_end, atol, rtol, W1, b1, W2, b2):
    def k_fn(t_, y_):  # noqa: ARG001
        y3 = y_**3
        h1 = torch.tanh(y3 @ W1.t() + b1)
        return h1 @ W2.t() + b2

    return _dopri5_step(y, t, h, t_end, atol, rtol, k_fn)


_odefunc_body_compiled = torch.compile(_odefunc_body, mode="default", fullgraph=True)


def _extract_model_params(f: nn.Module):
    """Return ``(compiled_body, [param tensors])`` for a supported model.

    Supported models: ``Lambda`` (param list = [A]) and ``ODEFunc``
    (param list = [W1, b1, W2, b2]). Extend this helper to plug other
    architectures in.
    """
    if isinstance(f, Lambda):
        return _lambda_body_compiled, [f.A]
    if isinstance(f, ODEFunc):
        lin0, _, lin1 = f.net[0], f.net[1], f.net[2]
        return _odefunc_body_compiled, [lin0.weight, lin0.bias, lin1.weight, lin1.bias]
    raise TypeError(
        f"stf_odeint does not yet know how to bind vector field of type "
        f"{type(f).__name__}. Extend _extract_model_params to add support."
    )


def _warmup_body(body_compiled, params, y0, dtype, device, t_end_val):
    """Pre-compile the body outside any STF / CUDA graph capture.

    Required to avoid Dynamo's first-call RNG snapshot firing during
    capture and raising "Cannot call CUDAGeneratorImpl::current_seed
    during CUDA graph capture".

    IMPORTANT: The warmup call must match the *Dynamo guard signature* of
    the production call, not just shapes/dtypes. In particular, ``nn.Parameter``
    has ``requires_grad=True`` while the tensors STF hands back through
    ``pytorch_task`` are ``requires_grad=False`` views. Passing the
    Parameters directly here would compile a cache entry Dynamo can't
    reuse on the captured path, and it would then try to recompile
    mid-capture. We explicitly detach everything so warmup and production
    look identical to Dynamo.
    """
    detached_params = [p.detach().clone() for p in params]
    y0_det = y0.detach().clone()
    t_scalar = torch.zeros((), device=device, dtype=dtype)
    h_scalar = torch.full((), 0.1, device=device, dtype=dtype)
    t_end_scalar = torch.full((), t_end_val, device=device, dtype=dtype)
    _ = body_compiled(
        y0_det,
        t_scalar,
        h_scalar,
        t_end_scalar,
        1e-6,
        1e-6,
        *detached_params,
    )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# STF drop-in solver
# ---------------------------------------------------------------------------


def _build_stf_odeint_persistent(
    f: nn.Module,
    y0: torch.Tensor,
    t_span,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    """Persistent-context form of stf_odeint.

    Returns ``(forward, ctx, y_host)`` where ``forward()`` runs one full
    integration from ``t_span[0]`` to ``t_span[1]`` into the host-backed
    logical_data ``y_host``. The context + compiled body are shared
    across calls.

    Use this when you call the solver many times (e.g. inside a rollout
    loop); the one-shot ``stf_odeint`` simply wraps this.
    """
    t0_f, t1_f = float(t_span[0]), float(t_span[1])
    device = y0.device
    dtype = y0.dtype
    assert y0.ndim == 2, "y0 must be (B, D)"

    body_compiled, params = _extract_model_params(f)
    # Pre-compile the body once OUTSIDE any STF capture. Dynamo's first-
    # call RNG probe blows up inside CUDA graph capture; this forces that
    # probe to happen now.
    _warmup_body(body_compiled, params, y0, dtype, device, t1_f)

    ctx = stf.stackable_context()

    np_dtype = np.dtype({torch.float32: "float32", torch.float64: "float64"}[dtype])

    # y is host-backed so the caller can read the final state back after
    # ctx.finalize().
    y_host = y0.detach().cpu().numpy().astype(np_dtype).copy()
    l_y = ctx.logical_data(y_host, name="y")
    l_t = ctx.logical_data_empty((1,), np_dtype, name="t")
    l_h = ctx.logical_data_empty((1,), np_dtype, name="h")
    l_cond = ctx.logical_data_empty((1,), np_dtype, name="cond")

    # Parameters are read-only for the lifetime of the solver -- mark them
    # as such so the stackable_ctx auto-pushes READ at every nesting level
    # instead of RW (see the same treatment in neural_ode_rk4.py).
    # logical_data takes host-backed numpy arrays; we own a copy so the
    # live nn.Module weights can continue to train without aliasing this
    # solver's frozen view of them.
    param_np = [p.detach().cpu().numpy().astype(np_dtype).copy() for p in params]
    l_params = [ctx.logical_data(p, name=f"p{i}") for i, p in enumerate(param_np)]
    for lp in l_params:
        lp.set_read_only()

    y0_cuda = y0.detach().to(device=device, dtype=dtype).clone()
    t_end_cuda = torch.full((), t1_f, device=device, dtype=dtype)
    h_init = (t1_f - t0_f) / 100.0

    def forward():
        # Reset (y, t, h) at the top of every forward so repeated calls
        # start from the same IC.
        with pytorch_task(ctx, l_y.write(), l_t.write(), l_h.write()) as (
            tY,
            tT,
            tH,
        ):
            tY.copy_(y0_cuda)
            tT.fill_(t0_f)
            tH.fill_(h_init)

        with ctx.while_loop() as loop:
            with pytorch_task(
                ctx,
                l_y.rw(),
                l_t.rw(),
                l_h.rw(),
                l_cond.write(),
                *[lp.read() for lp in l_params],
            ) as tensors:
                tY, tT, tH, tC = tensors[:4]
                param_tensors = tensors[4:]
                t0d = tT.squeeze()
                h0d = tH.squeeze()
                y_new, t_new, h_new, cond = body_compiled(
                    tY,
                    t0d,
                    h0d,
                    t_end_cuda,
                    atol,
                    rtol,
                    *param_tensors,
                )
                tY.copy_(y_new)
                tT.copy_(t_new.unsqueeze(0))
                tH.copy_(h_new.unsqueeze(0))
                tC.copy_(cond.unsqueeze(0))
            loop.continue_while(l_cond, ">", 0.5)

    return forward, ctx, y_host


def stf_odeint(
    f: nn.Module,
    y0: torch.Tensor,
    t_span,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """Minimal drop-in replacement for ``torchdiffeq.odeint(f, y0, [t0,t1])``.

    * ``f`` is a callable ``(t, y) -> dy/dt``; typically an ``nn.Module``.
    * ``y0`` is a (B, D) CUDA tensor.
    * ``t_span`` is ``(t0, t1)``; the solver integrates to ``t1`` and
      returns ``y(t1)``. Dense output is not supported yet.

    This is the one-shot form: it builds a fresh stackable_context per
    call, so per-call overhead is higher than the persistent form. Use
    ``_build_stf_odeint_persistent`` when calling in a loop.
    """
    forward, ctx, y_host = _build_stf_odeint_persistent(
        f,
        y0,
        t_span,
        atol=atol,
        rtol=rtol,
    )
    forward()
    ctx.finalize()
    torch.cuda.synchronize()
    return torch.as_tensor(y_host, device=y0.device, dtype=y0.dtype).clone()


# ---------------------------------------------------------------------------
# Manual CUDA-graph baseline  (NO STF)
# ---------------------------------------------------------------------------
#
# This is the honest "what would a motivated PyTorch user write without STF?"
# implementation. It reuses exactly the same compiled Dopri5 body, captures
# ONE step into a ``torch.cuda.CUDAGraph``, and drives the adaptive loop
# from the HOST by replaying the graph, reading the cond tensor with
# ``cond.item()``, and breaking when the integration finishes.
#
# Compared to the STF version it trades:
#   * no more per-iteration Python dispatch / tensor-metadata cost
#     (the graph replay is a single ``cudaGraphLaunch`` call),
#   * against ONE host<->device synchronization per iteration to read the
#     termination flag (``cond.item()`` implies a D2H copy + sync).
#
# That sync-per-iteration is exactly what STF's ``ctx.while_loop`` eliminates
# by connecting the cond tensor to a CUDA conditional-graph WHILE node that
# runs entirely on the device. So the gap between this baseline and the STF
# version quantifies the value of device-side loop control specifically,
# after factoring out the kernel-fusion / graph-launch-amortization wins.


def _build_cudagraph_host_odeint_persistent(
    f: nn.Module,
    y0: torch.Tensor,
    t_span,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 10_000,
):
    """Persistent CUDA-graph + host-driven termination solver.

    Returns ``(forward, y_out)`` where ``forward()`` integrates from
    ``t_span[0]`` to ``t_span[1]`` and leaves the result in the device
    tensor ``y_out`` (which is aliased to the capture's ``y`` buffer).
    """
    t0_f, t1_f = float(t_span[0]), float(t_span[1])
    device = y0.device
    dtype = y0.dtype
    assert y0.ndim == 2, "y0 must be (B, D)"

    body_compiled, params = _extract_model_params(f)
    # Same warmup as the STF path: compile Inductor artifacts now so no
    # compile fires inside the capture.
    _warmup_body(body_compiled, params, y0, dtype, device, t1_f)

    # Persistent device buffers that the captured graph reads/writes.
    y_buf = y0.detach().clone()
    t_buf = torch.zeros((), device=device, dtype=dtype)
    h_buf = torch.zeros((), device=device, dtype=dtype)
    cond_buf = torch.zeros((), device=device, dtype=dtype)
    t_end_buf = torch.full((), t1_f, device=device, dtype=dtype)
    param_bufs = [p.detach().clone() for p in params]

    y0_cuda = y0.detach().clone()
    h_init = (t1_f - t0_f) / 100.0

    # Prime the buffers so the first capture call sees realistic values
    # (Dynamo guard stability depends on matching attributes, which we
    # already ensured in _warmup_body).
    y_buf.copy_(y0_cuda)
    t_buf.fill_(t0_f)
    h_buf.fill_(h_init)

    # Do one un-captured body call on the exact buffers to force any last
    # lazy init (allocator pool warmup, kernel JIT, etc.) before capture.
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = body_compiled(
            y_buf,
            t_buf,
            h_buf,
            t_end_buf,
            atol,
            rtol,
            *param_bufs,
        )
    torch.cuda.synchronize()

    # Capture one Dopri5 step into a CUDAGraph. Outputs are copied back
    # into the persistent buffers so the next replay reads updated state.
    graph = torch.cuda.CUDAGraph()
    # Re-prime for the captured invocation.
    y_buf.copy_(y0_cuda)
    t_buf.fill_(t0_f)
    h_buf.fill_(h_init)

    # A dedicated stream for capture -- required by torch.cuda.graph.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph):
            y_new, t_new, h_new, cond = body_compiled(
                y_buf,
                t_buf,
                h_buf,
                t_end_buf,
                atol,
                rtol,
                *param_bufs,
            )
            y_buf.copy_(y_new)
            t_buf.copy_(t_new)
            h_buf.copy_(h_new)
            cond_buf.copy_(cond)
    torch.cuda.current_stream().wait_stream(s)

    def forward():
        # Reset initial state on the host (cheap -- small tensors).
        y_buf.copy_(y0_cuda)
        t_buf.fill_(t0_f)
        h_buf.fill_(h_init)
        cond_buf.fill_(1.0)

        # Host-driven adaptive loop. Each iteration pays one cond.item()
        # which forces a D2H copy + sync. That's the cost we're measuring.
        for _ in range(max_iters):
            graph.replay()
            if cond_buf.item() < 0.5:
                break
        else:
            raise RuntimeError(
                f"manual CUDA-graph solver did not converge in {max_iters} iterations"
            )

    return forward, y_buf


def cudagraph_host_odeint(
    f: nn.Module,
    y0: torch.Tensor,
    t_span,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> torch.Tensor:
    """One-shot wrapper around ``_build_cudagraph_host_odeint_persistent``."""
    forward, y_buf = _build_cudagraph_host_odeint_persistent(
        f,
        y0,
        t_span,
        atol=atol,
        rtol=rtol,
    )
    forward()
    torch.cuda.synchronize()
    return y_buf.clone()


# ---------------------------------------------------------------------------
# Correctness test -- the core deliverable of this file
# ---------------------------------------------------------------------------


def _ode_demo_cfg():
    """Return the ode_demo.py-style integration configuration."""
    return {
        "y0": torch.tensor([[2.0, 0.0]], device="cuda", dtype=torch.float32),
        "t_span": (0.0, float(os.environ.get("LLM_ODE_DEMO_TEND", "25"))),
        "atol": 1e-6,
        "rtol": 1e-6,
    }


def _assert_endpoints_match(label: str, *ys, atol=1e-4, rtol=1e-4):
    """Pairwise compare a bunch of endpoint tensors as numpy arrays."""
    ys_np = [y.detach().cpu().numpy() for y in ys]
    for i in range(1, len(ys_np)):
        np.testing.assert_allclose(
            ys_np[0],
            ys_np[i],
            atol=atol,
            rtol=rtol,
            err_msg=f"[{label}] solver #{i} disagrees with solver #0",
        )


def test_dopri5_correctness_lambda():
    """Ground-truth dynamics: the STF drop-in must agree on ``y(t_end)``.

    The reference is ``cudagraph_host_odeint``: an independent, torch-only
    solver that runs the *same* Dopri5 body but drives the adaptive loop from
    the host (replay a captured graph + read the termination flag with
    ``cond.item()``). Same math, different control-loop driver, so agreement
    isolates the STF ``ctx.while_loop`` plumbing.
    """
    cfg = _ode_demo_cfg()
    f = Lambda().cuda()

    y_stf = stf_odeint(f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"])
    y_ref = cudagraph_host_odeint(
        f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"]
    )

    _assert_endpoints_match("Lambda", y_ref, y_stf)


def test_dopri5_correctness_odefunc():
    """Actual Neural ODE nn.Module -- same drop-in, same agreement."""
    cfg = _ode_demo_cfg()
    torch.manual_seed(0xC0DE)
    f = ODEFunc().cuda()

    y_stf = stf_odeint(f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"])
    y_ref = cudagraph_host_odeint(
        f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"]
    )

    _assert_endpoints_match("ODEFunc", y_ref, y_stf)


# ---------------------------------------------------------------------------
# Optional benchmark -- run with LLM_ODE_DEMO_BENCH=1
# ---------------------------------------------------------------------------


def _time_callable(fn, *, iters: int, warmup: int) -> float:
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


def _time_stf_forward(forward, *, iters: int, warmup: int) -> float:
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


def _print_timings(cfg, *, iters: int, warmup: int):
    """Print STF-vs-host-loop wall-clock timings (informational; no assertions).

    Both solvers run the same compiled Dopri5 body. The ``cuda-graph + host
    loop`` baseline replays a captured graph but reads the termination flag
    with ``cond.item()`` -- a host<->device sync per step -- while STF's
    ``ctx.while_loop`` keeps the loop control on the device. The gap is what
    device-side control flow buys.
    """
    torch.manual_seed(0xC0DE)
    f = ODEFunc().cuda()

    forward_cg, _ = _build_cudagraph_host_odeint_persistent(
        f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"]
    )
    t_cg = _time_callable(forward_cg, iters=iters, warmup=warmup)

    forward, ctx, _ = _build_stf_odeint_persistent(
        f, cfg["y0"], cfg["t_span"], atol=cfg["atol"], rtol=cfg["rtol"]
    )
    try:
        t_stf = _time_stf_forward(forward, iters=iters, warmup=warmup)
    finally:
        ctx.finalize()

    print(
        f"\n=== ode_demo.py-style eval: y0={cfg['y0'].tolist()}, "
        f"t_span={cfg['t_span']}, atol={cfg['atol']}, rtol={cfg['rtol']} ==="
    )
    print(f"  {'solver':<34} {'ms / run':>12} {'speedup vs host loop':>22}")
    print("  " + "-" * 70)
    for name, t in (
        ("cuda-graph + host loop (no STF)", t_cg),
        ("stf/while_loop (drop-in)", t_stf),
    ):
        sp = t_cg / t if t > 0 else float("nan")
        print(f"  {name:<34} {t:>10.2f}   {sp:>20.2f}x")


def main():
    test_dopri5_correctness_lambda()
    print("Dopri5 Lambda correctness: PASS")
    test_dopri5_correctness_odefunc()
    print("Dopri5 ODEFunc correctness: PASS")
    if os.environ.get("LLM_ODE_DEMO_BENCH", "0") != "0":
        cfg = _ode_demo_cfg()
        iters = int(os.environ.get("LLM_ODE_DEMO_ITERS", "30"))
        warmup = int(os.environ.get("LLM_ODE_DEMO_WARMUP", "5"))
        _print_timings(cfg, iters=iters, warmup=warmup)


if __name__ == "__main__":
    main()
