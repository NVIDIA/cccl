# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Drop-in STF Dopri5 on torchdiffeq's ``ode_demo.py`` setup.

Context
-------
``test_node_stf.py`` validates the underlying performance claim on a
hand-written MLP vector field. This file shows that the same mechanism
(compiled Dopri5 body + ``ctx.while_loop`` + device-side termination) can
serve as a drop-in replacement for ``torchdiffeq.odeint`` on the exact
Neural-ODE ``ODEFunc`` used by torchdiffeq's own
``examples/ode_demo.py`` -- the canonical "fit a 2D spiral" demo.

Scope
-----
* Forward-only drop-in: ``stf_odeint(f, y0, (t0, t1), atol, rtol)`` returns
  ``y(t1)``. No autograd/adjoint yet -- the training path of ode_demo.py
  still needs torchdiffeq. The *evaluation* path (``with torch.no_grad():
  odeint(func, true_y0, t)``) is what this file replaces.
* Endpoint-only output; no dense ``t_eval`` trajectory. Extending to dense
  output requires recording accepted snapshots inside the while_loop body
  and interpolating -- noted as a followup.
* Two vector fields are tested for robustness:
    1. ``Lambda()`` -- torchdiffeq's canonical ground-truth dynamics
       ``dy/dt = y**3 @ A`` (no learned parameters).
    2. ``ODEFunc()`` -- the actual Neural ODE architecture from
       ``ode_demo.py``, with the default std=0.1 random init. Random init
       is fine here because we only check solver *agreement*, not the
       quality of the fit.

Toggles
-------
    LLM_ODE_DEMO_BENCH=1   run the benchmark (default off).
    LLM_ODE_DEMO_TEND=25   integration horizon (default 25, matches ode_demo).
    LLM_ODE_DEMO_ITERS=30  timed iterations.
    LLM_ODE_DEMO_WARMUP=5  warmup iterations.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402


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
        return torch.mm(y ** 3, self.A)


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
        return self.net(y ** 3)


# ---------------------------------------------------------------------------
# Dormand-Prince 5(4) tableau
# ---------------------------------------------------------------------------
#
# Coefficients duplicated (rather than imported from test_node_stf) so this
# file stays standalone and can be read as a worked example.

_A21 = 1.0 / 5.0
_A31, _A32 = 3.0 / 40.0, 9.0 / 40.0
_A41, _A42, _A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_A51, _A52, _A53, _A54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
_A61, _A62, _A63, _A64, _A65 = (
    9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0,
)
_A71, _A73, _A74, _A75, _A76 = (
    35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0,
)
_E1, _E3, _E4, _E5, _E6, _E7 = (
    71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0,
    -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0,
)
_C2, _C3, _C4, _C5, _C6, _C7 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0


# The Dopri5 step body *itself* is generic; we just need a fresh k_fn(t, y)
# per vector field. To sidestep torch.compile's guards on nn.Module state
# (which fire when the body is re-entered inside a CUDA graph capture and
# try to re-take an RNG snapshot), we specialize the compiled body per
# vector-field family and pass all parameters explicitly. This mirrors the
# pattern already validated in ``test_node_stf.py``.


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
    y_prop = y + h * (
        _A71 * k1 + _A73 * k3 + _A74 * k4 + _A75 * k5 + _A76 * k6
    )
    k7 = k_fn(t + _C7 * h, y_prop)

    err = h * (
        _E1 * k1 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7
    )
    sc = atol + rtol * torch.maximum(y.abs(), y_prop.abs())
    err_norm = ((err / sc) ** 2).mean().sqrt()

    accept = err_norm <= 1.0
    safety = 0.9
    factor = (safety * (1.0 / err_norm.clamp(min=1e-20))
              .clamp_max(10.0) ** 0.2).clamp(0.2, 10.0)

    y_new = torch.where(accept, y_prop, y)
    t_new = torch.where(accept, t + h, t)
    h_next = torch.minimum(h * factor, t_end - t_new).clamp(min=1e-20)
    cond = (t_new < t_end).to(y.dtype)
    return y_new, t_new, h_next, cond


# ---- specialization: Lambda (y**3 @ A) ------------------------------------


def _lambda_body(y, t, h, t_end, atol, rtol, A):
    def k_fn(t_, y_):  # noqa: ARG001 (t unused, autonomous)
        return torch.mm(y_ ** 3, A)
    return _dopri5_step(y, t, h, t_end, atol, rtol, k_fn)


_lambda_body_compiled = torch.compile(_lambda_body, mode="default", fullgraph=True)


# ---- specialization: ODEFunc (Linear(2,50) -> Tanh -> Linear(50,2) on y**3) ----


def _odefunc_body(y, t, h, t_end, atol, rtol, W1, b1, W2, b2):
    def k_fn(t_, y_):  # noqa: ARG001
        y3 = y_ ** 3
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
        return _odefunc_body_compiled, [lin0.weight, lin0.bias,
                                        lin1.weight, lin1.bias]
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
        y0_det, t_scalar, h_scalar, t_end_scalar, 1e-6, 1e-6, *detached_params,
    )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# STF drop-in solver
# ---------------------------------------------------------------------------


def _build_stf_odeint_persistent(
    f: nn.Module, y0: torch.Tensor, t_span,
    *, atol: float = 1e-6, rtol: float = 1e-6,
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

    np_dtype = np.dtype(
        {torch.float32: "float32", torch.float64: "float64"}[dtype]
    )

    # y is host-backed so the caller can read the final state back after
    # ctx.finalize().
    y_host = y0.detach().cpu().numpy().astype(np_dtype).copy()
    l_y = ctx.logical_data(y_host, name="y")
    l_t = ctx.logical_data_empty((1,), np_dtype, name="t")
    l_h = ctx.logical_data_empty((1,), np_dtype, name="h")
    l_cond = ctx.logical_data_empty((1,), np_dtype, name="cond")

    # Parameters are read-only for the lifetime of the solver -- mark them
    # as such so the stackable_ctx auto-pushes READ at every nesting level
    # instead of RW (see the same treatment in test_node_stf.py).
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
            tY, tT, tH,
        ):
            tY.copy_(y0_cuda)
            tT.fill_(t0_f)
            tH.fill_(h_init)

        with ctx.while_loop() as loop:
            with pytorch_task(
                ctx,
                l_y.rw(), l_t.rw(), l_h.rw(), l_cond.write(),
                *[lp.read() for lp in l_params],
            ) as tensors:
                tY, tT, tH, tC = tensors[:4]
                param_tensors = tensors[4:]
                t0d = tT.squeeze()
                h0d = tH.squeeze()
                y_new, t_new, h_new, cond = body_compiled(
                    tY, t0d, h0d, t_end_cuda, atol, rtol, *param_tensors,
                )
                tY.copy_(y_new)
                tT.copy_(t_new.unsqueeze(0))
                tH.copy_(h_new.unsqueeze(0))
                tC.copy_(cond.unsqueeze(0))
            loop.continue_while(l_cond, ">", 0.5)

    return forward, ctx, y_host


def stf_odeint(
    f: nn.Module, y0: torch.Tensor, t_span,
    *, atol: float = 1e-6, rtol: float = 1e-6,
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
        f, y0, t_span, atol=atol, rtol=rtol,
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
    f: nn.Module, y0: torch.Tensor, t_span,
    *, atol: float = 1e-6, rtol: float = 1e-6, max_iters: int = 10_000,
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
            y_buf, t_buf, h_buf, t_end_buf, atol, rtol, *param_bufs,
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
                y_buf, t_buf, h_buf, t_end_buf, atol, rtol, *param_bufs,
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
                f"manual CUDA-graph solver did not converge in "
                f"{max_iters} iterations"
            )

    return forward, y_buf


def cudagraph_host_odeint(
    f: nn.Module, y0: torch.Tensor, t_span,
    *, atol: float = 1e-6, rtol: float = 1e-6,
) -> torch.Tensor:
    """One-shot wrapper around ``_build_cudagraph_host_odeint_persistent``."""
    forward, y_buf = _build_cudagraph_host_odeint_persistent(
        f, y0, t_span, atol=atol, rtol=rtol,
    )
    forward()
    torch.cuda.synchronize()
    return y_buf.clone()


# ---------------------------------------------------------------------------
# Baseline adapters (torchdiffeq / torchode)
# ---------------------------------------------------------------------------


def _torchdiffeq_odeint(f, y0, t_span, *, atol=1e-6, rtol=1e-6):
    from torchdiffeq import odeint
    t = torch.tensor(
        [float(t_span[0]), float(t_span[1])],
        device=y0.device, dtype=y0.dtype,
    )
    return odeint(f, y0, t, method="dopri5", atol=atol, rtol=rtol)[-1]


def _torchode_odeint(f, y0, t_span, *, atol=1e-6, rtol=1e-6):
    import torchode as to
    # torchode expects f(t, y); our nn.Modules already have that signature.
    term = to.ODETerm(f, with_stats=False)
    method = to.Dopri5(term=term)
    controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
    solver = to.AutoDiffAdjoint(method, controller)
    B = y0.shape[0]
    t_start = torch.full((B,), float(t_span[0]),
                         device=y0.device, dtype=y0.dtype)
    t_end = torch.full((B,), float(t_span[1]),
                       device=y0.device, dtype=y0.dtype)
    problem = to.InitialValueProblem(y0=y0, t_start=t_start, t_end=t_end)
    return solver.solve(problem).ys[:, -1]


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
            ys_np[0], ys_np[i], atol=atol, rtol=rtol,
            err_msg=f"[{label}] solver #{i} disagrees with solver #0",
        )


def test_ode_demo_correctness_lambda():
    """Ground-truth dynamics: all three solvers must agree on ``y(t_end)``."""
    cfg = _ode_demo_cfg()
    f = Lambda().cuda()

    y_td = _torchdiffeq_odeint(f, cfg["y0"], cfg["t_span"],
                               atol=cfg["atol"], rtol=cfg["rtol"])
    try:
        y_to = _torchode_odeint(f, cfg["y0"], cfg["t_span"],
                                atol=cfg["atol"], rtol=cfg["rtol"])
    except ImportError:
        y_to = None

    y_stf = stf_odeint(f, cfg["y0"], cfg["t_span"],
                      atol=cfg["atol"], rtol=cfg["rtol"])
    y_cg = cudagraph_host_odeint(f, cfg["y0"], cfg["t_span"],
                                 atol=cfg["atol"], rtol=cfg["rtol"])

    ys = [y_td, y_stf, y_cg] + ([y_to] if y_to is not None else [])
    _assert_endpoints_match("Lambda", *ys)


def test_ode_demo_correctness_odefunc():
    """Actual Neural ODE nn.Module -- same drop-in, same agreement."""
    cfg = _ode_demo_cfg()
    torch.manual_seed(0xC0DE)
    f = ODEFunc().cuda()

    y_td = _torchdiffeq_odeint(f, cfg["y0"], cfg["t_span"],
                               atol=cfg["atol"], rtol=cfg["rtol"])
    try:
        y_to = _torchode_odeint(f, cfg["y0"], cfg["t_span"],
                                atol=cfg["atol"], rtol=cfg["rtol"])
    except ImportError:
        y_to = None

    y_stf = stf_odeint(f, cfg["y0"], cfg["t_span"],
                      atol=cfg["atol"], rtol=cfg["rtol"])
    y_cg = cudagraph_host_odeint(f, cfg["y0"], cfg["t_span"],
                                 atol=cfg["atol"], rtol=cfg["rtol"])

    ys = [y_td, y_stf, y_cg] + ([y_to] if y_to is not None else [])
    _assert_endpoints_match("ODEFunc", *ys)


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


@pytest.mark.skipif(
    os.environ.get("LLM_ODE_DEMO_BENCH", "0") == "0",
    reason="Set LLM_ODE_DEMO_BENCH=1 to run the ode_demo benchmark.",
)
def test_ode_demo_benchmark():
    """Same workload as ode_demo.py's eval-time forward call, three solvers."""
    cfg = _ode_demo_cfg()
    iters = int(os.environ.get("LLM_ODE_DEMO_ITERS", "30"))
    warmup = int(os.environ.get("LLM_ODE_DEMO_WARMUP", "5"))

    torch.manual_seed(0xC0DE)
    f = ODEFunc().cuda()

    # torchdiffeq.odeint: closure over f since it's stateless across calls.
    t_td = _time_callable(
        lambda: _torchdiffeq_odeint(
            f, cfg["y0"], cfg["t_span"],
            atol=cfg["atol"], rtol=cfg["rtol"],
        ),
        iters=iters, warmup=warmup,
    )

    # torchode: build the solver once (torchode has per-call Python setup
    # that we don't want to amortise into every timed iteration).
    try:
        import torchode as to
        term = to.ODETerm(f, with_stats=False)
        method = to.Dopri5(term=term)
        controller = to.IntegralController(
            atol=cfg["atol"], rtol=cfg["rtol"], term=term,
        )
        solver_obj = to.AutoDiffAdjoint(method, controller)
        try:
            solver_obj = torch.compile(
                solver_obj, mode="reduce-overhead", fullgraph=False,
            )
        except Exception:  # noqa: BLE001
            pass

        B = cfg["y0"].shape[0]
        t_start = torch.full(
            (B,), float(cfg["t_span"][0]),
            device=cfg["y0"].device, dtype=cfg["y0"].dtype,
        )
        t_end = torch.full(
            (B,), float(cfg["t_span"][1]),
            device=cfg["y0"].device, dtype=cfg["y0"].dtype,
        )

        def torchode_call():
            problem = to.InitialValueProblem(
                y0=cfg["y0"], t_start=t_start, t_end=t_end,
            )
            return solver_obj.solve(problem).ys[:, -1]

        t_to = _time_callable(torchode_call, iters=iters, warmup=warmup)
    except ImportError:
        t_to = float("nan")

    # Manual CUDAGraph + host-driven termination (NO STF). Same compiled
    # body, same Dopri5 math, just a different outer loop.
    forward_cg, _ = _build_cudagraph_host_odeint_persistent(
        f, cfg["y0"], cfg["t_span"],
        atol=cfg["atol"], rtol=cfg["rtol"],
    )
    t_cg = _time_callable(forward_cg, iters=iters, warmup=warmup)

    # STF: persistent context.
    forward, ctx, _ = _build_stf_odeint_persistent(
        f, cfg["y0"], cfg["t_span"],
        atol=cfg["atol"], rtol=cfg["rtol"],
    )
    try:
        t_stf = _time_stf_forward(forward, iters=iters, warmup=warmup)
    finally:
        ctx.finalize()

    # Report.
    print(
        f"\n=== ode_demo.py-style eval: y0={cfg['y0'].tolist()}, "
        f"t_span={cfg['t_span']}, atol={cfg['atol']}, rtol={cfg['rtol']} ==="
    )
    print(f"  {'solver':<32} {'ms / run':>12} {'speedup vs torchdiffeq':>26}")
    print("  " + "-" * 72)
    for name, t in (
        ("torchdiffeq/dopri5", t_td),
        ("torchode/dopri5", t_to),
        ("cuda-graph + host loop (manual)", t_cg),
        ("stf/dopri5 (drop-in)", t_stf),
    ):
        if t != t:  # NaN
            print(f"  {name:<32} {'(skipped)':>12} {'-':>26}")
        else:
            sp = t_td / t if t > 0 else float("nan")
            print(f"  {name:<32} {t:>10.2f}   {sp:>24.2f}x")

    # Hard gate: STF must beat torchdiffeq (same algorithm, device-side vs
    # host-side control loop). If this ever fails, something regressed.
    assert t_stf == t_stf and t_td / t_stf >= 2.0, (
        f"stf/dopri5 is not >=2x faster than torchdiffeq/dopri5 "
        f"on the ode_demo workload (stf={t_stf:.2f} ms, td={t_td:.2f} ms). "
        f"Expected a comfortable margin since the algorithmic work is "
        f"identical and only the control-loop driver differs."
    )


# ---------------------------------------------------------------------------
# Size sweep -- where does the STF advantage plateau?
# ---------------------------------------------------------------------------
#
# ode_demo.py is a tutorial with a 2-wide state and a 50-wide hidden. That
# size is overhead-bound, which is exactly where STF's device-side control
# loop wins. Real Neural ODEs are bigger, and Python-loop overhead becomes
# a smaller fraction of total cost. This sweep measures the crossover.
#
# Configurations are picked to span the realistic regimes discussed in
# torchode/FFJORD literature:
#
#   toy           (B= 1, D=  2, H=  50)  ode_demo.py itself
#   small         (B= 1, D= 16, H= 128)  minimal "non-toy" Neural ODE
#   medium        (B=32, D= 64, H= 256)  latent-ODE time-series regime
#   medium-large  (B=64, D=128, H= 256)
#   large         (B=32, D=256, H= 512)  FFJORD-like per-step cost
#
# All configs use t in [0, 5] (vs. 25 for ode_demo) to keep total runtime
# reasonable across the sweep; step count still scales with the stiffness
# of each random-init vector field, so the #steps each solver actually
# takes will vary across rows.


_SWEEP_CONFIGS = [
    # (B, D, H, label)
    (1,    2,   50, "toy (ode_demo)"),
    (1,   16,  128, "small"),
    (32,  64,  256, "medium (latent-ODE)"),
    (64, 128,  256, "medium-large"),
    (32, 256,  512, "large (FFJORD-ish)"),
]


@pytest.mark.skipif(
    os.environ.get("LLM_ODE_DEMO_SWEEP", "0") == "0",
    reason="Set LLM_ODE_DEMO_SWEEP=1 to run the problem-size sweep.",
)
def test_ode_demo_sweep():
    """Sweep problem size from ode_demo.py's toy to FFJORD-ish per-step cost.

    Reports ``ms/run`` and speedup vs torchdiffeq for each config. Not a
    hard gate: we *expect* the STF advantage to shrink as the matmul
    work starts to dominate Python-loop overhead; the point is to
    quantify where the crossover is.
    """
    iters = int(os.environ.get("LLM_ODE_DEMO_ITERS", "10"))
    warmup = int(os.environ.get("LLM_ODE_DEMO_WARMUP", "3"))
    t_span = (0.0, float(os.environ.get("LLM_ODE_DEMO_SWEEP_TEND", "5")))
    atol = rtol = 1e-6

    rows = []

    for B, D, H, label in _SWEEP_CONFIGS:
        torch.manual_seed(0xC0DE + B * 131 + D * 17 + H)
        f = ODEFunc(dim=D, hidden=H).cuda()
        # 0.5 * N(0, 1) IC keeps |y**3| moderate and the adaptive solver
        # from taking pathologically small steps at random-init.
        y0 = (torch.randn(B, D, device="cuda", dtype=torch.float32) * 0.5)

        # ---- torchdiffeq ----
        t_td = _time_callable(
            lambda f=f, y0=y0: _torchdiffeq_odeint(
                f, y0, t_span, atol=atol, rtol=rtol,
            ),
            iters=iters, warmup=warmup,
        )

        # ---- torchode (built + compiled once per config) ----
        try:
            import torchode as to
            term = to.ODETerm(f, with_stats=False)
            method = to.Dopri5(term=term)
            controller = to.IntegralController(atol=atol, rtol=rtol, term=term)
            solver_obj = to.AutoDiffAdjoint(method, controller)
            try:
                solver_obj = torch.compile(
                    solver_obj, mode="reduce-overhead", fullgraph=False,
                )
            except Exception:  # noqa: BLE001
                pass

            t_start = torch.full((B,), float(t_span[0]),
                                 device=y0.device, dtype=y0.dtype)
            t_end = torch.full((B,), float(t_span[1]),
                               device=y0.device, dtype=y0.dtype)

            def torchode_call(solver_obj=solver_obj, y0=y0,
                              t_start=t_start, t_end=t_end):
                prob = to.InitialValueProblem(
                    y0=y0, t_start=t_start, t_end=t_end,
                )
                return solver_obj.solve(prob).ys[:, -1]

            t_to = _time_callable(torchode_call, iters=iters, warmup=warmup)
        except ImportError:
            t_to = float("nan")
        except Exception as e:  # noqa: BLE001
            # Some torchode compile paths crash on very small or unusual
            # shapes; record the failure but keep the sweep going.
            print(f"  [torchode error on {label}: {type(e).__name__}: {e}]")
            t_to = float("nan")

        # ---- Manual CUDAGraph + host-sync loop (NO STF) ----
        forward_cg, _ = _build_cudagraph_host_odeint_persistent(
            f, y0, t_span, atol=atol, rtol=rtol,
        )
        t_cg = _time_callable(forward_cg, iters=iters, warmup=warmup)

        # ---- STF drop-in (persistent context per config) ----
        forward, ctx, _ = _build_stf_odeint_persistent(
            f, y0, t_span, atol=atol, rtol=rtol,
        )
        try:
            t_stf = _time_stf_forward(forward, iters=iters, warmup=warmup)
        finally:
            ctx.finalize()

        rows.append((label, B, D, H, t_td, t_to, t_cg, t_stf))

    # Pretty table. Columns:
    #   td = torchdiffeq (pure host-loop)
    #   to = torchode (batched + torch.compile)
    #   cg = manual CUDAGraph replay + host-side cond.item() (NO STF)
    #   stf = STF ctx.while_loop + device-side cond
    print(
        f"\n=== Problem-size sweep: t_span={t_span}, atol=rtol={atol}, "
        f"iters={iters} (median) ==="
    )
    hdr = (
        f"  {'config':<22} {'B':>4} {'D':>5} {'H':>5} "
        f"{'torchdiffeq':>12} {'torchode':>12} {'manual-cg':>12} "
        f"{'stf':>10} {'vs td':>8} {'vs manual-cg':>14}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, B, D, H, t_td, t_to, t_cg, t_stf in rows:
        to_s = "n/a" if t_to != t_to else f"{t_to:8.2f} ms"
        vs_td = f"{t_td / t_stf:>5.2f}x" if t_stf > 0 else "nan"
        vs_cg = (
            f"{t_cg / t_stf:>5.2f}x"
            if t_cg == t_cg and t_stf > 0 else "n/a"
        )
        print(
            f"  {label:<22} {B:>4} {D:>5} {H:>5} "
            f"{t_td:8.2f} ms  {to_s:>10}  {t_cg:8.2f} ms "
            f"{t_stf:7.2f} ms {vs_td:>7}  {vs_cg:>12}"
        )
