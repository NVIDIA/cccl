# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-LoRA with streamed weights (experiment 1: scheduling matters).

Companion to ``bench_multi_lora_vs_fused.py``. That file measures the
fused-kernel side of the multi-LoRA problem in isolation, where STF's
scheduler has nothing to do. This file puts STF in the scenario it is
actually designed for: per-round adapter weights live in pinned host
memory and must be H2D-copied to GPU every forward (the cache-miss
regime that Punica / S-LoRA papers call out as the hard case), and we
ask whether the overlap of H2D transfer with compute buys us something
over a naive "copy then compute" loop.

Workload
--------
Simulate N rounds (each round is one LoRA-bearing layer's worth of
per-adapter work). Per round there are K LoRA adapters sharing a
single input ``x``. Round ``i`` produces deltas ``y_i[k] = alpha/r *
((x @ A_{i,k}) @ B_{i,k})``. All adapter weights across all N rounds
live in pinned host memory; the GPU-resident weight working set is
**two rounds (double-buffered)** -- mimicking an adapter cache that
can hold only the current and the next layer.

Per-round compute is the **Triton fused SGMV/BGMV kernel** from the
sibling ``bench_multi_lora_vs_fused.py`` (2 kernels per round,
independent of K, dispatching BGMV at seq=1 and SGMV otherwise).
That is deliberate: we are testing whether STF adds value *on top of*
a production-grade fused kernel, not whether STF competes with one.
Philosophy: use the fused kernel where fusion matters (across K
adapters within a round), and use ``task`` where scheduling matters
(between rounds, so H2D for round i+1 overlaps compute for round i).

Rows
----
- ``py/serial``    : single stream; for each round, H2D both weights
                     then run the compute. Zero overlap. Wallclock =
                     ``N * (t_h2d + t_compute)``.
- ``py/stream``    : manual double-buffer with two streams (copy +
                     compute) and events. ``h2d(i+1)`` is posted on
                     the copy stream while ``compute(i)`` runs on the
                     compute stream; events enforce the
                     read-after-write and write-after-read deps.
                     Scheduling LOC: ~25.
- ``stf``          : one STF DAG with 2*N tasks (``h2d_i`` as a
                     device task writing the GPU weight buffer,
                     ``compute_i`` reading it), expressed as plain
                     read/write deps on per-round logical data. No
                     stream or event bookkeeping in the user code.
                     Scheduling LOC: ~2 per task.

Metric
------
Wall-clock ms per N-round forward, and overlap ratio ``py/serial /
row`` (>=1.0 means faster than the serial baseline). Ideal overlap
ratio is ``(t_h2d + t_compute) / max(t_h2d, t_compute)``.

Env knobs
---------
- ``LLM_LORA_STREAM_BENCH=1``   enable the pytest entry
- ``LLM_LORA_STREAM_QUICK=1``   run only one cell
- ``LLM_LORA_STREAM_N=8,32``    N rounds values
- ``LLM_LORA_STREAM_SEQ=1,512`` seq values
- ``LLM_LORA_STREAM_ITERS=20``  timed iterations per cell
- ``LLM_LORA_STREAM_WARMUP=5``  warmup iterations per cell

Methodology note (important, read before interpreting)
-------------------------------------------------------
An earlier version of this benchmark registered each round's host
weights as ``ctx.logical_data(A_host_np[i], data_place.host())`` and
marked them read-only. STF's scheduler then H2D-copied each weight
once (on first access during warmup/correctness) and cached the
device-resident copy for the rest of the run -- so subsequent
forwards did **zero** H2D while the py rows blindly re-copied pinned
host → GPU every forward. That made STF look ~1.5x faster than
``py/stream``, but the win was almost entirely from caching, not
scheduling.

``nsys stats --report cuda_gpu_mem_size_sum --filter-nvtx ...`` on
the timed regions was what caught it::

  (buggy version)
  py/serial/timed : 96 HtoD copies, 50.3 MB
  py/stream/timed : 96 HtoD copies, 50.3 MB
  stf/timed       :  0 HtoD copies,  0.0 MB   <-- cached, not scheduled

The current file fixes this by NOT registering host weights as
logical_data. Instead the STF DAG carries only device-side state (a
double-buffered pair of weight buffers plus the per-round output),
and each round's H2D is a ``pytorch_task(..., l_A_dev[buf].write())``
whose body calls ``.copy_(host_pinned_tensor, non_blocking=True)``.
The device buffer gets overwritten every other round (double-buffer
wrap), so STF's scheduler is forced to re-H2D every forward, same
as the py rows.

Post-fix H2D accounting (same nsys filter)::

  py/serial/timed : 96 HtoD copies, 50.3 MB
  py/stream/timed : 96 HtoD copies, 50.3 MB
  stf/timed       : 96 HtoD copies, 50.3 MB   <-- apples-to-apples

Results (run on this box, ``ITERS=20 WARMUP=5``, K=4, r=16, Triton fused)
-------------------------------------------------------------------------

ms / forward and overlap ratios vs ``py/serial``, fair H2D::

   seq   N      py/serial    py/stream          stf   str/ser   stf/ser   stf/str
  ----   -   ----------   ----------   ----------   -------   -------   -------
     1   8       2.94 ms      2.74 ms      2.74 ms     1.08x     1.07x     1.00x
     1  32      11.66 ms     10.92 ms     11.00 ms     1.07x     1.06x     0.99x
   512   8       3.20 ms      2.75 ms      2.77 ms     1.16x     1.16x     0.99x
   512  32      12.82 ms     10.97 ms     11.01 ms     1.17x     1.16x     1.00x

Reading
-------
- **STF is at parity with ``py/stream``** across the entire sweep
  (``stf/str`` = 0.99-1.00x). It is NOT 1.5x faster; that earlier
  number was the caching artifact above.
- Both overlap paths achieve the expected arithmetic overlap: ~1.07x
  at seq=1 (compute is tiny, overlap window almost closed) and
  ~1.16x at seq=512 (compute and transfer comparable).
- STF's actual value-add vs ``py/stream`` at these shapes is
  **developer ergonomics**, not raw perf:

  * ``py/stream``: ~25 LOC of explicit ``torch.cuda.Stream`` +
    ``torch.cuda.Event`` + ``wait_event`` / ``record`` scaffolding
    per forward, plus manual reasoning about WAR/WAW on the
    double-buffer indices.
  * ``stf``: ~2 LOC per task (one ``pytorch_task`` for the merged
    H2D, one for the compute), read/write deps on ``l_A_dev[buf]``
    / ``l_B_dev[buf]`` doing the same job. No events in user code.

This is the complement of ``bench_multi_lora_vs_fused.py``: on an
isolated fused-kernel layer, STF loses to the fused kernel because
there is nothing to schedule. On a multi-round layer with weights
streamed from host memory, STF matches a careful hand-written
stream prefetcher -- with an order of magnitude less scheduling code
in the user program.

Philosophy restated
-------------------
- Fused kernels where fusion matters (across K adapters within a
  round). ``fused_triton`` is called unchanged in every row.
- ``task`` where scheduling matters (across rounds). The STF version
  adds one merged H2D ``pytorch_task`` + one compute ``pytorch_task``
  per round; the scheduler places them on separate streams and
  enforces the double-buffer WAR/WAW dependencies.
- Neither primitive is being asked to do the other's job.

Caveats
-------
- ``py/stream`` uses a single copy stream. A more aggressive
  hand-written version with a pool of copy streams could probably
  squeeze a few more percent over STF here; the point of the row is
  "what does ~25 LOC of careful PyTorch get you", and that's already
  enough to match STF.
- STF's compute task writes ``ty.copy_(fused_triton(...))`` because
  the fused launcher allocates its own output -- one extra
  (K,S,H)-sized D2D copy per compute task that py rows do not pay.
  At seq=512 that's ~16 MB per round = ~5 us kernel time. It does
  not change the parity conclusion.
- Both overlap rows' absolute wins over ``py/serial`` are modest
  (1.06-1.17x) because PCIe H2D is the real bottleneck at these
  shapes and there is only so much compute to overlap against
  transfer. A workload with bigger per-round compute (larger seq,
  larger rank, or multi-layer DAG within a round) would widen the
  overlap window and push both ``py/stream`` and ``stf`` toward the
  ~2x theoretical ceiling.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Reuse the Triton SGMV/BGMV fused kernels from the sibling bench. This keeps
# the "kernel" story identical across rows -- all three rows call the same
# fused Triton launcher inside their per-round compute step. The only thing
# that differs between rows is how H2D transfer is scheduled against that
# compute.
from bench_multi_lora_vs_fused import fused_triton  # noqa: E402
from pytorch_task import pytorch_task  # noqa: E402

from cuda import stf  # noqa: E402
from cuda.stf._experimental import data_place  # noqa: E402

_FP16 = torch.float16
_NP_FP16 = np.float16


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamedConfig:
    N_rounds: int = 32
    K: int = 4  # adapters per round
    hidden: int = 4096
    rank: int = 16
    seq: int = 512
    alpha: float = 16.0

    def alpha_over_r(self) -> float:
        return self.alpha / float(self.rank)


# ---------------------------------------------------------------------------
# Pinned host weight generation.
# ---------------------------------------------------------------------------


@dataclass
class StreamedCase:
    cfg: StreamedConfig
    # Per-round host weights, pinned. Shape per round: A=(K,H,r), B=(K,r,H).
    A_host_np: list[np.ndarray]  # N entries, numpy views backed by pinned tensors
    B_host_np: list[np.ndarray]
    A_host_pt: list[torch.Tensor]  # pinned CPU tensors (same storage as _np views)
    B_host_pt: list[torch.Tensor]
    x_dev: torch.Tensor  # (S, H), persistent on device


def _gen_case(cfg: StreamedConfig, *, seed: int = 0) -> StreamedCase:
    rng = np.random.default_rng(seed)
    N, K, H, r, S = cfg.N_rounds, cfg.K, cfg.hidden, cfg.rank, cfg.seq

    A_host_pt: list[torch.Tensor] = []
    B_host_pt: list[torch.Tensor] = []
    A_host_np: list[np.ndarray] = []
    B_host_np: list[np.ndarray] = []
    for i in range(N):
        a = rng.standard_normal((K, H, r), dtype=np.float32).astype(_NP_FP16) * 0.02
        b = rng.standard_normal((K, r, H), dtype=np.float32).astype(_NP_FP16) * 0.02
        a_pt = torch.from_numpy(a).pin_memory()
        b_pt = torch.from_numpy(b).pin_memory()
        A_host_pt.append(a_pt)
        B_host_pt.append(b_pt)
        A_host_np.append(a_pt.numpy())
        B_host_np.append(b_pt.numpy())

    x_np = rng.standard_normal((S, H), dtype=np.float32).astype(_NP_FP16) * 0.02
    x_dev = torch.from_numpy(x_np).cuda()
    return StreamedCase(
        cfg=cfg,
        A_host_np=A_host_np,
        B_host_np=B_host_np,
        A_host_pt=A_host_pt,
        B_host_pt=B_host_pt,
        x_dev=x_dev,
    )


# ---------------------------------------------------------------------------
# Compute kernel: Triton fused SGMV / BGMV (reused from sibling bench).
# 2 kernels per round across all K adapters; dispatches BGMV at seq=1 and
# SGMV otherwise. Identical to what a production multi-LoRA server would
# use. Fused across K adapters, NOT fused across rounds (the whole point
# of this benchmark is that inter-round scheduling is the remaining lever).
# ---------------------------------------------------------------------------


def _compute_round(
    x: torch.Tensor,  # (S, H)
    A_stack: torch.Tensor,  # (K, H, r)
    B_stack: torch.Tensor,  # (K, r, H)
    alpha: float,
) -> torch.Tensor:  # (K, S, H)
    """One round: y[k] = alpha * ((x @ A[k]) @ B[k]) for k in 0..K-1.

    Dispatches to SGMV / BGMV via ``fused_triton``. Returns a freshly
    allocated ``(K, S, H)`` tensor on the currently active stream.
    """
    return fused_triton(x, A_stack, B_stack, alpha)


# ---------------------------------------------------------------------------
# Row 1: py/serial -- single stream, no overlap.
# ---------------------------------------------------------------------------


def _build_py_serial_forward(case: StreamedCase):
    cfg = case.cfg
    N, K, S, H, r = cfg.N_rounds, cfg.K, cfg.seq, cfg.hidden, cfg.rank
    alpha = cfg.alpha_over_r()
    dev = torch.device("cuda")

    # Single GPU weight buffer; serial means no pipelining.
    A_gpu = torch.empty((K, H, r), dtype=_FP16, device=dev)
    B_gpu = torch.empty((K, r, H), dtype=_FP16, device=dev)

    x = case.x_dev
    A_host = case.A_host_pt
    B_host = case.B_host_pt
    y_list: list[torch.Tensor | None] = [None] * N

    def forward() -> list[torch.Tensor | None]:
        for i in range(N):
            A_gpu.copy_(A_host[i], non_blocking=True)
            B_gpu.copy_(B_host[i], non_blocking=True)
            y_list[i] = _compute_round(x, A_gpu, B_gpu, alpha)
        return y_list

    return forward, y_list


# ---------------------------------------------------------------------------
# Row 2: py/stream -- manual double-buffered prefetch.
# ---------------------------------------------------------------------------


def _build_py_stream_forward(case: StreamedCase):
    cfg = case.cfg
    N, K, S, H, r = cfg.N_rounds, cfg.K, cfg.seq, cfg.hidden, cfg.rank
    alpha = cfg.alpha_over_r()
    dev = torch.device("cuda")

    # Double-buffered GPU weights.
    A_gpu = [torch.empty((K, H, r), dtype=_FP16, device=dev) for _ in range(2)]
    B_gpu = [torch.empty((K, r, H), dtype=_FP16, device=dev) for _ in range(2)]

    x = case.x_dev
    A_host = case.A_host_pt
    B_host = case.B_host_pt
    copy_stream = torch.cuda.Stream()
    y_list: list[torch.Tensor | None] = [None] * N

    def forward() -> list[torch.Tensor | None]:
        compute_stream = torch.cuda.current_stream()
        copy_done: list[torch.cuda.Event] = [torch.cuda.Event() for _ in range(N)]
        compute_done: list[torch.cuda.Event] = [torch.cuda.Event() for _ in range(N)]

        with torch.cuda.stream(copy_stream):
            A_gpu[0].copy_(A_host[0], non_blocking=True)
            B_gpu[0].copy_(B_host[0], non_blocking=True)
            copy_done[0].record(copy_stream)

        for i in range(N):
            compute_stream.wait_event(copy_done[i])

            if i + 1 < N:
                buf = (i + 1) % 2
                with torch.cuda.stream(copy_stream):
                    # Must wait for compute on this same buffer to finish
                    # reading it (WAR). That's compute i-1 for i+1>=2.
                    if i - 1 >= 0 and (i - 1) % 2 == buf:
                        copy_stream.wait_event(compute_done[i - 1])
                    A_gpu[buf].copy_(A_host[i + 1], non_blocking=True)
                    B_gpu[buf].copy_(B_host[i + 1], non_blocking=True)
                    copy_done[i + 1].record(copy_stream)

            buf_i = i % 2
            y_list[i] = _compute_round(x, A_gpu[buf_i], B_gpu[buf_i], alpha)
            compute_done[i].record(compute_stream)

        return y_list

    return forward, y_list


# ---------------------------------------------------------------------------
# Row 3: STF.
# ---------------------------------------------------------------------------


def _build_stf_forward(case: StreamedCase):
    """Build the STF forward.

    Design note on fair H2D accounting: we deliberately do NOT register
    the per-round host weights as ``logical_data(host_np, host_place)``,
    because STF would then cache the device-resident copy after the
    first access -- so subsequent forwards would do zero H2D, while the
    py rows always re-H2D from pinned host. To make the comparison
    apples-to-apples we mirror py/stream: the STF DAG carries only
    device-side state (a double-buffered pair of weight buffers plus
    the per-round output), and each round issues explicit H2D via
    ``pytorch_task(..., l_A_dev[buf].write())`` whose body calls
    ``d.copy_(A_host_pt[i], non_blocking=True)`` on pinned host
    tensors. That keeps STF in charge of scheduling (choice of stream,
    WAR/WAW ordering on the double buffer, overlap with compute) while
    guaranteeing the same number of H2D bytes per forward as the py
    rows.
    """
    cfg = case.cfg
    N, K, S, H, r = cfg.N_rounds, cfg.K, cfg.seq, cfg.hidden, cfg.rank
    alpha = cfg.alpha_over_r()

    ctx = stf.stackable_context()

    # Device-resident, persistent.
    l_x = ctx.logical_data(case.x_dev, data_place.device(0), name="x")
    if hasattr(l_x, "set_read_only"):
        l_x.set_read_only()

    # Double-buffered device-side weight storage (exactly like py/stream).
    A_gpu = [
        torch.empty((K, H, r), dtype=_FP16, device="cuda"),
        torch.empty((K, H, r), dtype=_FP16, device="cuda"),
    ]
    B_gpu = [
        torch.empty((K, r, H), dtype=_FP16, device="cuda"),
        torch.empty((K, r, H), dtype=_FP16, device="cuda"),
    ]
    l_A_dev = [
        ctx.logical_data(A_gpu[b], data_place.device(0), name=f"A_dev_{b}")
        for b in range(2)
    ]
    l_B_dev = [
        ctx.logical_data(B_gpu[b], data_place.device(0), name=f"B_dev_{b}")
        for b in range(2)
    ]

    # Per-round output (K, S, H). STF allocates on device when written.
    l_ys = [
        ctx.logical_data_empty((K, S, H), _NP_FP16, name=f"y_{i}") for i in range(N)
    ]

    y_host_buf = np.empty((N, K, S, H), dtype=_NP_FP16)

    # Pinned-host source tensors (captured by closure; not STF logical_data).
    A_host_pt = case.A_host_pt
    B_host_pt = case.B_host_pt

    def forward() -> None:
        for i in range(N):
            buf = i % 2
            # Single H2D task per round: copy both A[i] and B[i] from
            # pinned host into device buffers A_gpu[buf], B_gpu[buf].
            # Merging the two copies into one task halves per-round
            # task-submit overhead vs splitting into A-task and B-task.
            with pytorch_task(
                ctx,
                l_A_dev[buf].write(),
                l_B_dev[buf].write(),
            ) as (d_a, d_b):
                d_a.copy_(A_host_pt[i], non_blocking=True)
                d_b.copy_(B_host_pt[i], non_blocking=True)
            # Compute task: reads device weights and x, writes y_i.
            with pytorch_task(
                ctx,
                l_x.read(),
                l_A_dev[buf].read(),
                l_B_dev[buf].read(),
                l_ys[i].write(),
            ) as (tx, tA, tB, ty):
                ty.copy_(_compute_round(tx, tA, tB, alpha))

    def read_y_host() -> np.ndarray:
        """Enqueue N host_launch copies to get the full (N,K,S,H) result."""
        for i in range(N):
            target = y_host_buf[i]

            def _copy(y_arr, tgt=target):
                np.copyto(tgt, y_arr)

            ctx.host_launch(l_ys[i].read(), fn=_copy)
        return y_host_buf

    def finalize() -> None:
        ctx.finalize()

    return ctx, forward, read_y_host, finalize


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------


def _check_close(
    label: str, got: np.ndarray, ref: np.ndarray, *, atol=5e-2, rtol=1e-2
) -> None:
    if got.shape != ref.shape:
        raise AssertionError(
            f"[correctness:{label}] shape mismatch: got {got.shape} vs ref {ref.shape}"
        )
    got_f32 = got.astype(np.float32)
    ref_f32 = ref.astype(np.float32)
    diff = np.abs(got_f32 - ref_f32)
    allowed = atol + rtol * np.abs(ref_f32)
    bad = diff > allowed
    if bad.any():
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(ref_f32) + 1e-6)).max())
        raise AssertionError(
            f"[correctness:{label}] {int(bad.sum())}/{bad.size} elements "
            f"exceed atol={atol} rtol={rtol}; max_abs={max_abs:.3e} "
            f"max_rel={max_rel:.3e}"
        )


def _stack_list(y_list) -> np.ndarray:
    return torch.stack([y for y in y_list], dim=0).detach().cpu().numpy()


def correctness_sanity(case: StreamedCase) -> None:
    # py/serial is the fp16 reference.
    fwd_ser, y_list_ser = _build_py_serial_forward(case)
    fwd_ser()
    torch.cuda.synchronize()
    ref = _stack_list(y_list_ser)

    # py/stream
    fwd_str, y_list_str = _build_py_stream_forward(case)
    fwd_str()
    torch.cuda.synchronize()
    got_str = _stack_list(y_list_str)
    _check_close("py/stream", got_str, ref)

    # stf
    ctx, fwd_stf, read_y_host, fin = _build_stf_forward(case)
    try:
        fwd_stf()
        got_stf = read_y_host()
    finally:
        fin()
    _check_close("stf", got_stf, ref)


# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------


_nvtx_push = torch.cuda.nvtx.range_push
_nvtx_pop = torch.cuda.nvtx.range_pop


def _time_callable(fn, *, iters: int, warmup: int, label: str = "") -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    _nvtx_push(f"{label}/timed")
    t0 = time.perf_counter()
    for i in range(iters):
        _nvtx_push(f"{label}/iter-{i}")
        _ = fn()
        _nvtx_pop()
    torch.cuda.synchronize()
    _nvtx_pop()
    return (time.perf_counter() - t0) / iters


def _time_stf(
    case: StreamedCase, *, iters: int, warmup: int, label: str = "stf"
) -> float:
    ctx, fwd, _read, fin = _build_stf_forward(case)
    try:
        for _ in range(warmup):
            fwd()
        torch.cuda.synchronize()
        _nvtx_push(f"{label}/timed")
        t0 = time.perf_counter()
        for i in range(iters):
            _nvtx_push(f"{label}/iter-{i}")
            fwd()
            _nvtx_pop()
        torch.cuda.synchronize()
        _nvtx_pop()
        elapsed = (time.perf_counter() - t0) / iters
    finally:
        fin()
    return elapsed


# ---------------------------------------------------------------------------
# Sweep.
# ---------------------------------------------------------------------------


_ROWS = ("py/serial", "py/stream", "stf")


def run_cell(cfg: StreamedConfig, *, iters: int, warmup: int) -> dict:
    case = _gen_case(cfg, seed=0)
    correctness_sanity(case)
    row: dict = {"seq": cfg.seq, "N": cfg.N_rounds, "K": cfg.K, "r": cfg.rank}

    fwd_ser, _ = _build_py_serial_forward(case)
    row["py/serial"] = _time_callable(
        fwd_ser, iters=iters, warmup=warmup, label="py/serial"
    )

    fwd_str, _ = _build_py_stream_forward(case)
    row["py/stream"] = _time_callable(
        fwd_str, iters=iters, warmup=warmup, label="py/stream"
    )

    row["stf"] = _time_stf(case, iters=iters, warmup=warmup, label="stf")
    return row


def _parse_int_list(env: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(env)
    if raw is None:
        return default
    return tuple(int(x) for x in raw.split(",") if x.strip())


def run_full() -> list[dict]:
    iters = int(os.environ.get("LLM_LORA_STREAM_ITERS", "20"))
    warmup = int(os.environ.get("LLM_LORA_STREAM_WARMUP", "5"))
    Ns = _parse_int_list("LLM_LORA_STREAM_N", (8, 32))
    seqs = _parse_int_list("LLM_LORA_STREAM_SEQ", (1, 512))

    rows: list[dict] = []
    for seq in seqs:
        for N in Ns:
            cfg = StreamedConfig(N_rounds=N, seq=seq)
            rows.append(run_cell(cfg, iters=iters, warmup=warmup))
    return rows


def run_quick() -> list[dict]:
    iters = int(os.environ.get("LLM_LORA_STREAM_ITERS", "10"))
    warmup = int(os.environ.get("LLM_LORA_STREAM_WARMUP", "3"))
    cfg = StreamedConfig(N_rounds=16, seq=512)
    return [run_cell(cfg, iters=iters, warmup=warmup)]


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _fmt_table(rows: list[dict]) -> str:
    hdr = f"{'seq':>4} {'N':>3} {'K':>3} {'r':>3}"
    for r in _ROWS:
        hdr += f" {r:>14}"
    hdr += f" {'str/ser':>10} {'stf/ser':>10} {'stf/str':>10}"
    lines = [hdr, "-" * len(hdr)]
    for row in rows:
        ser = row["py/serial"]
        stream = row["py/stream"]
        stf_ms = row["stf"]
        r_str = ser / stream if stream > 0 else float("inf")
        r_stf = ser / stf_ms if stf_ms > 0 else float("inf")
        r_stf_str = stream / stf_ms if stf_ms > 0 else float("inf")
        line = (
            f"{row['seq']:>4} {row['N']:>3} {row['K']:>3} {row['r']:>3}"
            f" {ser * 1e3:>11.3f} ms"
            f" {stream * 1e3:>11.3f} ms"
            f" {stf_ms * 1e3:>11.3f} ms"
            f" {r_str:>9.2f}x"
            f" {r_stf:>9.2f}x"
            f" {r_stf_str:>9.2f}x"
        )
        lines.append(line)
    return "\n".join(lines)


def _print_results(rows: list[dict]) -> None:
    print("\n=== Multi-LoRA streamed-weights sweep (ms / forward) ===")
    print(_fmt_table(rows))
    print(
        "\nOverlap ratios vs py/serial (higher = better):\n"
        "  str/ser : py/stream speedup over py/serial\n"
        "            (ideal ~ (t_h2d + t_comp) / max(t_h2d, t_comp))\n"
        "  stf/ser : STF speedup over py/serial\n"
        "            (if ~1.0, STF runtime did not overlap H2D and compute)\n"
        "  stf/str : STF vs hand-written manual-stream prefetcher\n"
    )


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("LLM_LORA_STREAM_BENCH", "0") != "1",
    reason="set LLM_LORA_STREAM_BENCH=1 to run the streamed-weights bench",
)
def test_multi_lora_streamed_bench():
    if os.environ.get("LLM_LORA_STREAM_QUICK", "0") == "1":
        rows = run_quick()
    else:
        rows = run_full()
    _print_results(rows)
    for row in rows:
        for name in _ROWS:
            assert row[name] > 0.0, f"non-positive time for {name}: {row}"


if __name__ == "__main__":
    if os.environ.get("LLM_LORA_STREAM_QUICK", "0") == "1":
        rows = run_quick()
    else:
        rows = run_full()
    _print_results(rows)
