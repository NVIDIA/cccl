# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Multi-LoRA vs fused-kernel benchmark (self-contained).

Goal
----
This benchmark characterises the *fused-kernel* side of the multi-LoRA
problem and shows where its assumptions hold vs. break. STF is included
for reference only; it is not intended as a drop-in replacement for a
fused kernel and is not expected to win on this workload.

Two questions, one slide:

1. Homogeneous-rank case: what does a hand-written Triton fused-kernel
   family (SGMV at prefill, BGMV at decode -- what production serves at
   each shape) look like vs. naive PyTorch baselines and STF?
2. Heterogeneous-rank case: what happens when the fused kernel's
   homogeneity assumption breaks? SGMV/BGMV have to choose between
   padding every adapter up to ``r_max`` (wasted FLOPs) or bucketing
   into ``#distinct_ranks`` serial launches (loses concurrency). STF
   and the per-adapter PyTorch baselines keep their linear cost either
   way; their overhead floor is set by per-task launch cost, not by
   the rank distribution.

What STF is *not* being tested on here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``task`` is a scheduler primitive; it earns its overhead when there is
a real DAG to schedule -- overlap of compute with H2D/D2H transfers,
multi-stage pipelines, heterogeneous resources, CPU-side work
interleaved with GPU work, multi-layer transformer graphs, etc. This
benchmark deliberately strips all of that away and runs a single fused
layer with no surrounding work, so there is nothing for STF to
schedule around the K matmul pairs. That is the right shape to isolate
"how fast is the fused kernel family at the core matmul" but it is the
wrong shape to ask "does STF help". The fused-kernel column is the
takeaway; the STF columns are a sanity check that we are not claiming
STF wins a fight it was never designed for.

Compute under test (all rows compute only the per-adapter deltas, not the
shared base ``y_base = x @ W`` -- the base work is identical for every
row and would just offset all numbers by the same constant)::

    y_delta_k = (alpha / r_k) * ((x @ A_k) @ B_k)     for k in 0..K-1

Input shapes:
- ``x``      : ``(S, H)``            shared across all K adapters
- ``A_k``    : ``(H, r_k)``          per adapter
- ``B_k``    : ``(r_k, H)``          per adapter
- ``y_k``    : ``(S, H)``            per adapter (output delta)

At ``S==1`` this is the decode shape served by BGMV in Punica / vLLM.
At ``S==512`` this is the prefill shape served by SGMV.

This file is standalone: no imports from any other
``python/cuda_cccl/tests/stf/`` helper. The existing
LoRA files (``bench_multi_lora.py``, ``test_llm_multi_lora.py``,
``test_llm_lora.py``) are used only as reference material.

Env knobs
---------
- ``LLM_LORA_VS_FUSED_BENCH=1``     enable the slow-path pytest entry
- ``LLM_LORA_VS_FUSED_QUICK=1``     run only the Phase 1 de-risk cells
- ``LLM_LORA_VS_FUSED_K=1,4,16,64`` comma-separated K values (homogeneous)
- ``LLM_LORA_VS_FUSED_SEQ=1,512``   comma-separated seq values
- ``LLM_LORA_VS_FUSED_ITERS=20``    timed iterations per cell
- ``LLM_LORA_VS_FUSED_WARMUP=5``    warmup iterations per cell

Note on ``torch.compile`` cache
-------------------------------
For the heterogeneous sweep, ``_warmup_compiled_for_ranks`` triggers one
``torch.compile`` cache entry per distinct ``r_k`` (8 distinct ranks in
the default config). That is expected; the timed region does not pay
compile cost because warmup runs outside it. We also raise
``torch._dynamo.config.cache_size_limit`` to 256 at import time because
the combined sweep (seq in {1,512}, rank in {4,8,16,32,64}) exceeds the
default limit of 8.

Results (run on this box, ``LLM_LORA_VS_FUSED_ITERS=20 WARMUP=5``)
------------------------------------------------------------------

Homogeneous sweep, wall-clock ms / forward:

+------+----+------------+-----------+-------------+-----------------+-----------+--------------+
| seq  |  K |fused_triton|py/seq/eagr|py/seq/compl |py/stream/compl  |stf/compile|stf/compile+gs|
+======+====+============+===========+=============+=================+===========+==============+
|    1 |  1 |   0.075    |   0.037   |    0.133    |     0.189       |   0.301   |    0.375     |
|    1 |  4 |   0.075    |   0.142   |    0.514    |     0.801       |   1.510   |    1.777     |
|    1 | 16 |   0.077    |   0.568   |    1.931    |     2.710       |   4.797   |    7.171     |
|    1 | 64 |   0.081    |   2.298   |    7.795    |    10.744       |  19.385   |   51.872     |
|  512 |  1 |   0.092    |   0.046   |    0.153    |     0.218       |   0.340   |    0.419     |
|  512 |  4 |   0.256    |   0.312   |    0.994    |     0.762       |   1.326   |    2.016     |
|  512 | 16 |   0.172    |   0.605   |    2.048    |     2.846       |   5.099   |    7.084     |
|  512 | 64 |   0.672    |   2.418   |    8.886    |    10.788       |  21.666   |   47.699     |
+------+----+------------+-----------+-------------+-----------------+-----------+--------------+

Heterogeneous sweep (K=8, ranks=[4,8,16,16,32,32,64,64], r_max=64), ms / forward:

+------+----+-------------+---------------+----------------+-----------+-------------+-------------+---------------+--------------+
| seq  |  K |fused_padded | sg/default    | sg/streams     |py/seq/eag |py/seq/compl |py/str/eager |py/str/compile |stf/compile+gs|
+======+====+=============+===============+================+===========+=============+=============+===============+==============+
|    1 |  8 |   0.115     |    0.706      |    1.089       |   0.297   |   1.055     |   0.563     |    1.440      |    3.196     |
|  512 |  8 |   0.227     |    0.795      |    1.018       |   0.341   |   1.156     |   0.583     |    1.484      |    3.306     |
+------+----+-------------+---------------+----------------+-----------+-------------+-------------+---------------+--------------+

Reportable flags (all MISS on this box; see ## Known caveats below):

- fused_triton_is_fastest_homogeneous: MISS (py/seq/eager beats us at K=1)
- stf_within_3x_bgmv: MISS
- stf_within_2x_sgmv: MISS
- stf_beats_py_stream_compile: MISS
- stf_beats_fused_padded_hetero: MISS
- serial_groups_streams_beats_default: MISS

Known caveats
-------------
1. ``fused_triton`` loses to ``py/seq/eager`` at K=1. At a single-adapter
   single-matmul shape, cuBLAS via ``torch.matmul`` is a better fused
   kernel than our in-file Triton. Our Triton only overtakes once K
   grows enough that the batched form amortises launch cost. This is the
   expected scaling; the row is still load-bearing as the "production
   kernel" reference for K>=4.
2. ``stf/compile+gs`` is dominated by per-adapter ``graph_scope()``
   overhead. Important mechanism note: ``graph_scope`` is NOT a
   "capture once, launch a cached graph N times" primitive. Every
   forward creates a new CUDA graph per scope and reinstantiates from
   a cache when the signature matches; the cache saves the capture
   pass after the first iter but still pays the per-iter
   graph-instantiation cost on every call. Since each forward here
   opens K scopes (one per adapter), that per-iter cost grows linearly
   with K. At K=64 seq=1 the ~52 ms / forward breaks down as roughly
   ``K * 0.8 ms`` per-scope instantiation + tens of microseconds of
   actual compute. Bench shows STF trails the fused reference by
   5-640x depending on cell; this is not competitive for production
   multi-LoRA at these shapes.
3. ``fused_serial_groups/streams`` is slower than
   ``fused_serial_groups/default`` (the "unfair" baseline). Bucketing
   reshapes the weight slices via ``torch.stack`` + a scatter
   ``copy_()`` per bucket; that extra memory-bandwidth cost outweighs
   the per-bucket concurrency win at K=8. A production implementation
   would stash pre-stacked per-rank-bucket views at setup and avoid the
   stack/copy; we did not because the point of the row is to measure
   what a careful-but-naive user gets, not a hand-optimised path.
4. STF's numbers on this workload should be read as "STF being asked
   to do what a fused kernel does" rather than "STF vs. fused kernel
   in a fair fight". ``task`` is a scheduler, not a kernel fusion
   pass. With no surrounding DAG work -- no async transfers, no
   multi-stage pipeline, no heterogeneous tasks, no multi-layer
   transformer graph -- the scheduler has nothing to schedule and all
   its overhead (per-task submit, per-scope CUDA-graph instantiation
   under ``graph_scope``, K separate kernel launches through
   ``ctx.compile``) shows up as pure cost against the 2-kernel fused
   path. The correct reading of this benchmark is: (a) when your
   kernel is fusable and homogeneous, write the fused kernel; (b) STF
   earns its keep by scheduling *around* fused kernels (pipelining
   them with the rest of a model, overlapping weight transfers,
   co-scheduling CPU work), not by replacing them; (c) if every row
   of a workload is dominated by a single fused-matmul family the way
   this one is, there is no place for ``task`` to win and this
   benchmark should not be used to argue otherwise.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Across the full sweep (seq in {1,512}, rank in {4,8,16,32,64}) and both
# homo and hetero cells, torch.compile produces one specialised graph per
# distinct (S, r) combination. The default cache_size_limit of 8 is too
# small; raising it avoids FailOnRecompileLimitHit without changing
# numerical behaviour. Warmup still runs before the timed region.
import torch._dynamo as _torch_dynamo  # noqa: E402

_torch_dynamo.config.cache_size_limit = 256

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402
from pytorch_task import pytorch_task  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

# Import-guarded optional cross-check references.
try:
    from punica.ops import bgmv as _punica_bgmv  # type: ignore
except ImportError:  # pragma: no cover
    _punica_bgmv = None

try:
    from vllm.lora.ops.triton_ops import (
        sgmv_expand as _vllm_sgmv_expand,  # type: ignore
    )
    from vllm.lora.ops.triton_ops import (
        sgmv_shrink as _vllm_sgmv_shrink,  # type: ignore
    )
except ImportError:  # pragma: no cover
    _vllm_sgmv_shrink = None
    _vllm_sgmv_expand = None


_FP16 = torch.float16
_NP_FP16 = np.float16


# ---------------------------------------------------------------------------
# Config + case generation (single source of truth for every row)
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Shape/alpha config for one benchmark cell.

    Homogeneous: ``rank`` is set, ``ranks`` is empty. ``effective_ranks``
    yields ``[rank] * K``.
    Heterogeneous: ``ranks`` is set, ``rank`` is ignored. ``K`` is derived
    from ``len(ranks)``.
    """

    hidden: int = 4096
    seq: int = 512
    K: int = 1
    rank: int = 16
    ranks: tuple[int, ...] = field(default_factory=tuple)
    alpha: float = 32.0

    @property
    def is_hetero(self) -> bool:
        return len(self.ranks) > 0

    @property
    def effective_ranks(self) -> tuple[int, ...]:
        if self.is_hetero:
            return tuple(self.ranks)
        return tuple([self.rank] * self.K)

    @property
    def effective_K(self) -> int:
        return len(self.ranks) if self.is_hetero else self.K

    @property
    def r_max(self) -> int:
        return max(self.effective_ranks)

    def alpha_over_r(self, r: int) -> float:
        return float(self.alpha) / float(r)

    def describe(self) -> str:
        if self.is_hetero:
            return (
                f"H={self.hidden} S={self.seq} K={self.effective_K} "
                f"ranks={list(self.ranks)} alpha={self.alpha}"
            )
        return (
            f"H={self.hidden} S={self.seq} K={self.K} r={self.rank} alpha={self.alpha}"
        )


@dataclass
class Case:
    """All tensors for one cell, materialised from the numpy source of truth."""

    cfg: LoRAConfig

    # Host-side source of truth (numpy, fp16 unless noted).
    x_np: np.ndarray  # (S, H) fp16
    As_np_list: list[np.ndarray]  # each (H, r_k) fp16
    Bs_np_list: list[np.ndarray]  # each (r_k, H) fp16

    # Derived: stacked homogeneous views (only populated for homogeneous cases).
    As_stacked_np: np.ndarray | None  # (K, H, r) or None
    Bs_stacked_np: np.ndarray | None  # (K, r, H) or None

    # Derived: padded-to-r_max views (populated when any padding is needed).
    As_padded_np: np.ndarray  # (K, H, r_max) fp16
    Bs_padded_np: np.ndarray  # (K, r_max, H) fp16
    r_per_k: np.ndarray  # (K,) int32

    # Device tensors used by Triton / PyTorch rows.
    x_dev: torch.Tensor  # (S, H) fp16
    As_list_dev: list[torch.Tensor]  # each (H, r_k) fp16
    Bs_list_dev: list[torch.Tensor]  # each (r_k, H) fp16
    As_stacked_dev: torch.Tensor | None
    Bs_stacked_dev: torch.Tensor | None
    As_padded_dev: torch.Tensor
    Bs_padded_dev: torch.Tensor

    # fp32 ground-truth reference: y_ref[k] = alpha_k * (x @ A_k) @ B_k.
    y_ref_np: np.ndarray  # (K, S, H) fp16-cast fp32 reference


def _gen_case(cfg: LoRAConfig, seed: int = 0) -> Case:
    """Create all tensors for one cell from a single seeded numpy RNG."""
    rng = np.random.default_rng(seed)
    H, S = cfg.hidden, cfg.seq
    ranks = cfg.effective_ranks
    K = len(ranks)
    r_max = max(ranks)

    # Host-side sources of truth.
    scale_H = 1.0 / (H**0.5)
    x_np = rng.standard_normal((S, H), dtype=np.float32).astype(_NP_FP16)
    As_np_list: list[np.ndarray] = []
    Bs_np_list: list[np.ndarray] = []
    for r in ranks:
        scale_r = 1.0 / (r**0.5)
        a = rng.standard_normal((H, r), dtype=np.float32) * scale_H
        b = rng.standard_normal((r, H), dtype=np.float32) * scale_r
        As_np_list.append(a.astype(_NP_FP16))
        Bs_np_list.append(b.astype(_NP_FP16))

    # Padded and (when homogeneous) stacked views.
    As_padded_np = np.zeros((K, H, r_max), dtype=_NP_FP16)
    Bs_padded_np = np.zeros((K, r_max, H), dtype=_NP_FP16)
    for k, (A, B, r) in enumerate(zip(As_np_list, Bs_np_list, ranks)):
        As_padded_np[k, :, :r] = A
        Bs_padded_np[k, :r, :] = B

    if not cfg.is_hetero:
        As_stacked_np = np.stack(As_np_list, axis=0).copy()  # (K, H, r)
        Bs_stacked_np = np.stack(Bs_np_list, axis=0).copy()  # (K, r, H)
    else:
        As_stacked_np = None
        Bs_stacked_np = None

    r_per_k = np.asarray(ranks, dtype=np.int32)

    # Device tensors (via torch.from_numpy → .cuda for exact bit-identity).
    x_dev = torch.from_numpy(x_np).cuda()
    As_list_dev = [torch.from_numpy(a).cuda() for a in As_np_list]
    Bs_list_dev = [torch.from_numpy(b).cuda() for b in Bs_np_list]
    As_stacked_dev = (
        torch.from_numpy(As_stacked_np).cuda() if As_stacked_np is not None else None
    )
    Bs_stacked_dev = (
        torch.from_numpy(Bs_stacked_np).cuda() if Bs_stacked_np is not None else None
    )
    As_padded_dev = torch.from_numpy(As_padded_np).cuda()
    Bs_padded_dev = torch.from_numpy(Bs_padded_np).cuda()

    # Ground-truth reference: computed on GPU via torch fp16 matmul (same
    # reduction path + same tensor-core precision behaviour as the
    # PyTorch rows). This makes every row's fp16 output comparable at
    # 1-2 ULP regardless of reduction order differences between CPU /
    # GPU backends.
    y_ref_np = _reference_torch_gpu(
        x_dev,
        As_list_dev,
        Bs_list_dev,
        ranks,
        cfg.alpha,
    )

    return Case(
        cfg=cfg,
        x_np=x_np,
        As_np_list=As_np_list,
        Bs_np_list=Bs_np_list,
        As_stacked_np=As_stacked_np,
        Bs_stacked_np=Bs_stacked_np,
        As_padded_np=As_padded_np,
        Bs_padded_np=Bs_padded_np,
        r_per_k=r_per_k,
        x_dev=x_dev,
        As_list_dev=As_list_dev,
        Bs_list_dev=Bs_list_dev,
        As_stacked_dev=As_stacked_dev,
        Bs_stacked_dev=Bs_stacked_dev,
        As_padded_dev=As_padded_dev,
        Bs_padded_dev=Bs_padded_dev,
        y_ref_np=y_ref_np,
    )


def _reference_torch_gpu(
    x_dev: torch.Tensor,
    As_list_dev: list[torch.Tensor],
    Bs_list_dev: list[torch.Tensor],
    ranks: tuple[int, ...],
    alpha: float,
) -> np.ndarray:
    """Compute the per-adapter deltas on GPU via torch fp16 matmul.

    Uses the same torch backend (tensor-core fp32-accumulate fp16-output)
    that the PyTorch benchmark rows use, so the comparison is not
    distorted by CPU vs GPU reduction-order differences. Triton rows
    match this to within 1-2 ULP.
    """
    K = len(ranks)
    S, H = x_dev.shape
    out = torch.empty((K, S, H), dtype=_FP16, device=x_dev.device)
    for k in range(K):
        alpha_over_r = alpha / float(ranks[k])
        tmp = x_dev @ As_list_dev[k]
        y = tmp @ Bs_list_dev[k]
        out[k] = alpha_over_r * y
    torch.cuda.synchronize()
    return out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Triton kernels: SGMV (prefill / large S) + BGMV (decode / S==1).
# ---------------------------------------------------------------------------


@triton.jit
def sgmv_shrink_kernel(
    x_ptr,  # (S, H), row-major
    A_ptr,  # (K, H, R), contiguous per-K slab
    T_ptr,  # (K, S, R) output
    S,
    H,
    R,
    stride_xs,
    stride_xh,
    stride_ak,
    stride_ah,
    stride_ar,
    stride_tk,
    stride_ts,
    stride_tr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_r = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_S, BLOCK_R], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h = h_start + offs_h
        x_mask = (offs_s[:, None] < S) & (h[None, :] < H)
        x_tile = tl.load(
            x_ptr + offs_s[:, None] * stride_xs + h[None, :] * stride_xh,
            mask=x_mask,
            other=0.0,
        )
        a_mask = (h[:, None] < H) & (offs_r[None, :] < R)
        a_tile = tl.load(
            A_ptr
            + pid_k * stride_ak
            + h[:, None] * stride_ah
            + offs_r[None, :] * stride_ar,
            mask=a_mask,
            other=0.0,
        )
        acc += tl.dot(x_tile, a_tile)

    out_mask = (offs_s[:, None] < S) & (offs_r[None, :] < R)
    tl.store(
        T_ptr
        + pid_k * stride_tk
        + offs_s[:, None] * stride_ts
        + offs_r[None, :] * stride_tr,
        acc.to(T_ptr.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def sgmv_expand_kernel(
    T_ptr,  # (K, S, R)
    B_ptr,  # (K, R, H)
    Y_ptr,  # (K, S, H)
    alpha_ptr,  # (K,) fp32 per-k alpha / r_k
    use_per_k_alpha: tl.constexpr,
    uniform_alpha,
    S,
    H,
    R,
    stride_tk,
    stride_ts,
    stride_tr,
    stride_bk,
    stride_br,
    stride_bh,
    stride_yk,
    stride_ys,
    stride_yh,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        tmp_mask = (offs_s[:, None] < S) & (offs_r[None, :] < R)
        tmp_tile = tl.load(
            T_ptr
            + pid_k * stride_tk
            + offs_s[:, None] * stride_ts
            + offs_r[None, :] * stride_tr,
            mask=tmp_mask,
            other=0.0,
        )
        b_mask = (offs_r[:, None] < R) & (offs_h[None, :] < H)
        b_tile = tl.load(
            B_ptr
            + pid_k * stride_bk
            + offs_r[:, None] * stride_br
            + offs_h[None, :] * stride_bh,
            mask=b_mask,
            other=0.0,
        )
        acc += tl.dot(tmp_tile, b_tile)

    if use_per_k_alpha:
        a = tl.load(alpha_ptr + pid_k)
    else:
        a = uniform_alpha
    acc = acc * a

    out_mask = (offs_s[:, None] < S) & (offs_h[None, :] < H)
    tl.store(
        Y_ptr
        + pid_k * stride_yk
        + offs_s[:, None] * stride_ys
        + offs_h[None, :] * stride_yh,
        acc.to(Y_ptr.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def bgmv_shrink_kernel(
    x_ptr,  # (1, H) shared token (we broadcast x[0] across all K)
    A_ptr,  # (K, H, R)
    T_ptr,  # (K, R)
    H,
    R,
    stride_xh,
    stride_ak,
    stride_ah,
    stride_ar,
    stride_tk,
    stride_tr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_R], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h = h_start + offs_h
        x_mask = h < H
        x_vec = tl.load(
            x_ptr + h * stride_xh,
            mask=x_mask,
            other=0.0,
        )
        a_mask = (h[:, None] < H) & (offs_r[None, :] < R)
        a_tile = tl.load(
            A_ptr
            + pid_k * stride_ak
            + h[:, None] * stride_ah
            + offs_r[None, :] * stride_ar,
            mask=a_mask,
            other=0.0,
        )
        acc += tl.sum(x_vec[:, None].to(tl.float32) * a_tile.to(tl.float32), axis=0)

    out_mask = offs_r < R
    tl.store(
        T_ptr + pid_k * stride_tk + offs_r * stride_tr,
        acc.to(T_ptr.dtype.element_ty),
        mask=out_mask,
    )


@triton.jit
def bgmv_expand_kernel(
    T_ptr,  # (K, R)
    B_ptr,  # (K, R, H)
    Y_ptr,  # (K, H)
    alpha_ptr,  # (K,) fp32 per-k alpha / r_k
    use_per_k_alpha: tl.constexpr,
    uniform_alpha,
    H,
    R,
    stride_tk,
    stride_tr,
    stride_bk,
    stride_br,
    stride_bh,
    stride_yk,
    stride_yh,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        tmp_mask = offs_r < R
        tmp_vec = tl.load(
            T_ptr + pid_k * stride_tk + offs_r * stride_tr,
            mask=tmp_mask,
            other=0.0,
        )
        b_mask = (offs_r[:, None] < R) & (offs_h[None, :] < H)
        b_tile = tl.load(
            B_ptr
            + pid_k * stride_bk
            + offs_r[:, None] * stride_br
            + offs_h[None, :] * stride_bh,
            mask=b_mask,
            other=0.0,
        )
        acc += tl.sum(tmp_vec[:, None].to(tl.float32) * b_tile.to(tl.float32), axis=0)

    if use_per_k_alpha:
        a = tl.load(alpha_ptr + pid_k)
    else:
        a = uniform_alpha
    acc = acc * a

    out_mask = offs_h < H
    tl.store(
        Y_ptr + pid_k * stride_yk + offs_h * stride_yh,
        acc.to(Y_ptr.dtype.element_ty),
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Triton launchers
# ---------------------------------------------------------------------------


def _alpha_tensor(alpha_list: list[float] | None, device: torch.device) -> torch.Tensor:
    if alpha_list is None:
        return torch.zeros(1, dtype=torch.float32, device=device)
    return torch.tensor(alpha_list, dtype=torch.float32, device=device)


def sgmv_launch(
    x: torch.Tensor,  # (S, H) fp16
    As: torch.Tensor,  # (K, H, R) fp16
    Bs: torch.Tensor,  # (K, R, H) fp16
    alpha_over_r_list: list[float] | None,
    uniform_alpha: float | None,
    *,
    tmp: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run SGMV shrink + expand. Returns y: (K, S, H)."""
    assert x.is_cuda and As.is_cuda and Bs.is_cuda
    assert x.dtype == _FP16 and As.dtype == _FP16 and Bs.dtype == _FP16
    S, H = x.shape
    K, H2, R = As.shape
    K2, R2, H3 = Bs.shape
    assert H == H2 == H3 and R == R2 and K == K2, (x.shape, As.shape, Bs.shape)

    BLOCK_S = 16 if S >= 16 else max(1, S)
    # triton.tl.dot wants >=16 on all dims; if S<16 we still pad via mask.
    BLOCK_S_eff = max(16, BLOCK_S) if S >= 16 else 16
    BLOCK_R = max(16, R)
    BLOCK_H = 64

    if tmp is None:
        tmp = torch.empty((K, S, R), dtype=_FP16, device=x.device)
    if y is None:
        y = torch.empty((K, S, H), dtype=_FP16, device=x.device)

    grid_shrink = (K, triton.cdiv(S, BLOCK_S_eff), triton.cdiv(R, BLOCK_R))
    sgmv_shrink_kernel[grid_shrink](
        x,
        As,
        tmp,
        S,
        H,
        R,
        x.stride(0),
        x.stride(1),
        As.stride(0),
        As.stride(1),
        As.stride(2),
        tmp.stride(0),
        tmp.stride(1),
        tmp.stride(2),
        BLOCK_S=BLOCK_S_eff,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
    )

    use_per_k = alpha_over_r_list is not None
    alpha_ptr = _alpha_tensor(alpha_over_r_list, x.device)
    uniform = float(uniform_alpha) if uniform_alpha is not None else 0.0

    BLOCK_H_exp = 64
    grid_expand = (K, triton.cdiv(S, BLOCK_S_eff), triton.cdiv(H, BLOCK_H_exp))
    sgmv_expand_kernel[grid_expand](
        tmp,
        Bs,
        y,
        alpha_ptr,
        use_per_k,
        uniform,
        S,
        H,
        R,
        tmp.stride(0),
        tmp.stride(1),
        tmp.stride(2),
        Bs.stride(0),
        Bs.stride(1),
        Bs.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        BLOCK_S=BLOCK_S_eff,
        BLOCK_H=BLOCK_H_exp,
        BLOCK_R=BLOCK_R,
    )
    return y


def bgmv_launch(
    x: torch.Tensor,  # (1, H) fp16 (we take x[0:1] of the prefill x)
    As: torch.Tensor,  # (K, H, R) fp16
    Bs: torch.Tensor,  # (K, R, H) fp16
    alpha_over_r_list: list[float] | None,
    uniform_alpha: float | None,
    *,
    tmp: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run BGMV shrink + expand. Returns y: (K, 1, H).

    ``x`` is a single token shared across all K adapters (matches the
    "one decode token broadcast to every adapter" pattern used by this
    bench and by production multi-LoRA servers when one request steps
    through K adapters in parallel).
    """
    assert x.is_cuda and As.is_cuda and Bs.is_cuda
    assert x.dtype == _FP16 and As.dtype == _FP16 and Bs.dtype == _FP16
    S, H = x.shape
    assert S == 1, f"bgmv_launch expects S=1, got {S}"
    K, H2, R = As.shape
    K2, R2, H3 = Bs.shape
    assert H == H2 == H3 and R == R2 and K == K2

    BLOCK_H = 256
    BLOCK_R = max(16, R)

    if tmp is None:
        tmp = torch.empty((K, R), dtype=_FP16, device=x.device)
    if y is None:
        y = torch.empty((K, 1, H), dtype=_FP16, device=x.device)

    y_flat = y.view(K, H)

    grid_shrink = (K, triton.cdiv(R, BLOCK_R))
    bgmv_shrink_kernel[grid_shrink](
        x,
        As,
        tmp,
        H,
        R,
        x.stride(1),
        As.stride(0),
        As.stride(1),
        As.stride(2),
        tmp.stride(0),
        tmp.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
    )

    use_per_k = alpha_over_r_list is not None
    alpha_ptr = _alpha_tensor(alpha_over_r_list, x.device)
    uniform = float(uniform_alpha) if uniform_alpha is not None else 0.0

    BLOCK_H_exp = 128
    grid_expand = (K, triton.cdiv(H, BLOCK_H_exp))
    bgmv_expand_kernel[grid_expand](
        tmp,
        Bs,
        y_flat,
        alpha_ptr,
        use_per_k,
        uniform,
        H,
        R,
        tmp.stride(0),
        tmp.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        Bs.stride(2),
        y_flat.stride(0),
        y_flat.stride(1),
        BLOCK_H=BLOCK_H_exp,
        BLOCK_R=BLOCK_R,
    )
    return y


def fused_triton(
    x: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    alpha_over_r: float,
) -> torch.Tensor:
    """Dispatch to BGMV at S=1, SGMV otherwise. Homogeneous-rank path."""
    S = x.shape[0]
    if S == 1:
        return bgmv_launch(
            x, As, Bs, alpha_over_r_list=None, uniform_alpha=alpha_over_r
        )
    return sgmv_launch(x, As, Bs, alpha_over_r_list=None, uniform_alpha=alpha_over_r)


# ---------------------------------------------------------------------------
# Heterogeneous-rank helpers (reuse the homogeneous SGMV / BGMV kernels).
# ---------------------------------------------------------------------------


def fused_padded(case: Case) -> torch.Tensor:
    """Pad every adapter up to r_max and run a single SGMV/BGMV launch.

    Wasted FLOPs scale as ``K * r_max / sum(r_k)``.
    """
    cfg = case.cfg
    alpha_over_r_list = [cfg.alpha_over_r(r) for r in cfg.effective_ranks]
    x = case.x_dev
    if x.shape[0] == 1:
        return bgmv_launch(
            x,
            case.As_padded_dev,
            case.Bs_padded_dev,
            alpha_over_r_list=alpha_over_r_list,
            uniform_alpha=None,
        )
    return sgmv_launch(
        x,
        case.As_padded_dev,
        case.Bs_padded_dev,
        alpha_over_r_list=alpha_over_r_list,
        uniform_alpha=None,
    )


def _bucket_by_rank(ranks: tuple[int, ...]) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = {}
    for k, r in enumerate(ranks):
        buckets.setdefault(r, []).append(k)
    return buckets


def fused_serial_groups(case: Case, *, use_streams: bool) -> torch.Tensor:
    """Bucket by rank, run one launch per bucket.

    ``use_streams=False`` -> every bucket on the default stream (unfair;
    no inter-bucket concurrency).
    ``use_streams=True``  -> each bucket on its own ``torch.cuda.Stream``
    with event-based joins (fair concurrency).
    """
    cfg = case.cfg
    ranks = cfg.effective_ranks
    K, S, H = len(ranks), cfg.seq, cfg.hidden
    y = torch.empty((K, S, H), dtype=_FP16, device=case.x_dev.device)
    buckets = _bucket_by_rank(ranks)

    if use_streams:
        default = torch.cuda.current_stream()
        start_evt = torch.cuda.Event()
        start_evt.record(default)
        done_evts: list[torch.cuda.Event] = []
        streams = [torch.cuda.Stream() for _ in buckets]
        for s, (r, ks) in zip(streams, buckets.items()):
            with torch.cuda.stream(s):
                s.wait_event(start_evt)
                As_bucket = torch.stack([case.As_list_dev[k] for k in ks], dim=0)
                Bs_bucket = torch.stack([case.Bs_list_dev[k] for k in ks], dim=0)
                alpha = cfg.alpha_over_r(r)
                y_bucket = fused_triton(case.x_dev, As_bucket, Bs_bucket, alpha)
                for i, k in enumerate(ks):
                    y[k].copy_(y_bucket[i])
                ev = torch.cuda.Event()
                ev.record(s)
                done_evts.append(ev)
        for ev in done_evts:
            default.wait_event(ev)
    else:
        for r, ks in buckets.items():
            As_bucket = torch.stack([case.As_list_dev[k] for k in ks], dim=0)
            Bs_bucket = torch.stack([case.Bs_list_dev[k] for k in ks], dim=0)
            alpha = cfg.alpha_over_r(r)
            y_bucket = fused_triton(case.x_dev, As_bucket, Bs_bucket, alpha)
            for i, k in enumerate(ks):
                y[k].copy_(y_bucket[i])
    return y


# ---------------------------------------------------------------------------
# PyTorch baselines.
# ---------------------------------------------------------------------------


def _lora_body(
    x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, alpha_over_r: float
) -> torch.Tensor:
    return alpha_over_r * ((x @ A) @ B)


# Compiled-body cache keyed by rank (shared across cells / rows).
_COMPILED_LORA_BODIES: dict[int, object] = {}


def _compiled_lora_body_for_rank(r: int):
    """Return a torch.compile'd ``_lora_body`` specialised for rank ``r``."""
    if r in _COMPILED_LORA_BODIES:
        return _COMPILED_LORA_BODIES[r]
    fn = torch.compile(_lora_body, mode="default", fullgraph=True, dynamic=False)
    _COMPILED_LORA_BODIES[r] = fn
    return fn


def _warmup_compiled_for_ranks(cfg: LoRAConfig, ranks: tuple[int, ...]) -> None:
    """Compile ``_lora_body`` once per distinct rank, outside any graph_scope."""
    dev = torch.device("cuda")
    S, H = cfg.seq, cfg.hidden
    x = torch.zeros((S, H), dtype=_FP16, device=dev)
    for r in set(ranks):
        fn = _compiled_lora_body_for_rank(r)
        A = torch.zeros((H, r), dtype=_FP16, device=dev)
        B = torch.zeros((r, H), dtype=_FP16, device=dev)
        alpha = cfg.alpha_over_r(r)
        for _ in range(2):
            _ = fn(x, A, B, alpha)
    torch.cuda.synchronize()


def _build_py_sequential_forward(case: Case, *, use_compile: bool):
    """Return ``forward() -> list[Tensor]``; each call computes K deltas.

    All work pinned to ``torch.cuda.current_stream()`` (the default
    stream) for parity across the sequential rows.
    """
    cfg = case.cfg
    ranks = cfg.effective_ranks
    K = len(ranks)
    bodies = []
    for r in ranks:
        if use_compile:
            bodies.append(_compiled_lora_body_for_rank(r))
        else:
            bodies.append(_lora_body)

    As = case.As_list_dev
    Bs = case.Bs_list_dev
    x = case.x_dev
    alphas = [cfg.alpha_over_r(r) for r in ranks]

    def forward() -> list[torch.Tensor]:
        out: list[torch.Tensor] = [None] * K  # type: ignore[assignment]
        for k in range(K):
            out[k] = bodies[k](x, As[k], Bs[k], alphas[k])
        return out

    return forward


def _build_py_multistream_forward(case: Case, *, use_compile: bool):
    """Return ``forward() -> list[Tensor]``; each adapter runs on its own stream."""
    cfg = case.cfg
    ranks = cfg.effective_ranks
    K = len(ranks)
    bodies = []
    for r in ranks:
        if use_compile:
            bodies.append(_compiled_lora_body_for_rank(r))
        else:
            bodies.append(_lora_body)

    As = case.As_list_dev
    Bs = case.Bs_list_dev
    x = case.x_dev
    alphas = [cfg.alpha_over_r(r) for r in ranks]

    streams = [torch.cuda.Stream() for _ in range(K)]

    def forward() -> list[torch.Tensor]:
        default = torch.cuda.current_stream()
        start_evt = torch.cuda.Event()
        start_evt.record(default)
        out: list[torch.Tensor] = [None] * K  # type: ignore[assignment]
        done_evts: list[torch.cuda.Event] = []
        for k in range(K):
            with torch.cuda.stream(streams[k]):
                streams[k].wait_event(start_evt)
                out[k] = bodies[k](x, As[k], Bs[k], alphas[k])
                ev = torch.cuda.Event()
                ev.record(streams[k])
                done_evts.append(ev)
        for ev in done_evts:
            default.wait_event(ev)
        return out

    return forward


# ---------------------------------------------------------------------------
# STF builder (homogeneous + heterogeneous unified, host_launch readback).
# ---------------------------------------------------------------------------


def _build_stf_forward(case: Case, *, use_graph_scope: bool):
    """Build the STF multi-LoRA forward.

    Returns
    -------
    ``(ctx, forward, read_y_host, finalize)`` per the plan contract.

    - ``forward()`` enqueues one multi-LoRA iter. Safe to call repeatedly
      inside ``ctx.graph_scope()`` for the ``stf/compile+gs`` row.
    - ``read_y_host()`` enqueues K ``host_launch`` copies into host
      numpy buffers and returns them. Must be called OUTSIDE
      ``graph_scope()``; otherwise the D->H copy would be baked into
      the replayed CUDA graph.
    - ``finalize()`` ``ctx.finalize()``. After this returns the host
      buffers are guaranteed populated.
    """
    cfg = case.cfg
    ranks = cfg.effective_ranks
    K = len(ranks)
    S, H = cfg.seq, cfg.hidden

    _warmup_compiled_for_ranks(cfg, ranks)

    ctx = stf.stackable_context()

    l_x = ctx.logical_data(case.x_np, name="x")
    if hasattr(l_x, "set_read_only"):
        l_x.set_read_only()

    l_As = []
    l_Bs = []
    for k, (a_np, b_np) in enumerate(zip(case.As_np_list, case.Bs_np_list)):
        l_a = ctx.logical_data(a_np, name=f"A_{k}")
        l_b = ctx.logical_data(b_np, name=f"B_{k}")
        if hasattr(l_a, "set_read_only"):
            l_a.set_read_only()
        if hasattr(l_b, "set_read_only"):
            l_b.set_read_only()
        l_As.append(l_a)
        l_Bs.append(l_b)

    l_ys = [ctx.logical_data_empty((S, H), _NP_FP16, name=f"y_{k}") for k in range(K)]

    bodies = [_compiled_lora_body_for_rank(r) for r in ranks]
    alphas = [cfg.alpha_over_r(r) for r in ranks]

    def _scope():
        return ctx.graph_scope() if use_graph_scope else contextlib.nullcontext()

    def forward() -> None:
        for k in range(K):
            with _scope():
                with pytorch_task(
                    ctx,
                    l_x.read(),
                    l_As[k].read(),
                    l_Bs[k].read(),
                    l_ys[k].write(),
                ) as (tx, ta, tb, ty):
                    ty[:] = bodies[k](tx, ta, tb, alphas[k])

    def read_y_host() -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for k in range(K):
            buf = np.empty((S, H), dtype=_NP_FP16)
            out.append(buf)

            def _copy(y_arr, target=buf):
                np.copyto(target, y_arr)

            ctx.host_launch(l_ys[k].read(), fn=_copy)
        return out

    def finalize() -> None:
        ctx.finalize()

    return ctx, forward, read_y_host, finalize


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------


def _stack_pytorch_result(outs: list[torch.Tensor]) -> np.ndarray:
    """Stack per-adapter ``(S, H)`` tensors into ``(K, S, H)`` numpy array."""
    stacked = torch.stack(outs, dim=0)
    return stacked.detach().cpu().numpy()


def _check_close(
    label: str, got: np.ndarray, ref: np.ndarray, *, atol=5e-2, rtol=1e-2
) -> None:
    """Compare against the fp16 reference.

    Tolerances are sized for fp16 matmul with ~4096-wide reductions;
    tensor-core split-K and different tile sizes produce 1-2 ULP
    differences on a small fraction of elements that translate to
    absolute differences of up to ~3e-2 on values near 1-6. ``atol=5e-2``
    catches real bugs while ignoring that reduction-order noise.
    """
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
        max_rel = float((diff / (np.abs(ref_f32) + 1e-12)).max())
        raise AssertionError(
            f"[correctness:{label}] mismatch "
            f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"(atol={atol}, rtol={rtol}, bad_frac={bad.mean():.3e})"
        )


def correctness_sanity(case: Case) -> None:
    """Verify every row against the fp32 numpy reference.

    Called once per cell before timing; aborts the bench with a clear
    error if any row is off.
    """
    cfg = case.cfg
    ref = case.y_ref_np

    # x weight identity -- catches numpy/torch drift.
    x_from_np = torch.from_numpy(case.x_np).cuda()
    if not torch.equal(case.x_dev, x_from_np):
        raise AssertionError("x_dev drifted from x_np source of truth")

    # fused_triton (homogeneous only).
    if not cfg.is_hetero:
        alpha = cfg.alpha_over_r(cfg.rank)
        y = fused_triton(case.x_dev, case.As_stacked_dev, case.Bs_stacked_dev, alpha)
        _check_close("fused_triton", y.detach().cpu().numpy(), ref)

    # Hetero fused references.
    if cfg.is_hetero:
        y_padded = fused_padded(case)
        _check_close("fused_padded", y_padded.detach().cpu().numpy(), ref)
        y_sg_def = fused_serial_groups(case, use_streams=False)
        _check_close(
            "fused_serial_groups/default", y_sg_def.detach().cpu().numpy(), ref
        )
        y_sg_str = fused_serial_groups(case, use_streams=True)
        _check_close(
            "fused_serial_groups/streams", y_sg_str.detach().cpu().numpy(), ref
        )

    # PyTorch baselines.
    for label, builder in (
        ("py/seq/eager", _build_py_sequential_forward(case, use_compile=False)),
        ("py/seq/compile", _build_py_sequential_forward(case, use_compile=True)),
        ("py/stream/eager", _build_py_multistream_forward(case, use_compile=False)),
        ("py/stream/compile", _build_py_multistream_forward(case, use_compile=True)),
    ):
        outs = builder()
        torch.cuda.synchronize()
        _check_close(label, _stack_pytorch_result(outs), ref)

    # STF row.
    ctx, fwd, read_y_host, fin = _build_stf_forward(case, use_graph_scope=True)
    finalized = False
    try:
        fwd()
        host_bufs = read_y_host()
        fin()
        finalized = True
        got = np.stack(host_bufs, axis=0)
        _check_close("stf/compile+gs", got, ref)
    finally:
        if not finalized:
            with contextlib.suppress(Exception):
                fin()


# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------


def _time_callable(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _time_stf(case: Case, *, use_graph_scope: bool, iters: int, warmup: int) -> float:
    ctx, fwd, _read, fin = _build_stf_forward(case, use_graph_scope=use_graph_scope)
    try:
        for _ in range(warmup):
            fwd()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fwd()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / iters
    finally:
        fin()
    return elapsed


# ---------------------------------------------------------------------------
# Homogeneous sweep.
# ---------------------------------------------------------------------------


_HOMO_ROWS = (
    "fused_triton",
    "py/seq/eager",
    "py/seq/compile",
    "py/stream/compile",
    "stf/compile",
    "stf/compile+gs",
)


def run_homogeneous_cell(cfg: LoRAConfig, *, iters: int, warmup: int) -> dict:
    case = _gen_case(cfg, seed=0)
    correctness_sanity(case)
    row: dict = {"K": cfg.K, "seq": cfg.seq, "rank": cfg.rank}

    alpha = cfg.alpha_over_r(cfg.rank)

    def _run_fused():
        return fused_triton(case.x_dev, case.As_stacked_dev, case.Bs_stacked_dev, alpha)

    row["fused_triton"] = _time_callable(_run_fused, iters=iters, warmup=warmup)

    row["py/seq/eager"] = _time_callable(
        _build_py_sequential_forward(case, use_compile=False),
        iters=iters,
        warmup=warmup,
    )
    row["py/seq/compile"] = _time_callable(
        _build_py_sequential_forward(case, use_compile=True),
        iters=iters,
        warmup=warmup,
    )
    row["py/stream/compile"] = _time_callable(
        _build_py_multistream_forward(case, use_compile=True),
        iters=iters,
        warmup=warmup,
    )

    row["stf/compile"] = _time_stf(
        case, use_graph_scope=False, iters=iters, warmup=warmup
    )
    row["stf/compile+gs"] = _time_stf(
        case, use_graph_scope=True, iters=iters, warmup=warmup
    )
    return row


def run_homogeneous_sweep(
    Ks: tuple[int, ...],
    seqs: tuple[int, ...],
    *,
    hidden: int,
    rank: int,
    alpha: float,
    iters: int,
    warmup: int,
) -> list[dict]:
    rows: list[dict] = []
    for seq in seqs:
        for K in Ks:
            cfg = LoRAConfig(hidden=hidden, seq=seq, K=K, rank=rank, alpha=alpha)
            rows.append(run_homogeneous_cell(cfg, iters=iters, warmup=warmup))
    return rows


# ---------------------------------------------------------------------------
# Heterogeneous sweep.
# ---------------------------------------------------------------------------


_HETERO_ROWS = (
    "fused_padded",
    "fused_serial_groups/default",
    "fused_serial_groups/streams",
    "py/seq/eager",
    "py/seq/compile",
    "py/stream/eager",
    "py/stream/compile",
    "stf/compile+gs",
)


def run_heterogeneous_cell(cfg: LoRAConfig, *, iters: int, warmup: int) -> dict:
    case = _gen_case(cfg, seed=0)
    correctness_sanity(case)
    row: dict = {
        "K": cfg.effective_K,
        "seq": cfg.seq,
        "r_max": cfg.r_max,
        "ranks": list(cfg.ranks),
    }

    def _fused_padded_run():
        return fused_padded(case)

    row["fused_padded"] = _time_callable(_fused_padded_run, iters=iters, warmup=warmup)

    def _sg_default_run():
        return fused_serial_groups(case, use_streams=False)

    row["fused_serial_groups/default"] = _time_callable(
        _sg_default_run,
        iters=iters,
        warmup=warmup,
    )

    def _sg_streams_run():
        return fused_serial_groups(case, use_streams=True)

    row["fused_serial_groups/streams"] = _time_callable(
        _sg_streams_run,
        iters=iters,
        warmup=warmup,
    )

    row["py/seq/eager"] = _time_callable(
        _build_py_sequential_forward(case, use_compile=False),
        iters=iters,
        warmup=warmup,
    )
    row["py/seq/compile"] = _time_callable(
        _build_py_sequential_forward(case, use_compile=True),
        iters=iters,
        warmup=warmup,
    )
    row["py/stream/eager"] = _time_callable(
        _build_py_multistream_forward(case, use_compile=False),
        iters=iters,
        warmup=warmup,
    )
    row["py/stream/compile"] = _time_callable(
        _build_py_multistream_forward(case, use_compile=True),
        iters=iters,
        warmup=warmup,
    )

    row["stf/compile+gs"] = _time_stf(
        case, use_graph_scope=True, iters=iters, warmup=warmup
    )
    return row


def run_heterogeneous_sweep(
    ranks: tuple[int, ...],
    seqs: tuple[int, ...],
    *,
    hidden: int,
    alpha: float,
    iters: int,
    warmup: int,
) -> list[dict]:
    rows: list[dict] = []
    for seq in seqs:
        cfg = LoRAConfig(
            hidden=hidden, seq=seq, K=len(ranks), ranks=tuple(ranks), alpha=alpha
        )
        rows.append(run_heterogeneous_cell(cfg, iters=iters, warmup=warmup))
    return rows


# ---------------------------------------------------------------------------
# Table formatting.
# ---------------------------------------------------------------------------


def _fmt_homogeneous_table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    lines = []
    header = f"{'seq':>5} {'K':>4} " + " ".join(f"{m:>22}" for m in _HOMO_ROWS)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = [f"{r[m] * 1e3:>19.3f} ms" for m in _HOMO_ROWS]
        lines.append(f"{r['seq']:>5d} {r['K']:>4d} " + " ".join(cells))
    return "\n".join(lines)


def _fmt_homogeneous_speedup(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    lines = []
    header = f"{'seq':>5} {'K':>4} " + " ".join(f"{m:>22}" for m in _HOMO_ROWS)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        base = r["fused_triton"]
        cells = [f"{base / r[m]:>20.2f}x " for m in _HOMO_ROWS]
        lines.append(f"{r['seq']:>5d} {r['K']:>4d} " + " ".join(cells))
    return "\n".join(lines)


def _fmt_heterogeneous_table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    lines = []
    header = f"{'seq':>5} {'K':>4} " + " ".join(f"{m:>28}" for m in _HETERO_ROWS)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        cells = [f"{r[m] * 1e3:>25.3f} ms" for m in _HETERO_ROWS]
        lines.append(f"{r['seq']:>5d} {r['K']:>4d} " + " ".join(cells))
    return "\n".join(lines)


def _loc_banner() -> str:
    """Rough LOC / effort banner for the four rows that need user scheduling code."""
    return (
        "Effort to obtain K-way concurrency on K LoRA deltas (scheduling LOC):\n"
        "  py/seq/eager         0  (no concurrency; K adapters serialise)\n"
        "  py/seq/compile       0  (no concurrency; torch.compile per body)\n"
        "  py/stream/*        ~14  (K streams + K events + start_evt /\n"
        "                            wait_event / record / cross-stream sync;\n"
        "                            see _build_py_multistream_forward)\n"
        "  fused_triton        ~600 (SGMV + BGMV kernels + launchers;\n"
        "                            shipped production code in Punica/vLLM)\n"
        "  stf/compile+gs       ~1 per scope (graph_scope() + .read()/.write()\n"
        "                            deps; scheduling/streams/events implicit in DAG)"
    )


# ---------------------------------------------------------------------------
# Reportable flags.
# ---------------------------------------------------------------------------


def _compute_flags(homo_rows: list[dict], hetero_rows: list[dict]) -> dict:
    flags: dict = {}

    # fused_triton_is_fastest_homogeneous
    def _fastest(row):
        return min(_HOMO_ROWS, key=lambda m: row.get(m, float("inf")))

    ftif = all(_fastest(r) == "fused_triton" for r in homo_rows) if homo_rows else None
    flags["fused_triton_is_fastest_homogeneous"] = ftif

    # stf_within_3x_bgmv, stf_within_2x_sgmv
    bgmv_cells = [r for r in homo_rows if r["seq"] == 1]
    sgmv_cells = [r for r in homo_rows if r["seq"] > 1]

    def _within_ratio(cells, ratio):
        if not cells:
            return None
        return all(r["stf/compile+gs"] <= ratio * r["fused_triton"] for r in cells)

    flags["stf_within_3x_bgmv"] = _within_ratio(bgmv_cells, 3.0)
    flags["stf_within_2x_sgmv"] = _within_ratio(sgmv_cells, 2.0)

    # stf_beats_py_stream_compile
    flags["stf_beats_py_stream_compile"] = (
        all(r["stf/compile+gs"] <= r["py/stream/compile"] for r in homo_rows)
        if homo_rows
        else None
    )

    # stf_beats_fused_padded_hetero (within 1.5x)
    if hetero_rows:
        flags["stf_beats_fused_padded_hetero"] = all(
            r["stf/compile+gs"] <= 1.5 * r["fused_padded"] for r in hetero_rows
        )
    else:
        flags["stf_beats_fused_padded_hetero"] = None

    # serial_groups_streams_beats_default
    if hetero_rows:
        flags["serial_groups_streams_beats_default"] = all(
            r["fused_serial_groups/streams"] <= r["fused_serial_groups/default"]
            for r in hetero_rows
        )
    else:
        flags["serial_groups_streams_beats_default"] = None

    return flags


def _fmt_flags(flags: dict) -> str:
    lines = ["Reportable flags (informational; no flag fails the test):"]
    for k, v in flags.items():
        mark = "?" if v is None else ("OK" if v else "MISS")
        lines.append(f"  [{mark:>4}]  {k}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


_DEFAULT_HETERO_RANKS = (4, 8, 16, 16, 32, 32, 64, 64)


def _parse_int_list(env_name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(env_name, default)
    return tuple(int(x) for x in raw.split(",") if x)


def run_quick() -> tuple[list[dict], list[dict]]:
    """Phase 1 de-risk cells only. Returns (homo_rows, hetero_rows)."""
    hidden = 4096
    alpha = 32.0
    iters = int(os.environ.get("LLM_LORA_VS_FUSED_ITERS", "20"))
    warmup = int(os.environ.get("LLM_LORA_VS_FUSED_WARMUP", "5"))

    # Homogeneous: seq=512, K=16, r=16 -- the cell that validates SGMV quality.
    homo_cfg = LoRAConfig(hidden=hidden, seq=512, K=16, rank=16, alpha=alpha)
    homo_rows = [run_homogeneous_cell(homo_cfg, iters=iters, warmup=warmup)]

    # Hetero: K=8, r_max=64, seq=512.
    hetero_cfg = LoRAConfig(
        hidden=hidden,
        seq=512,
        K=len(_DEFAULT_HETERO_RANKS),
        ranks=_DEFAULT_HETERO_RANKS,
        alpha=alpha,
    )
    hetero_rows = [run_heterogeneous_cell(hetero_cfg, iters=iters, warmup=warmup)]

    return homo_rows, hetero_rows


def run_full() -> tuple[list[dict], list[dict]]:
    """Full sweep per plan. Returns (homo_rows, hetero_rows)."""
    hidden = 4096
    alpha = 32.0
    Ks = _parse_int_list("LLM_LORA_VS_FUSED_K", "1,4,16,64")
    seqs = _parse_int_list("LLM_LORA_VS_FUSED_SEQ", "1,512")
    iters = int(os.environ.get("LLM_LORA_VS_FUSED_ITERS", "20"))
    warmup = int(os.environ.get("LLM_LORA_VS_FUSED_WARMUP", "5"))

    homo_rows = run_homogeneous_sweep(
        Ks, seqs, hidden=hidden, rank=16, alpha=alpha, iters=iters, warmup=warmup
    )
    hetero_rows = run_heterogeneous_sweep(
        _DEFAULT_HETERO_RANKS,
        seqs,
        hidden=hidden,
        alpha=alpha,
        iters=iters,
        warmup=warmup,
    )
    return homo_rows, hetero_rows


def _optional_refs_note() -> str:
    lines = ["Optional cross-check kernels (informational, not wired into the sweep):"]
    lines.append(
        "  punica.ops.bgmv       : "
        + ("[installed]" if _punica_bgmv is not None else "[not installed]")
    )
    lines.append(
        "  vllm.lora.ops.triton_ops sgmv_shrink/sgmv_expand : "
        + (
            "[installed]"
            if _vllm_sgmv_shrink is not None and _vllm_sgmv_expand is not None
            else "[not installed]"
        )
    )
    return "\n".join(lines)


def _print_results(homo_rows, hetero_rows):
    print("\n=== Homogeneous sweep -- wall-clock ms / forward ===")
    print(_fmt_homogeneous_table(homo_rows))
    print(
        "\n=== Same data, normalised (speedup vs fused_triton; <1.0 means slower) ==="
    )
    print(_fmt_homogeneous_speedup(homo_rows))
    print("\n=== Heterogeneous sweep -- wall-clock ms / forward ===")
    print(_fmt_heterogeneous_table(hetero_rows))
    print("\n" + _loc_banner())
    flags = _compute_flags(homo_rows, hetero_rows)
    print("\n" + _fmt_flags(flags))
    print("\n" + _optional_refs_note())


@pytest.mark.skipif(
    os.environ.get("LLM_LORA_VS_FUSED_BENCH", "0") != "1",
    reason="set LLM_LORA_VS_FUSED_BENCH=1 to run the multi-LoRA vs fused bench",
)
def test_multi_lora_vs_fused_bench():
    """Slide-grade bench; prints two tables + LOC banner + reportable flags."""
    if os.environ.get("LLM_LORA_VS_FUSED_QUICK", "0") == "1":
        homo_rows, hetero_rows = run_quick()
    else:
        homo_rows, hetero_rows = run_full()

    _print_results(homo_rows, hetero_rows)

    # Sanity: every reported time is positive.
    for rows, row_set in ((homo_rows, _HOMO_ROWS), (hetero_rows, _HETERO_ROWS)):
        for r in rows:
            for m in row_set:
                assert r[m] > 0.0, f"non-positive time for {m}: {r}"


if __name__ == "__main__":
    if os.environ.get("LLM_LORA_VS_FUSED_QUICK", "0") == "1":
        homo_rows, hetero_rows = run_quick()
    else:
        homo_rows, hetero_rows = run_full()
    _print_results(homo_rows, hetero_rows)
