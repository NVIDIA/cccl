# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Shared building blocks for the LLM STF demos.

Not a test module. Provides:
  - ``MiniGPTConfig``, ``TINY`` and ``DRAFT`` presets.
  - ``build_random_weights(ctx, cfg, ...)`` that allocates all weights as
    named STF logical data so they show up labelled in DOT graphs.
  - STF-wrapped primitive ops (layernorm, linear, fused FFN, SDPA attention,
    per-head "parallel" attention, transformer block, greedy sampler).
  - ``spec_decode_loop(...)`` — the speculative-decoding scaffold used by
    both ``test_llm_speculative.py`` and ``test_llm_speculative_compiled.py``.

Everything is expressed via ``pytorch_task`` to keep each helper short enough
to fit on a presentation slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pytorch_task import pytorch_task  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiniGPTConfig:
    """nanoGPT-scale transformer config used by all LLM demos."""

    hidden: int
    heads: int
    head_dim: int
    ffn_mult: int
    n_layers: int
    seq: int
    vocab: int
    dtype: str = "float32"  # "float32" or "float16"

    @property
    def ffn_hidden(self) -> int:
        return self.hidden * self.ffn_mult

    @property
    def np_dtype(self):
        return np.float32 if self.dtype == "float32" else np.float16

    @property
    def torch_dtype(self):
        return torch.float32 if self.dtype == "float32" else torch.float16


# Preset used everywhere except speculative-decoding draft.
TINY = MiniGPTConfig(
    hidden=384,
    heads=6,
    head_dim=64,
    ffn_mult=4,
    n_layers=6,
    seq=256,
    vocab=1024,
)

# Smaller 2-layer draft used in speculative decoding.
DRAFT = MiniGPTConfig(
    hidden=384,
    heads=6,
    head_dim=64,
    ffn_mult=4,
    n_layers=2,
    seq=256,
    vocab=1024,
)


# ---------------------------------------------------------------------------
# Weight initialization (as STF logical data)
# ---------------------------------------------------------------------------


def _init_random(ctx, ld, seed_idx: int, fan_in: int):
    """Fill a ``logical_data_empty`` with a deterministic pseudo-random pattern.

    Runs entirely on device. Avoids host-to-device transfers, avoids torch's
    RNG (whose state is not graph-replay-safe), and still yields different
    values per weight via ``seed_idx``.
    """
    std = 1.0 / float(np.sqrt(max(1, fan_in)))
    with pytorch_task(ctx, ld.write()) as (tw,):
        idx = torch.arange(tw.numel(), device=tw.device, dtype=torch.float32)
        # Well-mixed non-periodic pattern: each weight gets a distinct frequency
        # + phase via the prime-ish seed_idx offsets.
        k = 0.0131 + 7.1e-5 * float(seed_idx)
        phi = 0.71 * float(seed_idx)
        vals = torch.sin(idx * k + phi) + 0.3 * torch.cos(idx * (3.7 * k) + 2 * phi)
        tw.copy_((vals * std).view(tw.shape).to(tw.dtype))


def _init_fill(ctx, ld, value: float):
    with pytorch_task(ctx, ld.write()) as (tw,):
        tw.fill_(float(value))


def build_random_weights(
    ctx,
    cfg: MiniGPTConfig,
    *,
    seed: int = 0,
    read_only: bool = True,
    share_emb_lm_head_from: "dict | None" = None,
):
    """Build named STF logical data for embeddings, layers, and lm_head.

    Weights are allocated via ``logical_data_empty`` (device-only memory) and
    filled on device by ``_init_random`` / ``_init_fill``. This works
    uniformly across ``stf.context()``, ``stf.context(use_graph=True)`` and
    ``stf.stackable_context()`` — no host-backed numpy buffers to coerce.

    Returns a dict:

        weights["emb"]          : (vocab, hidden)
        weights["lm_head"]      : (hidden, vocab)
        weights["layers"][i]    : dict with Wq/Wk/Wv/Wo/W_up/b_up/W_down/b_down
                                  + ln1_gamma/ln1_beta + ln2_gamma/ln2_beta

    If ``share_emb_lm_head_from`` is provided, ``emb`` and ``lm_head`` are
    reused from that weights dict — the convention for spec-decoding demos
    where draft and target share tokenizer + lm head.

    ``read_only=True`` marks weights via ``set_read_only()`` so they can be
    shared across graph replays (only matters in ``stackable_context``).
    """
    dtype = cfg.np_dtype
    weights: dict = {"layers": []}

    def _mark_ro(ld):
        if read_only and hasattr(ld, "set_read_only"):
            ld.set_read_only()

    def _alloc_random(name, shape, fan_in, sidx):
        ld = ctx.logical_data_empty(shape, dtype, name=name)
        _init_random(ctx, ld, seed_idx=sidx, fan_in=fan_in)
        return ld

    def _alloc_const(name, shape, value):
        ld = ctx.logical_data_empty(shape, dtype, name=name)
        _init_fill(ctx, ld, value)
        return ld

    sidx = seed * 1000  # keep seeds disjoint between target / draft builds

    # Token embedding + lm_head
    if share_emb_lm_head_from is not None:
        weights["emb"] = share_emb_lm_head_from["emb"]
        weights["lm_head"] = share_emb_lm_head_from["lm_head"]
    else:
        weights["emb"] = _alloc_random(
            "emb", (cfg.vocab, cfg.hidden), cfg.vocab, sidx + 1
        )
        weights["lm_head"] = _alloc_random(
            "lm_head", (cfg.hidden, cfg.vocab), cfg.hidden, sidx + 2
        )
        _mark_ro(weights["emb"])
        _mark_ro(weights["lm_head"])

    H, Fh = cfg.hidden, cfg.ffn_hidden
    for i in range(cfg.n_layers):
        base = sidx + 100 + i * 32
        layer = {
            "ln1_gamma": _alloc_const(f"L{i}.ln1_g", (H,), 1.0),
            "ln1_beta": _alloc_const(f"L{i}.ln1_b", (H,), 0.0),
            "Wq": _alloc_random(f"L{i}.Wq", (H, H), H, base + 1),
            "Wk": _alloc_random(f"L{i}.Wk", (H, H), H, base + 2),
            "Wv": _alloc_random(f"L{i}.Wv", (H, H), H, base + 3),
            "Wo": _alloc_random(f"L{i}.Wo", (H, H), H, base + 4),
            "ln2_gamma": _alloc_const(f"L{i}.ln2_g", (H,), 1.0),
            "ln2_beta": _alloc_const(f"L{i}.ln2_b", (H,), 0.0),
            "W_up": _alloc_random(f"L{i}.Wup", (H, Fh), H, base + 5),
            "b_up": _alloc_const(f"L{i}.bup", (Fh,), 0.0),
            "W_down": _alloc_random(f"L{i}.Wdn", (Fh, H), Fh, base + 6),
            "b_down": _alloc_const(f"L{i}.bdn", (H,), 0.0),
        }

        for ld in layer.values():
            _mark_ro(ld)

        weights["layers"].append(layer)

    return weights


# ---------------------------------------------------------------------------
# STF-wrapped primitive ops
# ---------------------------------------------------------------------------


def stf_layernorm(ctx, l_x, l_gamma, l_beta, l_out):
    """out = LayerNorm(x) * gamma + beta."""
    with pytorch_task(
        ctx, l_x.read(), l_gamma.read(), l_beta.read(), l_out.write()
    ) as (tx, tg, tb, to):
        to[:] = torch.nn.functional.layer_norm(tx, tg.shape, tg, tb, eps=1e-5)


def stf_linear(ctx, l_x, l_W, l_b, l_out):
    """out = x @ W (+ b if provided)."""
    if l_b is None:
        with pytorch_task(ctx, l_x.read(), l_W.read(), l_out.write()) as (tx, tw, to):
            to[:] = torch.matmul(tx, tw)
    else:
        with pytorch_task(ctx, l_x.read(), l_W.read(), l_b.read(), l_out.write()) as (
            tx,
            tw,
            tb,
            to,
        ):
            to[:] = torch.matmul(tx, tw) + tb


def stf_ffn_fused(ctx, l_x, l_Wup, l_bup, l_Wdn, l_bdn, l_out):
    """Fused FFN: out = Wdown(gelu(Wup(x) + bup)) + bdown — one STF task."""
    with pytorch_task(
        ctx,
        l_x.read(),
        l_Wup.read(),
        l_bup.read(),
        l_Wdn.read(),
        l_bdn.read(),
        l_out.write(),
    ) as (tx, twu, tbu, twd, tbd, to):
        h = torch.nn.functional.gelu(torch.matmul(tx, twu) + tbu)
        to[:] = torch.matmul(h, twd) + tbd


def stf_attention_sdpa(ctx, l_Q, l_K, l_V, l_out, cfg, *, is_causal=True):
    """Single-task SDPA attention (flash / mem-efficient backends)."""
    H, D = cfg.heads, cfg.head_dim
    with pytorch_task(ctx, l_Q.read(), l_K.read(), l_V.read(), l_out.write()) as (
        tq,
        tk,
        tv,
        to,
    ):
        b, s, _ = tq.shape
        q = tq.view(b, s, H, D).transpose(1, 2).contiguous()
        k = tk.view(b, s, H, D).transpose(1, 2).contiguous()
        v = tv.view(b, s, H, D).transpose(1, 2).contiguous()
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )
        to[:] = out.transpose(1, 2).contiguous().view(b, s, cfg.hidden)


def stf_attention_parallel_heads(ctx, l_Q, l_K, l_V, l_out, cfg, *, is_causal=True):
    """Per-head attention as H parallel STF tasks, used only for DAG viz.

    Each head is a separate ``pytorch_task`` whose only dependency is a small
    view of Q/K/V — from STF's perspective these tasks have no data conflict
    on their outputs, so they show up as H parallel branches in the DOT graph.
    """
    H, D = cfg.heads, cfg.head_dim
    B, S = l_Q.shape[0], l_Q.shape[1]

    head_outs = []
    for h_idx in range(H):
        l_head = ctx.logical_data_empty(
            (B, S, D), dtype=cfg.np_dtype, name=f"head{h_idx}_out"
        )
        with pytorch_task(ctx, l_Q.read(), l_K.read(), l_V.read(), l_head.write()) as (
            tq,
            tk,
            tv,
            th,
        ):
            b, s, _ = tq.shape
            q = tq.view(b, s, H, D)[:, :, h_idx, :]
            k = tk.view(b, s, H, D)[:, :, h_idx, :]
            v = tv.view(b, s, H, D)[:, :, h_idx, :]
            scores = torch.matmul(q, k.transpose(-1, -2)) / (D**0.5)
            if is_causal:
                mask = torch.triu(
                    torch.ones(s, s, device=scores.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            th[:] = torch.matmul(torch.softmax(scores, dim=-1), v)
        head_outs.append(l_head)

    # Final concat — one task with H reads + 1 write.
    deps = [l_out.write()] + [h.read() for h in head_outs]
    with pytorch_task(ctx, *deps) as tensors:
        to = tensors[0]
        th_list = tensors[1:]
        to[:] = torch.cat(list(th_list), dim=-1)


def make_block_scratches(ctx, cfg: MiniGPTConfig, *, tag: str) -> dict:
    """Allocate one shared scratch pool for a model.

    Returns a dict with the per-layer scratch (``xn``, ``Q``, ``K``, ``V``,
    ``attn``, ``proj``, ``x1``, ``x1n``, ``ffn``) plus an ``h_inter`` list of
    ``n_layers - 1`` stack-internal rolling buffers. All logical data are
    tagged with ``tag.*`` names so target and draft pools don't collide in
    DOT output.

    Concurrency note: share scratches only among operations that are already
    strictly serial on the data plane. Successive layers in a stack are
    serial (layer i+1 depends on layer i's output), and K sequential draft
    forwards are serial (each feeds the next via l_hidden). Target and draft
    models must therefore get DISJOINT pools, otherwise STF will serialize
    their forwards and you lose target∥draft overlap.
    """
    B, S, H = 1, cfg.seq, cfg.hidden
    dtype = cfg.np_dtype

    def _alloc(name, shape=(B, S, H)):
        return ctx.logical_data_empty(shape, dtype, name=f"{tag}.{name}")

    return {
        "xn": _alloc("xn"),
        "Q": _alloc("Q"),
        "K": _alloc("K"),
        "V": _alloc("V"),
        "attn": _alloc("attn"),
        "proj": _alloc("proj"),
        "x1": _alloc("x1"),
        "x1n": _alloc("x1n"),
        "ffn": _alloc("ffn_out"),
        # One dedicated "next hidden" buffer. The caller can use it as the
        # stack's output and then copy back into the real hidden — keeping
        # the forward's output (and the first-layer's input) on distinct
        # logical data avoids edge cases where STF's dep tracking sees an
        # aliased in/out across a deep chain of intermediates.
        "h_next": _alloc("h_next"),
        "h_inter": [_alloc(f"h{i + 1}") for i in range(cfg.n_layers - 1)],
    }


def stf_transformer_block(
    ctx, l_x, layer_w, l_out, cfg, *, scratches=None, attention="sdpa"
):
    """ln1 -> {Wq, Wk, Wv} -> attn -> Wo + residual -> ln2 -> ffn + residual.

    If ``scratches`` is provided, reuses its entries (``xn/Q/K/V/attn/proj/
    x1/x1n/ffn``). Otherwise allocates fresh ``logical_data_empty`` per call
    — the original behavior, used by contexts where captured-graph size is
    not a constraint (e.g. eager ``stf.context()``) or where scratch reuse
    empirically interferes with STF's dep tracking (e.g. the K-serial
    draft forwards inside ``spec_decode_loop``).
    """
    if scratches is None:
        B, S, H = l_x.shape[0], l_x.shape[1], cfg.hidden
        dtype = cfg.np_dtype
        scratches = {
            "xn": ctx.logical_data_empty((B, S, H), dtype, name="xn"),
            "Q": ctx.logical_data_empty((B, S, H), dtype, name="Q"),
            "K": ctx.logical_data_empty((B, S, H), dtype, name="K"),
            "V": ctx.logical_data_empty((B, S, H), dtype, name="V"),
            "attn": ctx.logical_data_empty((B, S, H), dtype, name="attn"),
            "proj": ctx.logical_data_empty((B, S, H), dtype, name="proj"),
            "x1": ctx.logical_data_empty((B, S, H), dtype, name="x1"),
            "x1n": ctx.logical_data_empty((B, S, H), dtype, name="x1n"),
            "ffn": ctx.logical_data_empty((B, S, H), dtype, name="ffn_out"),
        }
    s = scratches

    stf_layernorm(ctx, l_x, layer_w["ln1_gamma"], layer_w["ln1_beta"], s["xn"])

    # Three parallel projections
    stf_linear(ctx, s["xn"], layer_w["Wq"], None, s["Q"])
    stf_linear(ctx, s["xn"], layer_w["Wk"], None, s["K"])
    stf_linear(ctx, s["xn"], layer_w["Wv"], None, s["V"])

    if attention == "sdpa":
        stf_attention_sdpa(ctx, s["Q"], s["K"], s["V"], s["attn"], cfg)
    elif attention == "parallel_heads":
        stf_attention_parallel_heads(ctx, s["Q"], s["K"], s["V"], s["attn"], cfg)
    else:
        raise ValueError(f"unknown attention mode: {attention!r}")

    stf_linear(ctx, s["attn"], layer_w["Wo"], None, s["proj"])

    # Residual 1
    with pytorch_task(ctx, l_x.read(), s["proj"].read(), s["x1"].write()) as (
        tx,
        tp,
        tx1,
    ):
        tx1[:] = tx + tp

    stf_layernorm(ctx, s["x1"], layer_w["ln2_gamma"], layer_w["ln2_beta"], s["x1n"])

    stf_ffn_fused(
        ctx,
        s["x1n"],
        layer_w["W_up"],
        layer_w["b_up"],
        layer_w["W_down"],
        layer_w["b_down"],
        s["ffn"],
    )

    # Residual 2
    with pytorch_task(ctx, s["x1"].read(), s["ffn"].read(), l_out.write()) as (
        tx1,
        tf,
        to,
    ):
        to[:] = tx1 + tf


def stf_transformer_stack(
    ctx, l_hidden_in, weights, cfg, l_hidden_out, *, scratches=None
):
    """Run all n_layers blocks.

    If ``scratches`` is provided, uses ``scratches['h_inter']`` as the
    rolling inter-layer buffers and shares the block-level scratch across
    layers. Otherwise allocates fresh logical data per block (original
    behavior).
    """
    assert cfg.n_layers >= 1
    B, S, H = 1, cfg.seq, cfg.hidden
    dtype = cfg.np_dtype
    cur = l_hidden_in
    for i, layer_w in enumerate(weights["layers"]):
        if i == cfg.n_layers - 1:
            nxt = l_hidden_out
        elif scratches is not None:
            nxt = scratches["h_inter"][i]
        else:
            nxt = ctx.logical_data_empty((B, S, H), dtype, name=f"h{i + 1}")
        stf_transformer_block(ctx, cur, layer_w, nxt, cfg, scratches=scratches)
        cur = nxt


def stf_lm_head(ctx, l_hidden, l_lm_head, l_logits):
    """logits = hidden @ lm_head_weight."""
    with pytorch_task(ctx, l_hidden.read(), l_lm_head.read(), l_logits.write()) as (
        th,
        tw,
        tl,
    ):
        tl[:] = torch.matmul(th, tw)


def stf_sample_argmax_last(ctx, l_logits, l_next_token):
    """next_token = argmax(logits[:, -1, :]) — greedy sampler on last position."""
    with pytorch_task(ctx, l_logits.read(), l_next_token.write()) as (tl, tn):
        picked = tl[:, -1, :].argmax(dim=-1)
        tn.copy_(picked.to(tn.dtype).view(tn.shape))


def stf_append_token_hidden(ctx, l_hidden, l_emb_weight, l_next_token):
    """Update the last-position hidden slot with the embedding of next_token.

    Used by the simplified decode loop: hidden[:, -1, :] = emb[next_token].
    """
    with pytorch_task(
        ctx,
        l_hidden.rw(),
        l_emb_weight.read(),
        l_next_token.read(),
    ) as (th, tw, tn):
        idx = tn.to(torch.long).view(-1)  # (B,)
        # (B, H) row of emb for each batch item
        vec = torch.nn.functional.embedding(idx, tw)
        th[:, -1, :] = vec


def _logical_data_empty_no_export(ctx, shape, dtype, *, name=None):
    """``ctx.logical_data_empty(..., no_export=True)`` with a graceful
    fallback for non-stackable contexts.

    ``no_export=True`` is only honored by ``stackable_context``. On plain
    ``stf.context()`` / ``stf.context(use_graph=True)`` the kwarg doesn't
    exist and the scope-local behavior is implicit anyway (the whole
    program is one flat scope). This wrapper makes the LLM helpers portable
    across all three context kinds.
    """
    try:
        return ctx.logical_data_empty(shape, dtype, name=name, no_export=True)
    except TypeError:
        return ctx.logical_data_empty(shape, dtype, name=name)


def make_cond_scratch(ctx, *, name: str = "cond_scratch"):
    """Allocate the scalar ``float64`` scratch buffer that the while-loop
    continuation helper requires.

    Uses ``logical_data_empty(no_export=True)`` so the buffer is local to
    the current (parent) scope and never leaks into child graphs /
    replays. The scratch is only ever accessed as ``.write()`` inside the
    helper (the task writes it first, then reads that fresh write), so no
    host-seeded initial value is needed.

    Falls back to a host-seeded ``logical_data`` when ``no_export`` is not
    supported by the binding (older builds) or when ``ctx`` is not a
    stackable context.
    """
    return _logical_data_empty_no_export(ctx, (1,), np.float64, name=name)


def stf_advance_counter_flag(
    ctx,
    l_counter,
    l_flag,
    max_value: float,
    *,
    scratch,
):
    """``counter += 1 ; flag = (counter < max_value) ? 1 : 0`` — the standard
    ``while_loop`` continuation pattern.

    Workaround note
    ---------------
    The "obvious" one-task implementation is::

        with pytorch_task(ctx, l_counter.rw(), l_flag.write()) as (tc, tf):
            tc[:] = tc + 1.0
            tf.copy_((tc < max_value).to(tf.dtype))

    but that form triggers a mod-4 miscount when used inside a
    ``stackable_context.while_loop`` body: whenever the captured body
    contains K chained ``.rw()`` tasks on a persistent logical_data with
    ``K % 4 == 0`` in addition to this counter task, exactly one of the K
    updates is silently dropped. The bug has been pinned to the PyTorch
    caching allocator allocating a transient tensor (for the cast /
    comparison result) on a stream that is simultaneously under STF's
    while-graph capture — reproducer in
    ``tests/stf/probe_k_sweep_torch_variants.py`` (see mode
    ``r_single_cast_scratch`` for the passing baseline).

    The workaround is to sink the transient cast into an STF-owned scratch
    buffer so PyTorch never allocates inside the captured body. Pass a
    scratch obtained from :func:`make_cond_scratch` (which uses
    ``logical_data_empty(no_export=True)`` so the buffer stays local to
    the current scope). The scratch must be allocated **outside** the
    ``while_loop`` and accessed here as ``.write()`` — the task writes it
    first, then reads that fresh write, so no initial value is needed.

    ``l_flag`` keeps ``float64`` semantics (same shape/dtype as the caller
    passes) so ``ctx.while_loop`` / ``loop.continue_while`` can use its
    existing ``>`` 0.5 predicate.
    """
    # Every op below must be either (a) pure in-place on an STF-owned
    # buffer (``add_``) or (b) routed through the STF-owned ``scratch``.
    # Any PyTorch temporary whose result is then written into an STF
    # buffer reactivates the mod-4 bug.
    with pytorch_task(ctx, l_counter.rw(), scratch.write(), l_flag.write()) as (
        tc,
        tsc,
        tf,
    ):
        tc.add_(1.0)
        tsc[:] = (tc < float(max_value)).to(tsc.dtype)
        tf.copy_(tsc.view(tf.shape))


def validate_forward(host_out: np.ndarray, cfg: MiniGPTConfig) -> float:
    """Host-side sanity checks on a forward-pass hidden state."""
    assert not np.any(np.isnan(host_out)), "NaN in output"
    assert not np.any(np.isinf(host_out)), "Inf in output"
    assert host_out.shape[-1] == cfg.hidden, (
        f"hidden-dim mismatch: {host_out.shape} vs hidden={cfg.hidden}"
    )
    var = float(np.var(host_out.astype(np.float64)))
    assert 1e-6 < var < 1e6, f"variance out of healthy range: {var:.3e}"
    return var


# ---------------------------------------------------------------------------
# Speculative-decoding scaffold (shared by Layer D and D')
# ---------------------------------------------------------------------------


def spec_decode_loop(
    ctx,
    *,
    cfg_target: MiniGPTConfig,
    cfg_draft: MiniGPTConfig,
    target_forward: Callable,
    draft_forward: Callable,
    l_hidden_target,
    l_hidden_draft,
    l_emb,
    l_draft_tokens,
    l_target_tokens,
    l_accepted,
    l_round,
    l_done,
    l_cond_scratch,
    K: int,
    max_rounds: int,
):
    """Speculative-decoding body inside a ``stackable_context``'s ``while_loop``.

    Per outer iteration (one "spec round"):
      1) Draft proposes K tokens sequentially, feeding each back into its own
         hidden buffer via ``emb[next_tok]`` at the last slot.
      2) Target runs one forward on its own hidden buffer and produces an
         argmax sequence at the last K+1 positions (verifier tokens + bonus).
      3) Accept/reject by longest-common-prefix. ``accepted`` is accumulated
         into ``l_accepted`` for host-side reporting. The bonus token at the
         fixed position ``K`` is always emitted as ``final_token`` — a
         graph-capture-friendly simplification (no runtime GPU-scalar
         indexing) that keeps acceptance counting honest while avoiding a
         dynamic gather kernel inside a conditional graph body.
      4) Both hidden buffers slide: ``hidden[:, -1, :] = emb[final_token]``.

    Exit condition: ``round >= max_rounds`` (no EOS; random weights).
    A real speculative decoder exits on EOS or max_tokens — the same
    ``while_loop`` + counter/flag machinery carries that case too.

    ``*_forward(ctx, l_hidden_in, l_logits_out)`` must produce logits of
    shape ``(1, seq, vocab)``.
    """
    V = cfg_target.vocab

    with ctx.while_loop() as loop:
        # --- Scratches for the tasks below. All live only inside this
        # repeat-body scope thanks to no_export=True. --------------------
        # argmax(K+1,) sink for the target verifier task.
        l_picked_scratch = _logical_data_empty_no_export(
            ctx, (K + 1,), np.int64, name="picked_scratch"
        )
        # scalar scratch for the accepted-count delta (in tac's float dtype).
        l_acc_delta = _logical_data_empty_no_export(
            ctx, (1,), cfg_target.np_dtype, name="acc_delta"
        )

        # --- 1) Target: one forward pass on the target hidden ----------
        l_target_logits = ctx.logical_data_empty(
            (1, cfg_target.seq, V), cfg_target.np_dtype, name="tgt_logits"
        )
        target_forward(ctx, l_hidden_target, l_target_logits)

        # --- 2) Draft: K sequential single-token proposals -------------
        l_draft_logits = ctx.logical_data_empty(
            (1, cfg_draft.seq, V), cfg_draft.np_dtype, name="draft_logits"
        )
        l_draft_next = ctx.logical_data_empty((1,), np.int64, name="draft_next")

        for k in range(K):
            draft_forward(ctx, l_hidden_draft, l_draft_logits)
            stf_sample_argmax_last(ctx, l_draft_logits, l_draft_next)
            # Store into position k of the K-buffer of draft tokens.
            with pytorch_task(ctx, l_draft_tokens.rw(), l_draft_next.read()) as (
                tdt,
                tdn,
            ):
                tdt[0, k : k + 1].copy_(tdn.view(1))
            stf_append_token_hidden(ctx, l_hidden_draft, l_emb, l_draft_next)

        # Argmax at each of the last K+1 positions (target verifier tokens
        # + bonus). Use out= so argmax lands directly in the STF scratch
        # with no PyTorch temp allocation on the external stream.
        with pytorch_task(
            ctx,
            l_target_logits.read(),
            l_picked_scratch.write(),
            l_target_tokens.write(),
        ) as (tl, tps, tt):
            torch.argmax(tl[0, -(K + 1) :, :], dim=-1, out=tps)
            tt[0, : K + 1].copy_(tps)

        # --- 3) Accept/reject + emit bonus -----------------------------
        l_final_token = _logical_data_empty_no_export(
            ctx, (1,), np.int64, name="final_tok"
        )
        with pytorch_task(
            ctx,
            l_draft_tokens.read(),
            l_target_tokens.read(),
            l_accepted.rw(),
            l_acc_delta.write(),
            l_final_token.write(),
        ) as (tdt, tt, tac, tad, tft):
            match = (tdt[0, :K] == tt[0, :K]).to(torch.int64)
            prefix_ok = torch.cumprod(match, dim=0)
            # Sink the scalar accepted-delta into the STF scratch (casting
            # to tac's float dtype in the scratch write).
            tad[:] = prefix_ok.sum().to(tad.dtype).view(tad.shape)
            tac.add_(tad)
            tft.copy_(tt[0, K : K + 1])

        # --- 4) Slide both hidden states by one token ------------------
        stf_append_token_hidden(ctx, l_hidden_target, l_emb, l_final_token)
        stf_append_token_hidden(ctx, l_hidden_draft, l_emb, l_final_token)

        # --- 5) Round counter / termination ----------------------------
        stf_advance_counter_flag(
            ctx, l_round, l_done, max_rounds, scratch=l_cond_scratch
        )
        loop.continue_while(l_done, ">", 0.5)


__all__ = [
    "MiniGPTConfig",
    "TINY",
    "DRAFT",
    "build_random_weights",
    "make_block_scratches",
    "stf_layernorm",
    "stf_linear",
    "stf_ffn_fused",
    "stf_attention_sdpa",
    "stf_attention_parallel_heads",
    "stf_transformer_block",
    "stf_transformer_stack",
    "stf_lm_head",
    "stf_sample_argmax_last",
    "stf_append_token_hidden",
    "stf_advance_counter_flag",
    "make_cond_scratch",
    "validate_forward",
    "spec_decode_loop",
]
