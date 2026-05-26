"""Minimal reproducer for a second STF + while_loop hang.

Distinct from the mod-4 / PyTorch-caching-allocator bug fixed by
``stf_advance_counter_flag`` and ``make_cond_scratch``.

Symptom
-------
A ``stackable_context.while_loop()`` body that contains two asymmetric
transformer-stack chains (different number of ``stf_transformer_block``
calls per chain) never terminates at ``ctx.finalize()``. The same body
**passes cleanly** in the following equivalent setups:

  * Two SYMMETRIC stacks in the same while body (NA == NB).
  * Same asymmetric body run in eager ``stf.context()`` (no graph).
  * Same asymmetric body run OUTSIDE any while_loop.

So the trigger is specifically:

    stackable_context + while_loop + two chains of different length.

Why this blocks the Layer D demo
--------------------------------
The speculative-decoding demo is the natural setting for asymmetric
stacks: the target model has more layers than the draft model, and both
forwards live inside the outer spec-round while_loop. With the symmetric
simplification (same cfg for draft and target) the demo runs; with the
realistic asymmetric stacks it hangs in ``finalize()``.

This reproducer is kept minimal so the STF team can bisect inside
``cuda::experimental::stf`` (no PyTorch ``ExternalStream`` needed on the
reproducer path — we only need the two transformer chains; the bug is
already visible with plain matmul / linear helpers).
"""

from __future__ import annotations

import numpy as np

import cuda.stf._experimental as stf
from llm_helpers import (
    TINY,
    build_random_weights,
    make_cond_scratch,
    stf_advance_counter_flag,
    stf_lm_head,
    stf_transformer_stack,
)
from pytorch_task import pytorch_task


def _build_hidden(cfg, rng):
    return rng.standard_normal((1, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)


def run(NA: int, NB: int, *, rounds: int = 1) -> None:
    """Run a single-iteration while_loop body with two stacks of depth NA/NB."""
    import dataclasses

    cfg_a = dataclasses.replace(TINY, n_layers=NA)
    cfg_b = dataclasses.replace(TINY, n_layers=NB)
    rng = np.random.default_rng(0)

    ha = _build_hidden(cfg_a, rng)
    hb = _build_hidden(cfg_b, rng)
    round_h = np.zeros((1,), dtype=np.float64)
    done_h = np.ones((1,), dtype=np.float64)

    ctx = stf.stackable_context()
    l_ha = ctx.logical_data(ha, name="ha")
    l_hb = ctx.logical_data(hb, name="hb")
    l_r = ctx.logical_data(round_h, name="r")
    l_d = ctx.logical_data(done_h, name="d")
    l_cs = make_cond_scratch(ctx)

    wa = build_random_weights(ctx, cfg_a, seed=1, read_only=True)
    wb = build_random_weights(ctx, cfg_b, seed=2, read_only=True)

    def make_fwd(cfg, w):
        def _f(ctx, lh, ll):
            l_hn = ctx.logical_data_empty((1, cfg.seq, cfg.hidden), cfg.np_dtype)
            stf_transformer_stack(ctx, lh, w, cfg, l_hn)
            with pytorch_task(ctx, l_hn.read(), lh.write()) as (thn, th):
                th[:] = thn
            stf_lm_head(ctx, lh, w["lm_head"], ll)

        return _f

    fwd_a = make_fwd(cfg_a, wa)
    fwd_b = make_fwd(cfg_b, wb)

    with ctx.while_loop() as loop:
        l_la = ctx.logical_data_empty((1, cfg_a.seq, cfg_a.vocab), cfg_a.np_dtype)
        l_lb = ctx.logical_data_empty((1, cfg_b.seq, cfg_b.vocab), cfg_b.np_dtype)
        fwd_a(ctx, l_ha, l_la)
        fwd_b(ctx, l_hb, l_lb)
        stf_advance_counter_flag(ctx, l_r, l_d, rounds, scratch=l_cs)
        loop.continue_while(l_d, ">", 0.5)

    ctx.finalize()


if __name__ == "__main__":
    import sys

    # Symmetric — passes.
    print("NA=NB=2 ...", flush=True)
    run(2, 2)
    print("  OK", flush=True)

    print("NA=NB=6 ...", flush=True)
    run(6, 6)
    print("  OK", flush=True)

    # Asymmetric — hangs at ctx.finalize().
    print("NA=6 NB=2 ... (expected to hang)", flush=True)
    run(6, 2)
    print("  unexpectedly finished", flush=True)
    sys.exit(0)
