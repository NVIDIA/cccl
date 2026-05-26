# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
LLM demo — Layer A: a transformer block is a STF DAG.

One eager-mode ``stf.context``. One forward pass through a single transformer
block (LayerNorm + Q/K/V projections + multi-head attention + out-projection
+ residual + LayerNorm + FFN + residual). Attention is realized as H parallel
per-head STF tasks (``attention="parallel_heads"``) so the DOT graph shows
H independent attention branches.

Presentation role: the slide "STF discovered the parallelism" — we don't
race anything, we just show the DAG structure STF infers from the data-
dependency graph we described.

Perf numbers are deliberately **not** printed. This demo is about the graph.

Env knobs
---------
``CUDASTF_DOT_FILE=/tmp/layer_a.dot``
    Standard STF env variable. When set, STF writes the dataflow DAG to this
    file. Convert with ``dot -Tpng /tmp/layer_a.dot -o layer_a.png`` for the
    slide.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_helpers import (  # noqa: E402
    TINY,
    build_random_weights,
    stf_transformer_block,
    validate_forward,
)

import cuda.stf._experimental as stf  # noqa: E402


def test_transformer_block_dag():
    cfg = TINY
    B = 1

    rng = np.random.default_rng(0)
    x_host = rng.standard_normal((B, cfg.seq, cfg.hidden)).astype(cfg.np_dtype)
    out_host = np.zeros_like(x_host)

    ctx = stf.context()

    l_x = ctx.logical_data(x_host, name="x")
    l_out = ctx.logical_data(out_host, name="out")
    # For DAG viz we want the weights drawn as regular nodes rather than
    # collapsed read-only edges, so we intentionally do NOT mark read-only.
    weights = build_random_weights(ctx, cfg, seed=1, read_only=False)

    stf_transformer_block(
        ctx, l_x, weights["layers"][0], l_out, cfg,
        attention="parallel_heads",
    )

    ctx.finalize()

    var = validate_forward(out_host, cfg)

    print("=== LLM demo — Layer A: transformer-block DAG ===")
    print(f"Shape: {out_host.shape}, dtype: {out_host.dtype}, variance: {var:.3e}")
    print(f"Heads (parallel STF tasks): {cfg.heads}")
    print(f"Layers: 1, hidden: {cfg.hidden}, seq: {cfg.seq}")
    dot_file = os.environ.get("CUDASTF_DOT_FILE", "")
    if dot_file:
        print(f"DAG written to {dot_file}")
    else:
        print("Set CUDASTF_DOT_FILE=layer_a.dot to dump the DOT graph.")
    print("PASSED")


if __name__ == "__main__":
    test_transformer_block_dag()
