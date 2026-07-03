# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Small stackable STF picture: branches, a while loop in each branch, then join.

The graph shape is intentionally the point of the example:

    top launchable graph
      for each branch:
        child graph:
          seed_branch kernel
          while residual > 0.5:
            relax_branch kernel
      join_branches kernel

Generate the CUDA graph DOT:

    python example_stackable_branch_while_warp.py --cuda-dot branch_while.dot
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
from cuda.bindings import runtime as cudart  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

wp = pytest.importorskip("warp")
wp_stf = pytest.importorskip("warp.stf_experimental")

N = 64
WHILE_ITERS = 2
BRANCHES = (
    ("left", 1.0),
    ("middle", 2.0),
    ("right", 3.0),
)


@wp.kernel
def seed_branch(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
    bias: wp.float32,
):
    i = wp.tid()
    y[i] = x[i] + bias
    if i == 0:
        residual[0] = wp.float32(WHILE_ITERS)


@wp.kernel
def relax_branch(
    y: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    y[i] = y[i] + wp.float32(1.0)
    if i == 0:
        residual[0] = residual[0] - wp.float32(1.0)


@wp.kernel
def join_branches(
    out: wp.array(dtype=wp.float32),
    left: wp.array(dtype=wp.float32),
    middle: wp.array(dtype=wp.float32),
    right: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = left[i] + middle[i] + right[i]


def build_picture_graph():
    x_host = np.ones(N, dtype=np.float32)
    out_host = np.zeros(N, dtype=np.float32)

    graph = stf.task_graph()
    ctx = graph.context

    l_x = ctx.logical_data(x_host, name="x")
    l_out = ctx.logical_data(out_host, name="out")

    branch_data = []
    for name, bias in BRANCHES:
        branch_data.append(
            (
                name,
                wp.float32(bias),
                ctx.logical_data_empty((N,), np.float32, name=f"{name}_y"),
                ctx.logical_data_empty((1,), np.float32, name=f"{name}_residual"),
            )
        )

    with graph:
        # Each branch is its own child graph: first a normal kernel, then a
        # branch-local CUDA conditional while loop.
        for name, bias, l_y, l_residual in branch_data:
            with ctx.graph_scope():
                l_x.push(stf.AccessMode.READ)

                with wp_stf.task(ctx, l_x.read(), l_y.write(), l_residual.write()) as (
                    stream,
                    x,
                    y,
                    residual,
                ):
                    wp.launch(
                        seed_branch, dim=N, inputs=[x, y, residual, bias], stream=stream
                    )

                with ctx.while_loop() as loop:
                    with wp_stf.task(ctx, l_y.rw(), l_residual.rw()) as (
                        stream,
                        y,
                        residual,
                    ):
                        wp.launch(
                            relax_branch, dim=N, inputs=[y, residual], stream=stream
                        )

                    loop.continue_while(l_residual, ">", 0.5)
            print(f"built branch: {name}")

        # After all branch child graphs have completed, one final kernel joins them.
        with wp_stf.task(
            ctx,
            l_out.write(),
            branch_data[0][2].read(),
            branch_data[1][2].read(),
            branch_data[2][2].read(),
        ) as (stream, out, left, middle, right):
            wp.launch(
                join_branches,
                dim=N,
                inputs=[out, left, middle, right],
                stream=stream,
            )

    return graph, out_host


def dump_cuda_graph_dot(cuda_graph, path, verbose=False):
    flags = cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsConditionalNodeParams
    if verbose:
        flags |= cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose

    err = cudart.cudaGraphDebugDotPrint(
        cudart.cudaGraph_t(int(cuda_graph)),
        os.fsencode(path),
        int(flags),
    )
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphDebugDotPrint failed with cudaError_t={int(err)}")


def main(cuda_dot=None, cuda_dot_verbose=False):
    wp.init()

    graph, out_host = build_picture_graph()

    if cuda_dot:
        dump_cuda_graph_dot(graph.graph, cuda_dot, cuda_dot_verbose)
        print(f"CUDA graph DOT written to: {cuda_dot}")

    graph.launch()
    graph.reset()
    graph.finalize()

    expected = sum(1.0 + bias + WHILE_ITERS for _, bias in BRANCHES)
    print(f"out[0] = {out_host[0]} (expected {expected})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-dot", help="write CUDA runtime graph DOT to this path")
    parser.add_argument(
        "--cuda-dot-verbose",
        action="store_true",
        help="include verbose CUDA node params in --cuda-dot output",
    )
    args = parser.parse_args()

    main(cuda_dot=args.cuda_dot, cuda_dot_verbose=args.cuda_dot_verbose)
