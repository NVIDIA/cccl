# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fused k-means-assignment CuTe GEMM plus coop routing example.

This example is the current tensor-core bridge for the fused k-means assignment family:
the stock SM120 Blackwell GeForce CuTe GEMM sample computes the fp16
dot-product tile, then a qualified warp-group minimum kernel selects
the nearest centroid for each query row.

The underlying implementation lives in the benchmark reference module so the
example and benchmark harness exercise the same runtime path.
"""

from __future__ import annotations

from typing import Any

from benchmarks.cute.kmeans_assign_reference import (
    PreparedReference,
    prepare_cute_gemm_coop_argmin_reference,
)


def prepare_example() -> PreparedReference:
    """Prepare reusable inputs and a launch-only step for the example."""

    return prepare_cute_gemm_coop_argmin_reference()


def run_example() -> dict[str, Any]:
    """Run the CuTe GEMM plus coop assignment example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
