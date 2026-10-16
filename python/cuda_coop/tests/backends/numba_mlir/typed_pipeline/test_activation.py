# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUDA_COOP_NUMBA_MLIR_PIPELINE selects exactly one pipeline per process."""

from __future__ import annotations

import pytest

from ._support import run_python

_REPORT = """
import sys
import cuda.coop.numba_mlir
from numba_cuda_mlir import _whole_function_planners as wfp
from numba_cuda_mlir.numba_cuda.core.rewrites import rewrite_registry

planners = [p.__name__ for p in wfp._planner_registry._planners]
rewrites = [
    r.__name__ for r in rewrite_registry.rewrites["after-inference"]
]
backend = "cuda.coop.numba_mlir"
print("planners", planners)
print("planner-imported", f"{backend}._compiler._planner" in sys.modules)
print("pipeline-imported", f"{backend}._pipeline" in sys.modules)
print("call-resolution", "CoopCallResolution" in rewrites)
"""


@pytest.mark.parametrize("pipeline", [None, "", "planner"])
def test_default_registers_the_planner_only(pipeline):
    result = run_python(_REPORT, pipeline)
    assert result.returncode == 0, result.stderr
    assert "planners ['CoopWholeFunctionPlanner']" in result.stdout
    assert "pipeline-imported False" in result.stdout
    assert "call-resolution False" in result.stdout


@pytest.mark.parametrize("pipeline", ["typed", " Typed "])
def test_typed_registers_call_resolution_and_never_imports_the_planner(
    pipeline,
):
    result = run_python(_REPORT, pipeline)
    assert result.returncode == 0, result.stderr
    assert "planners []" in result.stdout
    assert "planner-imported False" in result.stdout
    assert "pipeline-imported True" in result.stdout
    assert "call-resolution True" in result.stdout


def test_unknown_pipeline_is_an_error():
    result = run_python("import cuda.coop.numba_mlir", "mixed")
    assert result.returncode != 0
    assert (
        "CUDA_COOP_NUMBA_MLIR_PIPELINE='mixed' is not a cuda.coop pipeline; "
        "use one of planner, typed"
    ) in result.stderr
