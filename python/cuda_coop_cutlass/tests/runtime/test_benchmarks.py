# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check source-built benchmark families through their actual launch path.

Three blocks expose accidentally shared tile origins. Poisoned direct and
16-node graph outputs must match independent NumPy references; the runner
also checks output tails and unchanged inputs. No timing is performed.
"""

import importlib
import shutil
from types import SimpleNamespace

import cutlass
import pytest

from cuda.bindings import driver
from tests.support.paths import PACKAGE_ROOT, REPO_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_OPERATIONS = (
    "load",
    "store",
    "sum",
    "exclusive_sum",
    "reduce_batched",
    "exchange",
    "shuffle",
    "merge_sort_pairs",
    "histogram",
    "topk_max_pairs",
    "radix_sort_pairs",
    "radix_rank_keys",
    "adjacent_difference",
    "discontinuity",
    "run_length_decode",
    "run_length_decode_into",
)


@pytest.fixture(autouse=True)
def source_benchmark_headers(monkeypatch):
    """Match the native baseline's checkout headers in installed-wheel CI."""
    monkeypatch.setenv("CUDA_COOP_CUTLASS_CCCL_ROOT", str(REPO_ROOT))


@pytest.mark.parametrize("provider", ("cub", "cutlass"))
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_source_benchmark_family(monkeypatch, tmp_path, provider, operation):
    pytest.importorskip("cuda.core", reason="benchmark launch dependency")
    if provider == "cub" and shutil.which("nvcc") is None:
        pytest.skip("the native benchmark baseline requires nvcc")
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT / "benchmarks"))
    cases = importlib.import_module("_cases")
    runner = importlib.import_module("run")
    case = next(
        case for case in cases.all_cases() if case.operation == operation
    )
    cutlass.cuda.initialize_cuda_context()
    stream = runner.checked(driver.cuStreamCreate(1))
    state = SimpleNamespace(
        get_int64=lambda axis: {"Blocks": 3, "Launches": 16}[axis],
        get_device=lambda: int(runner.checked(driver.cuCtxGetDevice())),
        get_stream=lambda: SimpleNamespace(addressof=lambda: int(stream)),
        add_element_count=lambda *args: None,
        add_summary=lambda *args: None,
    )
    args = SimpleNamespace(
        provider=provider,
        timing="cold",
        validate_only=True,
        export_inputs=False,
    )
    try:
        result = runner.run_case(state, case, args, tmp_path)
    finally:
        runner.checked(driver.cuStreamDestroy(stream))
    assert result["correct"]
    assert result["kernel"]["grid"] == [3, 1, 1]
    assert result["kernel"]["block"] == [case.threads, 1, 1]


def test_capture_failure_recovers_stream(monkeypatch, tmp_path):
    """A failed callback must not leave later cases trapped in capture mode."""
    pytest.importorskip("cuda.core", reason="benchmark launch dependency")
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT / "benchmarks"))
    kernels = importlib.import_module("_kernels")
    runner = importlib.import_module("run")
    cases = importlib.import_module("_cases")
    case = next(case for case in cases.all_cases() if case.operation == "sum")
    cutlass.cuda.initialize_cuda_context()
    stream = runner.checked(driver.cuStreamCreate(1))

    def fail():
        raise RuntimeError("deliberate capture callback failure")

    state = SimpleNamespace(
        get_int64=lambda axis: {"Blocks": 3, "Launches": 16}[axis],
        get_device=lambda: int(runner.checked(driver.cuCtxGetDevice())),
        get_stream=lambda: SimpleNamespace(addressof=lambda: int(stream)),
        add_element_count=lambda *args: None,
        add_summary=lambda *args: None,
    )
    args = SimpleNamespace(
        provider="cutlass",
        timing="cold",
        validate_only=True,
        export_inputs=False,
    )
    try:
        with pytest.raises(RuntimeError, match="deliberate capture"):
            kernels.capture(stream, fail)
        status = runner.checked(driver.cuStreamIsCapturing(stream))
        assert (
            status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE
        )
        result = runner.run_case(state, case, args, tmp_path)
        assert result["correct"]
    finally:
        runner.checked(driver.cuStreamDestroy(stream))


@pytest.mark.parametrize("route", ("0", "1"))
def test_compiler_route_parameter_binding(tmp_path, route):
    """Both compiler routes must preserve results after exporting a cubin.

    A fresh interpreter is necessary because CuTe caches compiler environment
    options. The extension route can remove the unused auxiliary input.
    """
    import json
    import os
    import subprocess
    import sys

    pytest.importorskip("cuda.bench", reason="benchmark runner dependency")
    artifacts = tmp_path / "artifacts"
    environment = dict(os.environ, CUTE_DSL_USE_EXTENSION_COMPILER=route)
    environment["PYTHONPATH"] = (
        str(PACKAGE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    command = [
        sys.executable,
        str(PACKAGE_ROOT / "benchmarks/run.py"),
        "--provider",
        "cutlass",
        "--case",
        "block128_load_int32_i4_direct",
        "--blocks",
        "3",
        "--launches",
        "1",
        "16",
        "--validate-only",
        "--artifacts",
        str(artifacts),
    ]
    completed = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((artifacts / "qualification.json").read_text())
    assert receipt["errors"] == []
    assert len(receipt["results"]) == 2
    for result in receipt["results"]:
        assert result["correct"] and result["source_launch_checked"]
        metadata = result["kernel"]
        assert metadata["compiler_route"] == (
            "extension" if route == "1" else "legacy"
        )
        assert metadata["abi"] == [
            parameter["binding"] for parameter in metadata["parameter_layout"]
        ]
        assert metadata["parameter_layout_source"] == "captured driver launch"
