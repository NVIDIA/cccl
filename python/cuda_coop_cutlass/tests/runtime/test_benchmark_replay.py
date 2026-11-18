# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check qualified cubin replay and reject mismatched source evidence."""

import importlib
import json
import shutil
from types import SimpleNamespace

import cutlass
import pytest

from cuda.bindings import driver
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.fixture
def replay_bundle(monkeypatch, tmp_path):
    pytest.importorskip("cuda.core", reason="benchmark launch dependency")
    if shutil.which("nvcc") is None:
        pytest.skip("creating the replay fixture requires nvcc")
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT / "benchmarks"))
    runner = importlib.import_module("run")
    cases = importlib.import_module("_cases")
    case = next(case for case in cases.all_cases() if case.operation == "sum")
    cutlass.cuda.initialize_cuda_context()
    stream = runner.checked(driver.cuStreamCreate(1))
    axes = {"Blocks": 3, "Launches": 1}
    state = SimpleNamespace(
        get_int64=axes.__getitem__,
        get_device=lambda: int(runner.checked(driver.cuCtxGetDevice())),
        get_stream=lambda: SimpleNamespace(addressof=lambda: int(stream)),
        add_element_count=lambda *args: None,
        add_summary=lambda *args: None,
    )
    args = SimpleNamespace(
        provider="cub",
        timing="cold",
        validate_only=True,
        export_inputs=False,
        compiled_kernels=None,
    )
    bundle = tmp_path / "source"
    try:
        result = runner.run_case(state, case, args, bundle)
        receipt = {
            "versions": {"source-test": "CUB"},
            "results": [result],
            "errors": [],
        }
        (bundle / "qualification.json").write_text(json.dumps(receipt))
        args.compiled_kernels = bundle
        axes["Launches"] = 16
        yield runner, state, case, args, receipt
    finally:
        runner.checked(driver.cuStreamDestroy(stream))


def test_qualified_replay_uses_cubin(monkeypatch, tmp_path, replay_bundle):
    runner, state, case, args, receipt = replay_bundle
    kernels = importlib.import_module("_kernels")

    def no_compilation(*args, **kwargs):
        pytest.fail("Replay unexpectedly attempted source compilation")

    monkeypatch.setattr(kernels, "prepare", no_compilation)
    result = runner.run_case(state, case, args, tmp_path / "replay")
    assert result["correct"]
    assert result["launches"] == 16
    assert result["kernel"]["compilation_environment"] == receipt["versions"]
    assert (
        result["kernel"]["cubin_sha256"]
        == (receipt["results"][0]["kernel"]["cubin_sha256"])
    )
    assert result["kernel"]["replay_of"]["qualification"] == str(
        (args.compiled_kernels / "qualification.json").resolve()
    )


@pytest.mark.parametrize(
    "damage", ("cubin", "architecture", "case", "abi", "layout")
)
def test_replay_rejects_mismatched_source(tmp_path, replay_bundle, damage):
    runner, state, case, args, receipt = replay_bundle
    source = receipt["results"][0]
    directory = args.compiled_kernels / (
        f"d{source['device']}-{case.name}-b3-l1"
    )
    if damage == "cubin":
        with (directory / "kernel.cubin").open("ab") as output:
            output.write(b"changed")
        match = "cubin hash mismatch"
    else:
        if damage == "architecture":
            source["kernel"]["arch"] = "sm_0"
        elif damage == "case":
            source["kernel"]["case"]["items"] += 1
        elif damage == "abi":
            source["kernel"]["abi"] = ["input", "input"]
        else:
            source["kernel"]["parameter_layout"][0]["size"] = 4
        (args.compiled_kernels / "qualification.json").write_text(
            json.dumps(receipt)
        )
        match = {
            "architecture": "metadata does not match",
            "case": "metadata does not match",
            "abi": "unique pointer subset",
            "layout": "not a pointer ABI",
        }[damage]
    with pytest.raises(ValueError, match=match):
        runner.run_case(state, case, args, tmp_path / "rejected")
