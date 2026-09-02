# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from types import ModuleType

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import _whole_function_planners, extending

from cuda.coop.numba_mlir._compiler import _activation
from cuda.coop.numba_mlir._compiler._planner import CoopBlockReducePlanner


def test_backend_registers_exactly_one_planner():
    registry = _whole_function_planners._planner_registry

    with registry._lock:
        assert registry._planners.count(CoopBlockReducePlanner) == 1


def test_activation_is_idempotent():
    registry = _whole_function_planners._planner_registry

    _activation._initialize_runtime_hooks()
    _activation._initialize_runtime_hooks()

    with registry._lock:
        assert registry._planners.count(CoopBlockReducePlanner) == 1


def test_runtime_import_failures_preserve_the_original_cause(monkeypatch):
    failure = OSError("broken runtime dependency")

    def fail_import(name):
        assert name == "numba_cuda_mlir"
        raise failure

    monkeypatch.setattr(_activation.importlib, "import_module", fail_import)

    with pytest.raises(_activation.NumbaMlirBackendImportError) as exc_info:
        _activation._load_runtime()

    assert exc_info.value.reason_code == "transitive-runtime-import-failed"
    assert exc_info.value.__cause__ is failure
    assert exc_info.value.details["exception_type"] == "OSError"


def test_cuda_runtime_import_failures_are_structured(monkeypatch):
    runtime = object()
    failure = OSError("CUDA dependency mismatch")

    def import_module(name):
        if name == "numba_cuda_mlir":
            return runtime
        assert name == "numba_cuda_mlir.cuda"
        raise failure

    monkeypatch.setattr(_activation.importlib, "import_module", import_module)

    with pytest.raises(_activation.NumbaMlirBackendImportError) as exc_info:
        _activation._load_runtime()

    assert exc_info.value.reason_code == "transitive-runtime-import-failed"
    assert exc_info.value.__cause__ is failure
    assert exc_info.value.details["exception_type"] == "OSError"


def test_registration_refresh_failure_rolls_back_and_is_structured(monkeypatch):
    registry = _whole_function_planners._planner_registry
    with registry._lock:
        before = tuple(registry._planners)
    probe_name = "cuda.coop.numba_mlir._compiler._failed_probe"
    failure = ModuleNotFoundError(
        "No module named 'cuda.bindings'", name="cuda.bindings"
    )

    def fail_refresh():
        with registry._lock:
            registry._planners.append(type("LeakedPlanner", (), {}))
        sys.modules[probe_name] = ModuleType(probe_name)
        raise failure

    monkeypatch.setattr(_activation, "_initialized", False)
    monkeypatch.setattr(extending, "refresh_registries", fail_refresh)

    with pytest.raises(_activation.NumbaMlirBackendImportError) as exc_info:
        _activation._initialize_runtime_hooks()

    with registry._lock:
        assert tuple(registry._planners) == before
    assert probe_name not in sys.modules
    assert exc_info.value.reason_code == "backend-hook-activation-failed"
    assert exc_info.value.details["activation_phase"] == "registry refresh"
    assert exc_info.value.__cause__ is failure


def test_runtime_requirement_reports_the_supported_version_range():
    requirement = _activation._runtime_requirement()

    assert "numba-cuda-mlir>=0.5.0,<0.6" in requirement
