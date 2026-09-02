# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from cuda.coop.numba_mlir._compiler import _activation
from cuda.coop.numba_mlir._compiler._group_planner import CoopGroupHierarchyPlanner
from cuda.coop.numba_mlir._compiler._rewrite import (
    CoopWholeFunctionPlanner,
)


def _fake_extension_modules(events):
    def register_planner(planner):
        events.append(("register", planner))
        return planner

    extending = SimpleNamespace(
        WholeFunctionPlanner=lambda: None,
        overload=lambda *args, **kwargs: None,
        register_planner=register_planner,
        require_launch_config=lambda state: state,
        typing_registry=object(),
    )
    rewrite = SimpleNamespace(CoopWholeFunctionPlanner=CoopWholeFunctionPlanner)
    group = SimpleNamespace(CoopGroupHierarchyPlanner=CoopGroupHierarchyPlanner)
    return extending, rewrite, group


def test_backend_registers_the_ordered_two_stage_flow_once(monkeypatch):
    events = []
    extending, rewrite, group = _fake_extension_modules(events)
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "numba_cuda_mlir.extending":
            return extending
        if name.endswith("._compiler._rewrite"):
            return rewrite
        if name.endswith("._compiler._group_planner"):
            return group
        raise AssertionError(f"unexpected private runtime import: {name}")

    monkeypatch.setattr(_activation, "_initialized", False)
    monkeypatch.setattr(_activation, "_load_runtime", lambda: None)
    monkeypatch.setattr(_activation.importlib, "import_module", import_module)

    _activation._initialize_runtime_hooks()
    _activation._initialize_runtime_hooks()

    assert events == [
        ("register", CoopGroupHierarchyPlanner),
        ("register", CoopWholeFunctionPlanner),
    ]
    assert imported == [
        "numba_cuda_mlir.extending",
        "cuda.coop.numba_mlir._compiler._group_planner",
        "cuda.coop.numba_mlir._compiler._rewrite",
    ]


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


def test_registration_failure_is_structured_and_retry_safe(monkeypatch):
    failure = ModuleNotFoundError(
        "No module named 'cuda.bindings'", name="cuda.bindings"
    )
    events = []
    registered = []
    failed_once = False

    def register_planner(planner):
        nonlocal failed_once
        events.append(("register", planner))
        if planner is CoopWholeFunctionPlanner and not failed_once:
            failed_once = True
            raise failure
        if planner not in registered:
            registered.append(planner)
        return planner

    extending, rewrite, group = _fake_extension_modules(events)
    extending.register_planner = register_planner

    def import_module(name):
        if name == "numba_cuda_mlir.extending":
            return extending
        if name.endswith("._compiler._rewrite"):
            return rewrite
        if name.endswith("._compiler._group_planner"):
            return group
        raise AssertionError(f"unexpected private runtime import: {name}")

    monkeypatch.setattr(_activation, "_initialized", False)
    monkeypatch.setattr(_activation, "_load_runtime", lambda: None)
    monkeypatch.setattr(_activation.importlib, "import_module", import_module)

    with pytest.raises(_activation.NumbaMlirBackendImportError) as exc_info:
        _activation._initialize_runtime_hooks()

    assert not _activation._initialized
    assert exc_info.value.reason_code == "backend-hook-activation-failed"
    assert exc_info.value.details["activation_phase"] == "planner registration"
    assert exc_info.value.__cause__ is failure

    _activation._initialize_runtime_hooks()

    assert registered == [CoopGroupHierarchyPlanner, CoopWholeFunctionPlanner]
    assert events == [
        ("register", CoopGroupHierarchyPlanner),
        ("register", CoopWholeFunctionPlanner),
        ("register", CoopGroupHierarchyPlanner),
        ("register", CoopWholeFunctionPlanner),
    ]


def test_runtime_requirement_reports_the_supported_version_range():
    requirement = _activation._runtime_requirement()

    assert "numba-cuda-mlir>=0.5.0,<0.6" in requirement
