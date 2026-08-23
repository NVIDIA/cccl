# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from collections import defaultdict
from importlib import import_module
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest

import cuda.coop.numba_mlir as coop


@pytest.mark.parametrize("count", [0, 2])
def test_registration_postconditions_reject_noop_or_duplicate_hooks(count):
    group_planner = type("CoopGroupHierarchyPlanner", (), {})
    whole_planner = type("CoopWholeFunctionPlanner", (), {})
    rewrite = type("CoopSinglePhaseRewrite", (), {})
    snapshot = coop._RegistrationSnapshot(
        planner_registry=SimpleNamespace(
            _lock=RLock(),
            _planners=[group_planner] * count + [whole_planner] * count,
        ),
        planners=(),
        rewrite_registry=SimpleNamespace(
            rewrites={"before-inference": [rewrite] * count},
        ),
        rewrites={},
    )

    with pytest.raises(coop._NumbaMlirBackendImportError) as exc_info:
        coop._verify_registration_postconditions(
            snapshot,
            SimpleNamespace(
                CoopWholeFunctionPlanner=whole_planner,
                CoopSinglePhaseRewrite=rewrite,
            ),
            SimpleNamespace(CoopGroupHierarchyPlanner=group_planner),
        )

    error = exc_info.value
    assert error.reason_code == "registration-postcondition-failed"
    assert error.details["registration_counts"] == {
        "CoopGroupHierarchyPlanner": count,
        "CoopWholeFunctionPlanner": count,
        "CoopSinglePhaseRewrite": count,
    }


def test_registration_rollback_supports_plain_rewrite_mappings():
    baseline_planner = type("BaselinePlanner", (), {})
    partial_planner = type("PartialPlanner", (), {})
    baseline_rewrite = type("BaselineRewrite", (), {})
    partial_rewrite = type("PartialRewrite", (), {})
    planner_registry = SimpleNamespace(
        _lock=RLock(),
        _planners=[baseline_planner, partial_planner],
    )
    rewrite_registry = SimpleNamespace(
        rewrites={
            "before-inference": [baseline_rewrite, partial_rewrite],
            "partial": [partial_rewrite],
        }
    )
    snapshot = coop._RegistrationSnapshot(
        planner_registry=planner_registry,
        planners=(baseline_planner,),
        rewrite_registry=rewrite_registry,
        rewrites={"before-inference": (baseline_rewrite,)},
    )

    coop._restore_registrations(snapshot)

    assert planner_registry._planners == [baseline_planner]
    assert rewrite_registry.rewrites == {"before-inference": [baseline_rewrite]}


def test_noop_registration_apis_fail_postconditions_and_roll_back(monkeypatch):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=lambda: None,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda phase: lambda rewrite: rewrite,
        rewrite_registry=SimpleNamespace(rewrites=defaultdict(list)),
    )
    baseline_planner = type("BaselinePlanner", (), {})
    baseline_rewrite = type("BaselineRewrite", (), {})
    planner_registry = SimpleNamespace(
        _lock=RLock(),
        _planners=[baseline_planner],
    )
    rewrites.rewrite_registry.rewrites["baseline"].append(baseline_rewrite)
    planners = SimpleNamespace(_planner_registry=planner_registry)
    planner = SimpleNamespace(
        CoopWholeFunctionPlanner=type("CoopWholeFunctionPlanner", (), {}),
        CoopSinglePhaseRewrite=type("CoopSinglePhaseRewrite", (), {}),
    )
    group_planner = SimpleNamespace(
        CoopGroupHierarchyPlanner=type("CoopGroupHierarchyPlanner", (), {})
    )
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
        "numba_cuda_mlir._whole_function_planners": planners,
        "cuda.coop.numba_mlir._single_phase_rewrites": planner,
        "cuda.coop.numba_mlir._group_rewrites": group_planner,
    }

    monkeypatch.setattr(coop, "_require_runtime", lambda: object())
    monkeypatch.setattr(coop.importlib, "import_module", lambda name: modules[name])

    with pytest.raises(coop._NumbaMlirBackendImportError) as exc_info:
        coop._initialize_runtime_hooks()

    assert exc_info.value.reason_code == "registration-postcondition-failed"
    assert planner_registry._planners == [baseline_planner]
    assert dict(rewrites.rewrite_registry.rewrites) == {"baseline": [baseline_rewrite]}


def test_real_planner_and_rewrite_registration_is_idempotent():
    planner = import_module(f"{coop.__name__}._single_phase_rewrites")
    group_planner = import_module(f"{coop.__name__}._group_rewrites")
    planner_registry = import_module(
        "numba_cuda_mlir._whole_function_planners"
    )._planner_registry
    rewrite_registry = import_module(
        "numba_cuda_mlir.numba_cuda.core.rewrites"
    ).rewrite_registry

    def registration_counts():
        with planner_registry._lock:
            planner_counts = (
                planner_registry._planners.count(
                    group_planner.CoopGroupHierarchyPlanner
                ),
                planner_registry._planners.count(planner.CoopWholeFunctionPlanner),
            )
        rewrite_count = rewrite_registry.rewrites["before-inference"].count(
            planner.CoopSinglePhaseRewrite
        )
        return planner_counts, rewrite_count

    assert registration_counts() == ((1, 1), 1)
    coop._initialize_runtime_hooks()
    assert registration_counts() == ((1, 1), 1)
    coop._initialize_runtime_hooks()
    assert registration_counts() == ((1, 1), 1)


def test_failed_initialization_restores_real_registries_before_retry(monkeypatch):
    rewrites = import_module("numba_cuda_mlir.numba_cuda.core.rewrites")
    planner_registry = import_module(
        "numba_cuda_mlir._whole_function_planners"
    )._planner_registry
    baseline = coop._snapshot_registrations(rewrites)
    original_import_module = coop.importlib.import_module
    failed_module = f"{coop.__name__}._partial_real_registration_test"
    partial_planner = type("PartialPlanner", (), {})
    partial_rewrite = type("PartialRewrite", (), {})

    def import_with_partial_registration(name):
        if name != f"{coop.__name__}._single_phase_rewrites":
            return original_import_module(name)
        with planner_registry._lock:
            planner_registry._planners.append(partial_planner)
        rewrites.rewrite_registry.rewrites["before-inference"].append(partial_rewrite)
        sys.modules[failed_module] = ModuleType(failed_module)
        raise RuntimeError("injected real-registry planner import failure")

    monkeypatch.setattr(
        coop.importlib,
        "import_module",
        import_with_partial_registration,
    )
    for _ in range(2):
        with pytest.raises(
            RuntimeError,
            match="injected real-registry planner import failure",
        ):
            coop._initialize_runtime_hooks()
        assert coop._snapshot_registrations(rewrites) == baseline
        assert failed_module not in sys.modules

    monkeypatch.setattr(coop.importlib, "import_module", original_import_module)
    coop._initialize_runtime_hooks()
    assert coop._snapshot_registrations(rewrites) == baseline


def test_runtime_hook_initialization_uses_required_capabilities(monkeypatch):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=lambda: None,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda phase: lambda rewrite: rewrite,
    )
    planner = SimpleNamespace(
        CoopWholeFunctionPlanner=type("CoopWholeFunctionPlanner", (), {})
    )
    group_planner = SimpleNamespace(
        CoopGroupHierarchyPlanner=type("CoopGroupHierarchyPlanner", (), {})
    )
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
        "cuda.coop.numba_mlir._single_phase_rewrites": planner,
        "cuda.coop.numba_mlir._group_rewrites": group_planner,
    }

    monkeypatch.setattr(coop, "_require_runtime", lambda: object())
    monkeypatch.setattr(
        coop.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(coop, "_snapshot_registrations", lambda rewrites: object())
    monkeypatch.setattr(
        coop,
        "_verify_registration_postconditions",
        lambda snapshot, planner_module, group_planner_module: None,
    )
    monkeypatch.setattr(
        coop,
        "_restore_registrations",
        lambda snapshot: pytest.fail("successful activation must not roll back"),
    )

    coop._initialize_runtime_hooks()


def test_failed_runtime_hook_initialization_rolls_back_before_retry(monkeypatch):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=lambda: None,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda phase: lambda rewrite: rewrite,
    )
    registrations = ["baseline"]
    restored = []
    partial_module = f"{coop.__name__}._partial_registration_test"

    def snapshot(_rewrites):
        assert _rewrites is rewrites
        return tuple(registrations)

    def restore(state):
        registrations[:] = state
        restored.append(state)

    def import_module(name):
        if name == "numba_cuda_mlir.extending":
            return extending
        if name == "numba_cuda_mlir.numba_cuda.core.rewrites":
            return rewrites
        if name == f"{coop.__name__}._single_phase_rewrites":
            registrations.append("partial")
            sys.modules[partial_module] = ModuleType(partial_module)
            raise RuntimeError("injected planner import failure")
        raise AssertionError(name)

    monkeypatch.setattr(coop, "_require_runtime", lambda: object())
    monkeypatch.setattr(coop, "_snapshot_registrations", snapshot)
    monkeypatch.setattr(coop, "_restore_registrations", restore)
    monkeypatch.setattr(coop.importlib, "import_module", import_module)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected planner import failure"):
            coop._initialize_runtime_hooks()
        assert registrations == ["baseline"]
        assert partial_module not in sys.modules

    assert restored == [("baseline",), ("baseline",)]


@pytest.mark.parametrize(
    "missing_name",
    ["refresh_registries", "set_required_dynamic_shared_memory"],
)
def test_runtime_hook_initialization_rejects_missing_capability(
    monkeypatch, missing_name
):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=lambda: None,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    delattr(extending, missing_name)
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda phase: lambda rewrite: rewrite,
    )
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
    }

    monkeypatch.setattr(coop, "_require_runtime", lambda: object())
    monkeypatch.setattr(
        coop.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(coop._NumbaMlirBackendImportError) as exc_info:
        coop._initialize_runtime_hooks()

    error = exc_info.value
    qualified_name = f"numba_cuda_mlir.extending.{missing_name}"
    assert error.reason_code == "incomplete-runtime-hook-api"
    assert error.details == {"missing_capabilities": (qualified_name,)}
    assert qualified_name in str(error)


@pytest.mark.parametrize("invalid_value", [None, 1])
def test_runtime_hook_initialization_rejects_noncallable_capability(
    monkeypatch,
    invalid_value,
):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=invalid_value,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda phase: lambda rewrite: rewrite,
    )
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
    }

    monkeypatch.setattr(coop, "_require_runtime", lambda: object())
    monkeypatch.setattr(
        coop.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(coop._NumbaMlirBackendImportError) as exc_info:
        coop._initialize_runtime_hooks()

    qualified_name = "numba_cuda_mlir.extending.refresh_registries"
    assert exc_info.value.reason_code == "incomplete-runtime-hook-api"
    assert exc_info.value.details["missing_capabilities"] == (qualified_name,)
