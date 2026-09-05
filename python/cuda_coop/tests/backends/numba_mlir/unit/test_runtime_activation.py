# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys
import textwrap
from threading import RLock
from types import SimpleNamespace

import pytest

from cuda.coop.numba_mlir._compiler import _activation
from tests.support.paths import PACKAGE_ROOT


@pytest.mark.parametrize(
    ("hook_name", "count"),
    [
        ("CoopGroupHierarchyPlanner", 0),
        ("CoopWholeFunctionPlanner", 2),
        ("CoopSinglePhaseRewrite", 0),
    ],
)
def test_registration_postcondition_rejects_noop_or_duplicate_hooks(
    hook_name,
    count,
):
    group_planner = type("CoopGroupHierarchyPlanner", (), {})
    whole_planner = type("CoopWholeFunctionPlanner", (), {})
    rewrite = type("CoopSinglePhaseRewrite", (), {})
    planner_counts = {
        "CoopGroupHierarchyPlanner": 1,
        "CoopWholeFunctionPlanner": 1,
    }
    rewrite_count = 1
    if hook_name in planner_counts:
        planner_counts[hook_name] = count
    else:
        rewrite_count = count
    snapshot = _activation._RegistrationSnapshot(
        planner_registry=SimpleNamespace(
            _lock=RLock(),
            _planners=(
                [group_planner] * planner_counts["CoopGroupHierarchyPlanner"]
                + [whole_planner] * planner_counts["CoopWholeFunctionPlanner"]
            ),
        ),
        planners=(),
        rewrite_registry=SimpleNamespace(
            rewrites={"before-inference": [rewrite] * rewrite_count}
        ),
        rewrites={},
    )

    with pytest.raises(_activation._NumbaMlirBackendImportError) as exc_info:
        _activation._verify_registration_postconditions(
            snapshot,
            SimpleNamespace(
                CoopWholeFunctionPlanner=whole_planner,
                CoopSinglePhaseRewrite=rewrite,
            ),
            SimpleNamespace(CoopGroupHierarchyPlanner=group_planner),
        )

    error = exc_info.value
    assert error.reason_code == "registration-postcondition-failed"
    assert error.details["registration_counts"][hook_name] == count


def test_registration_rollback_restores_both_compiler_registries():
    baseline = type("BaselinePlanner", (), {})
    partial = type("PartialPlanner", (), {})
    baseline_rewrite = type("BaselineRewrite", (), {})
    partial_rewrite = type("PartialRewrite", (), {})
    planner_registry = SimpleNamespace(
        _lock=RLock(),
        _planners=[baseline, partial],
    )
    rewrite_registry = SimpleNamespace(
        rewrites={
            "before-inference": [baseline_rewrite, partial_rewrite],
            "after-inference": [partial_rewrite],
        }
    )
    snapshot = _activation._RegistrationSnapshot(
        planner_registry=planner_registry,
        planners=(baseline,),
        rewrite_registry=rewrite_registry,
        rewrites={"before-inference": (baseline_rewrite,)},
    )

    _activation._restore_registrations(snapshot)

    assert planner_registry._planners == [baseline]
    assert rewrite_registry.rewrites == {"before-inference": [baseline_rewrite]}


def test_runtime_loading_retries_after_a_failed_qualified_import(monkeypatch):
    runtime = object()
    error = _activation._NumbaMlirBackendImportError(
        "backend-runtime-missing",
        "runtime unavailable",
    )
    outcomes = iter(((None, error), (runtime, None)))
    monkeypatch.setattr(_activation, "_cuda_module", None)
    monkeypatch.setattr(_activation, "_load_runtime", lambda: next(outcomes))

    with pytest.raises(_activation._NumbaMlirBackendImportError) as exc_info:
        _activation._require_runtime()

    assert exc_info.value is error
    assert _activation._require_runtime() is runtime
    assert _activation._cuda_module is runtime


def test_runtime_hook_initialization_requires_public_0_5_capabilities(monkeypatch):
    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        refresh_registries=lambda **kwargs: None,
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
    )
    rewrites = SimpleNamespace(
        Rewrite=type("Rewrite", (), {}),
        register_rewrite=lambda kind: lambda rewrite: rewrite,
    )
    group_planner_module = SimpleNamespace(
        CoopGroupHierarchyPlanner=type("CoopGroupHierarchyPlanner", (), {})
    )
    planner_module = SimpleNamespace(
        CoopWholeFunctionPlanner=type("CoopWholeFunctionPlanner", (), {}),
        CoopSinglePhaseRewrite=type("CoopSinglePhaseRewrite", (), {}),
    )
    modules = {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
        "cuda.coop.numba_mlir._compiler._rewrite": planner_module,
        "cuda.coop.numba_mlir._compiler._group_planner": group_planner_module,
    }

    monkeypatch.setattr(_activation, "_require_runtime", lambda: object())
    monkeypatch.setattr(
        _activation.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        _activation,
        "_snapshot_registrations",
        lambda rewrite_module: object(),
    )
    monkeypatch.setattr(
        _activation,
        "_verify_registration_postconditions",
        lambda snapshot, module, group_module: None,
    )
    monkeypatch.setattr(
        _activation,
        "_restore_registrations",
        lambda snapshot: pytest.fail("successful activation must not roll back"),
    )

    _activation._initialize_runtime_hooks()


def test_qualified_import_reports_a_missing_public_runtime():
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        class BlockRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "numba_cuda_mlir" or fullname.startswith(
                    "numba_cuda_mlir."
                ):
                    raise ImportError("blocked runtime", name=fullname)
                return None

        sys.meta_path.insert(0, BlockRuntime())

        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "numba-cuda-mlir"
            assert exc.reason_code == "backend-runtime-missing"
            assert exc.details["missing"] == "numba_cuda_mlir"
            assert isinstance(exc.__cause__, ImportError)
            message = str(exc)
            assert "numba-cuda-mlir>=0.5.0" in message
            assert "cuda-coop[numba-cuda-mlir-cu12]" in message
            assert "cuda-coop[numba-cuda-mlir-cu13]" in message
        else:
            raise AssertionError("qualified import unexpectedly succeeded")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
