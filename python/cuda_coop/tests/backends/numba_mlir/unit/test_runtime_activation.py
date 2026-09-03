# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest

from cuda.coop.numba_mlir._compiler import _activation

PACKAGE_ROOT = Path(__file__).parents[4]


def _run_import_probe(script: str) -> None:
    env = os.environ.copy()
    env.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_numba_first_root_import_automatically_activates_backend():
    _run_import_probe(
        """
        import sys

        import numba_cuda_mlir  # noqa: F401
        import cuda.coop  # noqa: F401

        expected = {
            "cuda.coop.numba_mlir",
            "cuda.coop.numba_mlir._compiler._group_planner",
            "cuda.coop.numba_mlir._compiler._rewrite",
        }
        assert expected <= set(sys.modules), expected - set(sys.modules)
        """
    )


def test_root_first_qualified_import_explicitly_activates_backend():
    _run_import_probe(
        """
        import sys

        before = set(sys.modules)
        import cuda.coop  # noqa: F401
        loaded = set(sys.modules) - before

        assert "cuda.coop.numba_mlir" not in sys.modules
        assert not any(
            name == "numba_cuda_mlir" or name.startswith("numba_cuda_mlir.")
            for name in loaded
        ), loaded
        assert not any(
            name == "cuda.bindings" or name.startswith("cuda.bindings.")
            for name in loaded
        ), loaded

        import cuda.coop.numba_mlir  # noqa: F401

        expected = {
            "cuda.coop.numba_mlir",
            "cuda.coop.numba_mlir._compiler._group_planner",
            "cuda.coop.numba_mlir._compiler._rewrite",
        }
        assert expected <= set(sys.modules), expected - set(sys.modules)
        """
    )


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


def test_registration_rollback_preserves_foreign_registrations():
    baseline = type("BaselinePlanner", (), {})
    baseline.__module__ = "cuda.coop.numba_mlir._compiler._group_planner"
    partial = type("PartialPlanner", (), {})
    partial.__module__ = "cuda.coop.numba_mlir._compiler._group_planner"
    foreign = type("ForeignPlanner", (), {})
    foreign.__module__ = "third_party.numba_extension"
    baseline_rewrite = type("BaselineRewrite", (), {})
    baseline_rewrite.__module__ = "cuda.coop.numba_mlir._compiler._rewrite"
    partial_rewrite = type("PartialRewrite", (), {})
    partial_rewrite.__module__ = "cuda.coop.numba_mlir._compiler._rewrite"
    foreign_rewrite = type("ForeignRewrite", (), {})
    foreign_rewrite.__module__ = "third_party.numba_extension"
    planner_registry = SimpleNamespace(
        _lock=RLock(),
        _planners=[baseline],
    )
    rewrite_registry = SimpleNamespace(
        rewrites={
            "before-inference": [baseline_rewrite],
        }
    )
    snapshot = _activation._RegistrationSnapshot(
        planner_registry=planner_registry,
        planners=(baseline,),
        rewrite_registry=rewrite_registry,
        rewrites={"before-inference": (baseline_rewrite,)},
    )

    activation_started = Event()
    foreign_registered = Event()

    def register_foreign_hooks():
        assert activation_started.wait(timeout=5)
        with planner_registry._lock:
            planner_registry._planners.append(foreign)
        rewrite_registry.rewrites["before-inference"].append(foreign_rewrite)
        foreign_registered.set()

    thread = Thread(target=register_foreign_hooks)
    thread.start()
    planner_registry._planners.append(partial)
    rewrite_registry.rewrites["before-inference"].append(partial_rewrite)
    activation_started.set()
    assert foreign_registered.wait(timeout=5)
    _activation._restore_registrations(snapshot)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert planner_registry._planners == [baseline, foreign]
    assert rewrite_registry.rewrites == {
        "before-inference": [baseline_rewrite, foreign_rewrite]
    }


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


def test_failed_activation_after_types_import_leaves_typeof_registry_unchanged():
    script = textwrap.dedent(
        """
        import importlib
        import os

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        # Establish the runtime's own lazy external-function registrations
        # before measuring this backend's activation transaction.
        from numba_cuda_mlir.compiler import ExternFunction  # noqa: F401
        from numba_cuda_mlir.extending import typeof_impl

        baseline = dict(typeof_impl.registry)
        real_import_module = importlib.import_module
        fail_once = True

        def fail_after_rewrite(name, package=None):
            global fail_once
            if fail_once and name.endswith("._compiler._group_planner"):
                fail_once = False
                raise RuntimeError("injected post-types activation failure")
            return real_import_module(name, package)

        importlib.import_module = fail_after_rewrite
        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except RuntimeError as error:
            assert str(error) == "injected post-types activation failure"
        else:
            raise AssertionError("injected activation failure did not occur")
        finally:
            importlib.import_module = real_import_module

        assert dict(typeof_impl.registry) == baseline

        import cuda.coop.numba_mlir  # noqa: F401

        assert dict(typeof_impl.registry) == baseline
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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
