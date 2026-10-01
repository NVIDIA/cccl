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

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

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


def test_numba_first_root_import_does_not_require_refresh_registries():
    _run_import_probe(
        """
        import sys

        import numba_cuda_mlir
        import numba_cuda_mlir.extending as extending

        del extending.refresh_registries
        import cuda.coop  # noqa: F401

        assert "cuda.coop.numba_mlir" in sys.modules
        assert numba_cuda_mlir.__version__.startswith("0.5.")
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


@pytest.mark.parametrize("backend", ["numba-cuda-mlir", "numba_cuda_mlir"])
@pytest.mark.parametrize("disable_auto_registration", ["0", "1"])
def test_root_first_register_explicitly_activates_backend_once(
    backend, disable_auto_registration
):
    _run_import_probe(
        f"""
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = (
            {disable_auto_registration!r}
        )
        from cuda import coop

        assert "numba_cuda_mlir" not in sys.modules
        assert "cuda.bindings" not in sys.modules
        assert "cuda.coop.numba_mlir" not in sys.modules
        assert coop.register({backend!r}) is None

        expected = {{
            "cuda.coop.numba_mlir",
            "cuda.coop.numba_mlir._compiler._group_planner",
            "cuda.coop.numba_mlir._compiler._rewrite",
        }}
        assert expected <= set(sys.modules), expected - set(sys.modules)

        from cuda.coop.numba_mlir._compiler._activation import (
            _snapshot_registrations,
        )

        snapshot = _snapshot_registrations()
        assert coop.register("numba-cuda-mlir") is None
        assert coop.register("numba_cuda_mlir") is None
        assert _snapshot_registrations() == snapshot
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


@pytest.mark.parametrize("error_type", ["RuntimeError", "AttributeError"])
def test_failed_activation_after_types_import_leaves_typeof_registry_unchanged(
    error_type,
):
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
        injected_error = ERROR_TYPE("injected post-types activation failure")

        def fail_after_rewrite(name, package=None):
            global fail_once
            if fail_once and name.endswith("._compiler._group_planner"):
                fail_once = False
                raise injected_error
            return real_import_module(name, package)

        importlib.import_module = fail_after_rewrite
        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except ERROR_TYPE as error:
            assert error is injected_error
        else:
            raise AssertionError("injected activation failure did not occur")
        finally:
            importlib.import_module = real_import_module

        assert dict(typeof_impl.registry) == baseline

        import cuda.coop.numba_mlir  # noqa: F401

        assert dict(typeof_impl.registry) == baseline
        """.replace("ERROR_TYPE", error_type)
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


@pytest.mark.parametrize(
    "activate",
    [
        "import cuda.coop.numba_mlir",
        'from cuda import coop; coop.register("numba-cuda-mlir")',
    ],
)
@pytest.mark.parametrize(
    "version", ["0.4.9", "0.5.0rc1", "0.6.0rc1", "0.6.0", "invalid", None]
)
def test_backend_registration_rejects_unsupported_runtime_version(
    activate, version
):
    _run_import_probe(
        """
        import importlib.metadata
        import os
        import sys
        from types import ModuleType

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        runtime = ModuleType("numba_cuda_mlir")
        runtime.__version__ = VERSION
        sys.modules[runtime.__name__] = runtime
        real_version = importlib.metadata.version

        def distribution_version(name):
            if name == "numba-cuda-mlir":
                raise importlib.metadata.PackageNotFoundError(name)
            return real_version(name)

        importlib.metadata.version = distribution_version
        try:
            ACTIVATE_BACKEND
        except ImportError as exc:
            assert exc.backend == "numba-cuda-mlir"
            assert exc.reason_code == "unsupported-runtime-version"
            assert exc.details["detected_version"] == VERSION
            assert exc.details["required_version"] == ">=0.5.0,<0.6"
            message = str(exc)
            assert "numba-cuda-mlir>=0.5.0,<0.6" in message
            assert "cuda-coop[numba-cuda-mlir-cu12]" in message
            assert "cuda-coop[numba-cuda-mlir-cu13]" in message
            if VERSION is not None:
                assert str(VERSION) in message
        else:
            raise AssertionError("unsupported runtime unexpectedly activated")

        assert "numba_cuda_mlir.cuda" not in sys.modules
        assert "cuda.coop.numba_mlir._compiler._rewrite" not in sys.modules
        assert (
            "cuda.coop.numba_mlir._compiler._group_planner" not in sys.modules
        )
        """.replace("ACTIVATE_BACKEND", activate).replace(
            "VERSION", repr(version)
        )
    )


@pytest.mark.parametrize("version", ["0.5.0+local", "0.5.1rc1", None])
def test_backend_registration_accepts_supported_runtime_version(version):
    _run_import_probe(
        f"""
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        import numba_cuda_mlir

        # An absent module version falls back to installed package metadata.
        version = {version!r}
        if version is None:
            del numba_cuda_mlir.__version__
        else:
            numba_cuda_mlir.__version__ = version

        from cuda import coop

        coop.register("numba-cuda-mlir")
        assert "cuda.coop.numba_mlir._compiler._rewrite" in sys.modules
        assert "cuda.coop.numba_mlir._compiler._group_planner" in sys.modules
        """
    )


def test_root_import_ignores_an_installed_but_unused_old_runtime(tmp_path):
    runtime = tmp_path / "numba_cuda_mlir"
    runtime.mkdir()
    (runtime / "__init__.py").write_text('__version__ = "0.4.9"\n')
    _run_import_probe(
        f"""
        import importlib.util
        import sys
        import warnings

        sys.path.insert(0, {str(tmp_path)!r})
        assert importlib.util.find_spec("numba_cuda_mlir") is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop

        assert callable(coop.ThreadData)
        assert not caught, caught
        assert "numba_cuda_mlir" not in sys.modules
        assert "cuda.coop.numba_mlir" not in sys.modules
        """
    )


def test_root_import_warns_when_an_old_runtime_was_already_imported():
    _run_import_probe(
        """
        import sys
        import warnings
        from types import ModuleType

        runtime = ModuleType("numba_cuda_mlir")
        runtime.__version__ = "0.4.9"
        sys.modules[runtime.__name__] = runtime

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop

        from cuda.coop._core._auto_registration import (
            CudaCoopAutoRegistrationWarning,
        )

        assert callable(coop.ThreadData)
        assert len(caught) == 1, caught
        assert caught[0].category is CudaCoopAutoRegistrationWarning
        assert "0.4.9" in str(caught[0].message)
        assert "numba-cuda-mlir>=0.5.0,<0.6" in str(caught[0].message)
        assert "cuda.coop.numba_mlir" not in sys.modules
        assert "numba_cuda_mlir.cuda" not in sys.modules
        """
    )


def test_missing_packaging_is_reported_before_compiler_import():
    _run_import_probe(
        """
        import importlib.abc
        import os
        import sys
        from types import ModuleType

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        runtime = ModuleType("numba_cuda_mlir")
        runtime.__version__ = "0.5.0"
        sys.modules[runtime.__name__] = runtime
        missing = ModuleNotFoundError(
            "No module named 'packaging'", name="packaging"
        )

        class BlockPackaging(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "packaging":
                    raise missing
                return None

        assert "packaging" not in sys.modules
        sys.meta_path.insert(0, BlockPackaging())
        from cuda import coop

        try:
            coop.register("numba-cuda-mlir")
        except ImportError as exc:
            assert exc.reason_code == "backend-dependency-missing"
            assert exc.details["missing"] == "packaging"
            assert exc.__cause__ is missing
            message = str(exc)
            assert "packaging" in message
            assert "cuda-coop[numba-cuda-mlir-cu12]" in message
            assert "cuda-coop[numba-cuda-mlir-cu13]" in message
        else:
            raise AssertionError(
                "activation without packaging unexpectedly succeeded"
            )

        assert "numba_cuda_mlir.cuda" not in sys.modules
        assert "cuda.coop.numba_mlir._compiler._rewrite" not in sys.modules
        assert (
            "cuda.coop.numba_mlir._compiler._group_planner" not in sys.modules
        )
        """
    )


@pytest.mark.parametrize(
    "activate",
    [
        "import cuda.coop.numba_mlir",
        'from cuda import coop; coop.register("numba-cuda-mlir")',
    ],
)
def test_backend_registration_reports_a_missing_public_runtime(activate):
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
            ACTIVATE_BACKEND
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
        """.replace("ACTIVATE_BACKEND", activate)
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
