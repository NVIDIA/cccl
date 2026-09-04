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

from cuda.coop.numba_mlir._compiler import _activation, _numba_mlir_compat

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


def _fake_compat_modules():
    planner_registry = SimpleNamespace(_lock=RLock(), _planners=[])
    rewrite_registry = SimpleNamespace(
        rewrites={"before-inference": [], "after-inference": []}
    )
    overload_base = type("_OverloadFunctionTemplate", (), {})

    def get_jit_decorator(self):
        del self

    overload_template = type(
        "_NumbaCudaMlirOverloadFunctionTemplate",
        (overload_base,),
        {"_get_jit_decorator": get_jit_decorator},
    )

    def make_overload_template(
        func,
        overload_func,
        jit_options,
        strict,
        inline,
        prefer_literal=False,
        base=overload_base,
        **kwargs,
    ):
        return (
            func,
            overload_func,
            jit_options,
            strict,
            inline,
            prefer_literal,
            base,
            kwargs,
        )

    extending = SimpleNamespace(
        WholeFunctionPlanner=type("WholeFunctionPlanner", (), {}),
        register_planner=lambda planner: planner,
        require_launch_config=lambda state: {},
        set_required_dynamic_shared_memory=lambda state, size: None,
        _NumbaCudaMlirOverloadFunctionTemplate=overload_template,
    )
    return {
        "numba_cuda_mlir.extending": extending,
        "numba_cuda_mlir._whole_function_planners": SimpleNamespace(
            _planner_registry=planner_registry
        ),
        "numba_cuda_mlir.numba_cuda.core.rewrites": SimpleNamespace(
            Rewrite=type("Rewrite", (), {}),
            register_rewrite=lambda kind: lambda rewrite: rewrite,
            rewrite_registry=rewrite_registry,
        ),
        "numba_cuda_mlir.numba_cuda.core.errors": SimpleNamespace(
            ConstantInferenceError=type("ConstantInferenceError", (Exception,), {})
        ),
        "numba_cuda_mlir.numba_cuda.typing.typeof": SimpleNamespace(
            typeof=lambda value: value
        ),
        "numba_cuda_mlir.numba_cuda.typing.templates": SimpleNamespace(
            _OverloadFunctionTemplate=overload_base,
            make_overload_template=make_overload_template,
        ),
    }


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


def test_compat_accepts_public_0_5_capabilities_without_refresh(monkeypatch):
    modules = _fake_compat_modules()
    extending = modules["numba_cuda_mlir.extending"]
    assert not hasattr(extending, "refresh_registries")
    monkeypatch.setattr(
        _numba_mlir_compat.importlib,
        "import_module",
        lambda name: modules[name],
    )

    compat = _numba_mlir_compat._load_numba_mlir_compat(
        SimpleNamespace(__version__="0.5.99")
    )

    assert compat.version == "0.5.99"
    snapshot = compat.snapshot_registrations()
    assert snapshot.planners == ()
    assert snapshot.rewrites["before-inference"] == ()


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        (None, False),
        ("", False),
        ("0.4.99", False),
        ("0.5", True),
        ("0.5.0", True),
        ("0.5.99", True),
        ("0.6", False),
        ("1.0.0", False),
    ],
)
def test_compat_owns_runtime_version_predicate(version, supported):
    assert _numba_mlir_compat._is_supported_runtime_version(version) is supported


def test_compat_runtime_requirement_uses_detected_version():
    requirement = _numba_mlir_compat._runtime_requirement(
        SimpleNamespace(__version__="0.5.7")
    )

    assert "detected numba-cuda-mlir==0.5.7" in requirement
    assert "numba-cuda-mlir>=0.5.0,<0.6" in requirement
    assert "cuda-coop[numba-cuda-mlir-cu12]" in requirement
    assert "cuda-coop[numba-cuda-mlir-cu13]" in requirement


def test_activation_consumes_compat_runtime_requirement(monkeypatch):
    diagnostic = "compat-owned runtime diagnostic"

    def missing_runtime(name):
        assert name == "numba_cuda_mlir"
        raise ImportError("missing runtime", name=name)

    monkeypatch.setattr(_activation.importlib, "import_module", missing_runtime)
    monkeypatch.setattr(
        _activation,
        "_runtime_requirement",
        lambda runtime=None: diagnostic,
    )

    runtime, error = _activation._load_runtime()

    assert runtime is None
    assert error is not None
    assert error.reason_code == "backend-runtime-missing"
    assert diagnostic in str(error)


def test_compat_wraps_required_module_import_failures(monkeypatch):
    modules = _fake_compat_modules()

    def import_module(name):
        if name == "numba_cuda_mlir.extending":
            raise ImportError("missing extending module", name=name)
        return modules[name]

    monkeypatch.setattr(
        _numba_mlir_compat.importlib,
        "import_module",
        import_module,
    )

    with pytest.raises(_activation._NumbaMlirBackendImportError) as exc_info:
        _numba_mlir_compat._load_numba_mlir_compat(SimpleNamespace(__version__="0.5.1"))

    error = exc_info.value
    assert error.reason_code == "runtime-hook-api-import-failed"
    assert error.details["module"] == "numba_cuda_mlir.extending"
    assert isinstance(error.__cause__, ImportError)


@pytest.mark.parametrize("version", ["0.4.9", "0.6.0", "1.0.0"])
def test_compat_rejects_unsupported_runtime_series(version):
    with pytest.raises(_activation._NumbaMlirBackendImportError) as exc_info:
        _numba_mlir_compat._load_numba_mlir_compat(SimpleNamespace(__version__=version))

    error = exc_info.value
    assert error.reason_code == "unsupported-runtime-version"
    assert error.details == {
        "detected_version": version,
        "supported_series": "0.5.x",
    }


@pytest.mark.parametrize(
    ("module_name", "attribute", "replacement", "reason_code"),
    [
        (
            "numba_cuda_mlir.extending",
            "_NumbaCudaMlirOverloadFunctionTemplate",
            None,
            "incomplete-runtime-hook-api",
        ),
        (
            "numba_cuda_mlir.extending",
            "_NumbaCudaMlirOverloadFunctionTemplate",
            type(
                "MalformedOverloadTemplate",
                (),
                {"_get_jit_decorator": lambda self: None},
            ),
            "incomplete-runtime-hook-api",
        ),
        (
            "numba_cuda_mlir.numba_cuda.typing.templates",
            "make_overload_template",
            lambda func: func,
            "incomplete-runtime-hook-api",
        ),
        (
            "numba_cuda_mlir._whole_function_planners",
            "_planner_registry",
            SimpleNamespace(_lock=RLock(), _planners=()),
            "registration-transaction-unavailable",
        ),
    ],
)
def test_compat_rejects_malformed_private_0_5_shapes(
    monkeypatch,
    module_name,
    attribute,
    replacement,
    reason_code,
):
    modules = _fake_compat_modules()
    setattr(modules[module_name], attribute, replacement)
    monkeypatch.setattr(
        _numba_mlir_compat.importlib,
        "import_module",
        lambda name: modules[name],
    )

    with pytest.raises(_activation._NumbaMlirBackendImportError) as exc_info:
        _numba_mlir_compat._load_numba_mlir_compat(SimpleNamespace(__version__="0.5.1"))

    assert exc_info.value.reason_code == reason_code


def test_qualified_import_rejects_unsupported_runtime_series():
    _run_import_probe(
        """
        import os

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        import numba_cuda_mlir

        numba_cuda_mlir.__version__ = "0.6.0"
        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "numba-cuda-mlir"
            assert exc.reason_code == "unsupported-runtime-version"
            assert exc.details["detected_version"] == "0.6.0"
            assert "supports numba-cuda-mlir 0.5.x" in str(exc)
        else:
            raise AssertionError("unsupported runtime unexpectedly activated")
        """
    )


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
