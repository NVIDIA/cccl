# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from ...support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]

_CUTLASS_RUNTIME_STUB = """
import sys
import types

cutlass_runtime = types.ModuleType("cutlass")
cutlass_runtime.__path__ = []
cutlass_dsl_runtime = types.ModuleType("cutlass.cutlass_dsl")

class CuTeDSL:
    _instance = None

    @classmethod
    def _get_dsl(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.trace_context_factories = []

    def register_trace_context_factory(self, factory):
        if factory not in self.trace_context_factories:
            self.trace_context_factories.append(factory)

    def register_trace_finalize_hook(self, hook):
        self.trace_finalize_hook = hook

cutlass_dsl_runtime.CuTeDSL = CuTeDSL
cute_runtime = types.ModuleType("cutlass.cute")
cute_runtime._get_launch_facts = lambda: {}
base_dsl_runtime = types.ModuleType("cutlass.base_dsl")
base_dsl_runtime.__path__ = []
compiler_runtime = types.ModuleType("cutlass.base_dsl.compiler")
compiler_runtime.LinkLibraries = type(
    "LinkLibraries",
    (),
    {"_option_name": "link-libraries"},
)
compiler_runtime.GPUArch = type("GPUArch", (), {})
cutlass_runtime.cutlass_dsl = cutlass_dsl_runtime
cutlass_runtime.cute = cute_runtime
cutlass_runtime.base_dsl = base_dsl_runtime
base_dsl_runtime.compiler = compiler_runtime
sys.modules["cutlass"] = cutlass_runtime
sys.modules["cutlass.cutlass_dsl"] = cutlass_dsl_runtime
sys.modules["cutlass.cute"] = cute_runtime
sys.modules["cutlass.base_dsl"] = base_dsl_runtime
sys.modules["cutlass.base_dsl.compiler"] = compiler_runtime
"""


def _run_source(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{PACKAGE_ROOT}{os.pathsep}{python_path}" if python_path else str(PACKAGE_ROOT)
    )
    return subprocess.run(
        [sys.executable, "-S", "-B", "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _assert_source_passes(script: str) -> None:
    result = _run_source(script)
    assert result.returncode == 0, (
        f"source subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _with_fake_cutlass_runtime(script: str) -> str:
    return _CUTLASS_RUNTIME_STUB + textwrap.dedent(script)


def test_opted_out_common_root_remains_cutlass_import_light() -> None:
    _assert_source_passes(
        """
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert not any(
            name == "cutlass" or name.startswith("cutlass.")
            for name in sys.modules
        )
        assert not any(
            name == "cuda.coop.cutlass" or name.startswith("cuda.coop.cutlass.")
            for name in sys.modules
        )
        """
    )


@pytest.mark.parametrize(
    ("setup", "reason_code"),
    [
        ("", "backend-runtime-missing"),
        (
            """
            import sys
            import types
            runtime = types.ModuleType("cutlass")
            runtime.__path__ = []
            sys.modules["cutlass"] = runtime
            """,
            "conflicting-backend-runtime",
        ),
    ],
)
def test_qualified_import_reports_structured_runtime_errors(
    setup: str,
    reason_code: str,
) -> None:
    _assert_source_passes(
        textwrap.dedent(setup)
        + textwrap.dedent(
            f"""
        try:
            import cuda.coop.cutlass
        except ImportError as error:
            assert error.backend == "cutlass"
            assert error.reason_code == {reason_code!r}
            assert isinstance(error.details, dict)
        else:
            raise AssertionError("qualified import unexpectedly succeeded")
        """
        )
    )


def test_qualified_import_activates_common_root_fallback() -> None:
    _assert_source_passes(
        _with_fake_cutlass_runtime(
            """
        import cuda.coop.cutlass as cutlass_coop
        from cuda import coop

        cutlass_coop.this_block = lambda *args, **kwargs: "cutlass-block"
        cutlass_coop.sum = lambda group, value, **kwargs: (group, value)

        group = coop.this_block()
        assert group == "cutlass-block"
        assert coop.sum(group, 7) == ("cutlass-block", 7)
        """
        )
    )


def test_failed_qualified_import_does_not_leave_common_root_fallback() -> None:
    _assert_source_passes(
        """
        import sys
        import types

        from cuda.coop._core import root_api

        runtime_dependency = types.ModuleType(
            "cuda.coop.cutlass._runtime_dependency"
        )
        runtime_dependency.validate_cutlass_runtime = lambda **kwargs: None
        sys.modules[runtime_dependency.__name__] = runtime_dependency

        compiler_activation = types.ModuleType("cuda.coop.cutlass._compiler")

        def fail_trace_context_registration():
            raise RuntimeError("injected trace-context registration failure")

        compiler_activation.register_trace_context = fail_trace_context_registration
        sys.modules[compiler_activation.__name__] = compiler_activation

        try:
            import cuda.coop.cutlass
        except RuntimeError as error:
            assert str(error) == "injected trace-context registration failure"
        else:
            raise AssertionError("qualified import unexpectedly succeeded")

        assert "cuda.coop.cutlass" not in sys.modules
        assert root_api._backend_module_name() is None
        """
    )


def test_qualified_import_structures_non_import_runtime_failures() -> None:
    _assert_source_passes(
        """
        import importlib

        real_import_module = importlib.import_module

        def broken_import(name, package=None):
            if name == "cutlass.cutlass_dsl":
                raise RuntimeError("broken binary runtime")
            return real_import_module(name, package)

        importlib.import_module = broken_import
        try:
            import cuda.coop.cutlass
        except ImportError as error:
            assert error.reason_code == "transitive-runtime-import-failed"
            assert error.details["exception_type"] == "RuntimeError"
            assert isinstance(error.__cause__, RuntimeError)
        else:
            raise AssertionError("qualified import unexpectedly succeeded")
        """
    )


def test_common_grid_sync_is_rejected_without_disabling_qualified_cutlass() -> None:
    _assert_source_passes(
        _with_fake_cutlass_runtime(
            """
        from cuda import coop as common
        import cuda.coop.cutlass as qualified

        common_grid = common.this_grid()

        assert isinstance(common_grid, qualified.ThreadGroup)
        assert common_grid.source == "common_root"
        qualified_grid = qualified.this_grid()
        assert qualified_grid.source != "common_root"

        try:
            common.discontinuity(
                common.this_block(), object(), mode="heads_and_tails"
            )
        except ValueError as error:
            assert str(error) == (
                "cuda.coop.discontinuity mode must be one of: heads, tails; "
                "use a backend-qualified import for backend-only controls"
            )
        else:
            raise AssertionError("qualified-only discontinuity mode was accepted")

        for method_name in ("sync", "sync_aligned"):
            try:
                getattr(common_grid, method_name)()
            except NotImplementedError as error:
                assert str(error) == (
                    f"cuda.coop.ThreadGroup.{method_name} grid synchronization "
                    "is unavailable through the common V1 profile; use "
                    "cuda.coop.cutlass.this_grid() under a verified cooperative "
                    "launch"
                )
            else:
                raise AssertionError(f"common grid {method_name} unexpectedly succeeded")

        import sys
        import types
        from cuda.coop._core import LaunchFactOrigin, LaunchFacts
        from cuda.coop.cutlass import _dsl

        launch_module = types.ModuleType("cuda.coop.cutlass._dsl._launch")
        launch_module.current_kernel_launch_facts = lambda: LaunchFacts(
            exact_block_dim=32,
            exact_grid_dim=2,
            cooperative_launch=True,
            cluster_launch=False,
            provenance=(
                LaunchFactOrigin(
                    fact="cooperative_launch",
                    source="test_compiler",
                    verified=True,
                ),
                LaunchFactOrigin(
                    fact="cluster_launch",
                    source="test_compiler",
                    verified=True,
                ),
            ),
        )
        sync_calls = []
        provider_module = types.ModuleType(
            "cuda.coop.cutlass._dsl._thread_group_provider"
        )
        provider_module.provider_group_sync = lambda **kwargs: sync_calls.append(kwargs)
        sys.modules[launch_module.__name__] = launch_module
        sys.modules[provider_module.__name__] = provider_module
        _dsl._thread_group_provider = provider_module

        qualified_grid.sync()
        qualified_grid.sync_aligned()
        assert [call["aligned"] for call in sync_calls] == [False, True]
        assert all(call["group"].kind == "grid" for call in sync_calls)
        """
        )
    )


def test_temp_storage_inference_and_validation_match_the_common_contract() -> None:
    _assert_source_passes(
        _with_fake_cutlass_runtime(
            """
        import cuda.coop.cutlass as coop

        inferred = coop.TempStorage()
        assert inferred.is_deferred
        assert inferred.size_in_bytes is None
        assert inferred.alignment is None
        assert inferred.auto_sync is True
        assert inferred.sharing == "shared"

        manual = coop.TempStorage(auto_sync=False)
        assert manual.auto_sync is False
        exclusive = coop.TempStorage(sharing=" EXCLUSIVE ")
        assert exclusive.sharing == "exclusive"
        assert exclusive.auto_sync is False

        cases = (
            ({"size_in_bytes": True}, TypeError, "integer or None"),
            ({"size_in_bytes": 0}, ValueError, "positive integer"),
            ({"alignment": True}, TypeError, "integer or None"),
            ({"alignment": 3}, ValueError, "power of 2"),
            ({"auto_sync": 1}, TypeError, "None/True/False"),
            ({"sharing": object()}, TypeError, "sharing must be a string"),
            ({"sharing": "private"}, ValueError, "'shared' or 'exclusive'"),
            (
                {"sharing": "exclusive", "auto_sync": True},
                ValueError,
                "does not support auto_sync=True",
            ),
        )
        for kwargs, error_type, message in cases:
            try:
                coop.TempStorage(**kwargs)
            except error_type as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(f"TempStorage accepted {kwargs!r}")
        """
        )
    )
