# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from ._imports import SOURCE_ROOT, run_python_with_source

_COMPATIBLE_CUTLASS = """
import sys
import types

cutlass = types.ModuleType("cutlass")
cutlass.__path__ = []
cutlass.__version__ = "99.1"
cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")

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
        self.hook = hook

cutlass_dsl.CuTeDSL = CuTeDSL
cute = types.ModuleType("cutlass.cute")
cute._get_launch_facts = lambda: {}
base_dsl = types.ModuleType("cutlass.base_dsl")
base_dsl.__path__ = []
compiler = types.ModuleType("cutlass.base_dsl.compiler")
compiler.LinkLibraries = type(
    "LinkLibraries",
    (),
    {"_option_name": "link-libraries"},
)
compiler.GPUArch = type("GPUArch", (), {})
cutlass.cutlass_dsl = cutlass_dsl
cutlass.cute = cute
cutlass.base_dsl = base_dsl
base_dsl.compiler = compiler
sys.modules["cutlass"] = cutlass
sys.modules["cutlass.cutlass_dsl"] = cutlass_dsl
sys.modules["cutlass.cute"] = cute
sys.modules["cutlass.base_dsl"] = base_dsl
sys.modules["cutlass.base_dsl.compiler"] = compiler
"""

_COMPATIBLE_NUMBA = """
import threading

runtime = types.ModuleType("numba_cuda_mlir")
runtime.__path__ = []
runtime.__version__ = "99.2"
cuda = types.ModuleType("numba_cuda_mlir.cuda")
extending = types.ModuleType("numba_cuda_mlir.extending")
extending.WholeFunctionPlanner = type("WholeFunctionPlanner", (), {})
extending.refresh_registries = lambda: None
extending.register_planner = lambda planner: planner
extending.require_launch_config = lambda state: {}
extending.set_required_dynamic_shared_memory = lambda state, size: None
numba_cuda = types.ModuleType("numba_cuda_mlir.numba_cuda")
numba_cuda.__path__ = []
core = types.ModuleType("numba_cuda_mlir.numba_cuda.core")
core.__path__ = []
rewrites = types.ModuleType("numba_cuda_mlir.numba_cuda.core.rewrites")
rewrites.Rewrite = type("Rewrite", (), {})
rewrites.register_rewrite = lambda phase: lambda rewrite: rewrite
planners = types.ModuleType("numba_cuda_mlir._whole_function_planners")

group_planner_type = type("CoopGroupHierarchyPlanner", (), {})
whole_planner_type = type("CoopWholeFunctionPlanner", (), {})
rewrite_type = type("CoopSinglePhaseRewrite", (), {})
planners._planner_registry = types.SimpleNamespace(
    _lock=threading.RLock(),
    _planners=[group_planner_type, whole_planner_type],
)
rewrites.rewrite_registry = types.SimpleNamespace(
    rewrites={"before-inference": [rewrite_type]},
)
single_phase = types.ModuleType("cuda.coop.numba_mlir._single_phase_rewrites")
single_phase.CoopWholeFunctionPlanner = whole_planner_type
single_phase.CoopSinglePhaseRewrite = rewrite_type
group_rewrites = types.ModuleType("cuda.coop.numba_mlir._group_rewrites")
group_rewrites.CoopGroupHierarchyPlanner = group_planner_type

runtime.cuda = cuda
runtime.numba_cuda = numba_cuda
numba_cuda.core = core
core.rewrites = rewrites
sys.modules["numba_cuda_mlir"] = runtime
sys.modules["numba_cuda_mlir.cuda"] = cuda
sys.modules["numba_cuda_mlir.extending"] = extending
sys.modules["numba_cuda_mlir.numba_cuda"] = numba_cuda
sys.modules["numba_cuda_mlir.numba_cuda.core"] = core
sys.modules["numba_cuda_mlir.numba_cuda.core.rewrites"] = rewrites
sys.modules["numba_cuda_mlir._whole_function_planners"] = planners
sys.modules["cuda.coop.numba_mlir._single_phase_rewrites"] = single_phase
sys.modules["cuda.coop.numba_mlir._group_rewrites"] = group_rewrites
"""


def _assert_source_passes(script: str) -> None:
    script = (
        "import os\n"
        'os.environ.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)\n'
        + textwrap.dedent(script)
    )
    result = run_python_with_source(script, inherit_pythonpath=False)
    assert result.returncode == 0, (
        f"source subprocess failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_known_dsl_registry_is_fixed() -> None:
    _assert_source_passes(
        """
        import os
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        from cuda.coop._core import _auto_registration

        assert _auto_registration._AUTO_DSL_CANDIDATES == (
            "cutlass",
            "numba_mlir",
        )
        """
    )


@pytest.mark.parametrize("value", [None, "", "0", "FALSE", " no ", "Off"])
def test_false_environment_values_keep_auto_registration_enabled(value: str | None):
    _assert_source_passes(
        f"""
        import os
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        from cuda.coop._core import _auto_registration

        value = {value!r}
        if value is None:
            os.environ.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION")
            assert not _auto_registration._auto_registration_disabled()
        else:
            assert not _auto_registration._auto_registration_disabled(value)
        """
    )


@pytest.mark.parametrize("value", ["1", "true", "YES", "enabled", "-1"])
def test_nonempty_environment_values_disable_auto_registration(value: str):
    _assert_source_passes(
        f"""
        import os
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        from cuda.coop._core import _auto_registration

        assert _auto_registration._auto_registration_disabled({value!r})
        """
    )


def test_opt_out_preserves_cold_root_and_never_probes_dsls() -> None:
    _assert_source_passes(
        """
        import importlib.abc
        import os
        import sys

        class RejectAllDSLs(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname in {"cutlass", "numba_cuda_mlir"}:
                    raise AssertionError(f"unexpected DSL probe: {fullname}")
                return None

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "true"
        sys.meta_path.insert(0, RejectAllDSLs())
        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        """
    )


def test_missing_runtimes_are_silent_and_leave_no_qualified_module_debris() -> None:
    _assert_source_passes(
        """
        import importlib.abc
        import sys
        import warnings

        attempted = []

        class BlockOptionalRuntimes(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname in {"cutlass", "numba_cuda_mlir"}:
                    attempted.append(fullname)
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockOptionalRuntimes())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop

        assert not caught
        assert attempted == ["cutlass", "numba_cuda_mlir"]
        assert coop.__name__ == "cuda.coop"
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in (
                "cuda.coop.cutlass",
                "cuda.coop.numba_mlir",
            )
            for name in sys.modules
        )
        """
    )


@pytest.mark.parametrize(
    ("setup", "backend", "dependency", "extra"),
    [
        pytest.param(
            """
            cutlass = types.ModuleType("cutlass")
            cutlass.__path__ = []
            sys.modules["cutlass"] = cutlass

            class BreakCutlass(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if fullname == "cutlass.cutlass_dsl":
                        raise ImportError(
                            "broken CUTLASS native dependency",
                            name="cutlass_native",
                        )
                    return None

            sys.meta_path.insert(0, BreakCutlass())
            """,
            "CUTLASS",
            "cutlass_native",
            "cuda-coop[cutlass]",
            id="cutlass",
        ),
        pytest.param(
            """
            class BreakNumba(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if fullname == "numba_cuda_mlir":
                        raise ImportError(
                            "broken Numba native dependency",
                            name="numba_native",
                        )
                    return None

            sys.meta_path.insert(0, BreakNumba())
            """,
            "Numba-CUDA-MLIR",
            "numba_native",
            "cuda-coop[numba-cuda-mlir]",
            id="numba-mlir",
        ),
    ],
)
def test_transitive_runtime_failures_warn_and_leave_root_available(
    setup: str,
    backend: str,
    dependency: str,
    extra: str,
) -> None:
    _assert_source_passes(
        textwrap.dedent(
            """
        import importlib.abc
        import sys
        import types
        import warnings
        """
        )
        + textwrap.dedent(setup)
        + textwrap.dedent(
            f"""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from cuda import coop

            assert coop.__name__ == "cuda.coop"
            assert len(caught) == 1
            message = str(caught[0].message)
            assert message.startswith("cuda.coop automatic DSL registration:")
            assert {backend!r} in message
            assert {dependency!r} in message
            assert {extra!r} in message
            """
        )
    )


def test_compatible_cutlass_is_enabled_as_the_common_root_fallback() -> None:
    _assert_source_passes(
        _COMPATIBLE_CUTLASS
        + textwrap.dedent(
            """
        from cuda import coop
        from cuda.coop._core import root_api

        assert coop.__name__ == "cuda.coop"
        assert root_api._backend_module_name() == "cuda.coop.cutlass"
        """
        )
    )


def test_compatible_cutlass_registers_one_compiler_owned_trace_context() -> None:
    _assert_source_passes(
        _COMPATIBLE_CUTLASS
        + textwrap.dedent(
            """
        import importlib

        from cuda import coop
        from cuda.coop._core import root_api
        import cuda.coop.cutlass as cutlass_coop

        dsl = CuTeDSL._get_dsl()
        assert len(dsl.trace_context_factories) == 1
        factory = dsl.trace_context_factories[0]

        with root_api._compiler_scope("cuda.coop.numba_mlir"):
            assert root_api._backend_module_name() == "cuda.coop.numba_mlir"
            with factory(dsl, "kernel"):
                assert root_api._backend_module_name() == "cuda.coop.cutlass"
            assert root_api._backend_module_name() == "cuda.coop.numba_mlir"

        importlib.reload(cutlass_coop)
        assert dsl.trace_context_factories == [factory]
        assert coop.__name__ == "cuda.coop"
        """
        )
    )


def test_compatible_numba_registers_when_cutlass_is_absent() -> None:
    _assert_source_passes(
        "import sys\nimport types\n"
        + _COMPATIBLE_NUMBA
        + textwrap.dedent(
            """
        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert "cuda.coop.cutlass" not in sys.modules
        assert "cuda.coop.numba_mlir" in sys.modules
        assert planners._planner_registry._planners == [
            group_planner_type,
            whole_planner_type,
        ]
        assert rewrites.rewrite_registry.rewrites["before-inference"] == [
            rewrite_type,
        ]
        """
        )
    )


def test_compatible_cutlass_and_numba_are_both_enabled() -> None:
    _assert_source_passes(
        _COMPATIBLE_CUTLASS
        + _COMPATIBLE_NUMBA
        + textwrap.dedent(
            """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop
        from cuda.coop._core import root_api

        assert not caught
        assert root_api._backend_module_name() == "cuda.coop.cutlass"
        assert "cuda.coop.numba_mlir" in sys.modules
        """
        )
    )


def test_all_compatible_dsl_adapters_are_enabled_without_changing_fallback() -> None:
    _assert_source_passes(
        _COMPATIBLE_CUTLASS
        + _COMPATIBLE_NUMBA
        + textwrap.dedent(
            """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop
        from cuda.coop._core import root_api

        assert not caught
        assert root_api._backend_module_name() == "cuda.coop.cutlass"
        assert "cuda.coop.numba_mlir" in sys.modules
        """
        )
    )


def test_auto_registration_order_is_fixed_and_root_reload_is_idempotent() -> None:
    _assert_source_passes(
        _COMPATIBLE_CUTLASS
        + _COMPATIBLE_NUMBA
        + textwrap.dedent(
            """
            import importlib
            import os
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                from cuda import coop
                importlib.reload(coop)

            from cuda.coop._core import root_api
            from cuda.coop._core import _auto_registration

            assert not caught
            assert _auto_registration._AUTO_DSL_CANDIDATES == (
                "cutlass",
                "numba_mlir",
            )
            assert root_api._backend_module_name() == "cuda.coop.cutlass"
            assert planners._planner_registry._planners == [
                group_planner_type,
                whole_planner_type,
            ]
            assert rewrites.rewrite_registry.rewrites["before-inference"] == [
                rewrite_type,
            ]
            """
        )
    )


def test_incompatible_cutlass_warns_once_and_numba_still_registers() -> None:
    incompatible = _COMPATIBLE_CUTLASS.replace(
        "    def register_trace_finalize_hook(self, hook):\n"
        "        self.hook = hook\n\n",
        "",
    )
    _assert_source_passes(
        incompatible
        + _COMPATIBLE_NUMBA
        + textwrap.dedent(
            """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop
        from cuda.coop._core._auto_registration import (
            CudaCoopAutoRegistrationWarning,
        )

        assert len(caught) == 1
        warning = caught[0]
        assert warning.category is CudaCoopAutoRegistrationWarning
        message = str(warning.message)
        assert message.startswith("cuda.coop automatic DSL registration:")
        assert "CUTLASS" in message
        assert "register_trace_finalize_hook" in message
        assert "cuda-coop[cutlass]" in message
        assert "https://nvidia.github.io/cccl" in message
        assert "installation instructions" in message
        assert "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION=1" in message
        assert "cuda.coop.numba_mlir" in sys.modules
        assert "cuda.coop.cutlass" not in sys.modules
        assert coop.__name__ == "cuda.coop"
        """
        )
    )


def test_noncallable_cutlass_link_libraries_warns_and_does_not_activate() -> None:
    incompatible = _COMPATIBLE_CUTLASS.replace(
        "compiler.LinkLibraries = type(\n"
        '    "LinkLibraries",\n'
        "    (),\n"
        '    {"_option_name": "link-libraries"},\n'
        ")",
        "compiler.LinkLibraries = object()",
    )
    assert incompatible != _COMPATIBLE_CUTLASS
    _assert_source_passes(
        incompatible
        + textwrap.dedent(
            """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert len(caught) == 1
        assert "cutlass.base_dsl.compiler.LinkLibraries" in str(
            caught[0].message
        )
        assert "cuda.coop.cutlass" not in sys.modules
        """
        )
    )


@pytest.mark.parametrize(
    ("old", "new", "missing"),
    (
        (
            "    def register_trace_context_factory(self, factory):\n"
            "        if factory not in self.trace_context_factories:\n"
            "            self.trace_context_factories.append(factory)\n",
            "    register_trace_context_factory = None\n",
            "cutlass.cutlass_dsl.CuTeDSL.register_trace_context_factory",
        ),
        (
            '    {"_option_name": "link-libraries"},',
            '    {"_option_name": "renamed-link-libraries"},',
            "cutlass.base_dsl.compiler.LinkLibraries._option_name=link-libraries",
        ),
        (
            'compiler.GPUArch = type("GPUArch", (), {})',
            "compiler.GPUArch = None",
            "cutlass.base_dsl.compiler.GPUArch",
        ),
        (
            "cute._get_launch_facts = lambda: {}",
            "cute._get_launch_facts = None",
            "cutlass.cute._get_launch_facts",
        ),
    ),
)
def test_cutlass_probe_uses_the_central_compiler_capability_contract(
    old: str,
    new: str,
    missing: str,
) -> None:
    incompatible = _COMPATIBLE_CUTLASS.replace(old, new)
    assert incompatible != _COMPATIBLE_CUTLASS
    _assert_source_passes(
        incompatible
        + textwrap.dedent(
            f"""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert len(caught) == 1
        assert {missing!r} in str(caught[0].message)
        assert "cuda.coop.cutlass" not in sys.modules
        """
        )
    )


def test_standard_warning_filter_can_suppress_auto_registration_warning() -> None:
    incompatible = _COMPATIBLE_CUTLASS.replace(
        "    def register_trace_finalize_hook(self, hook):\n"
        "        self.hook = hook\n\n",
        "",
    )
    _assert_source_passes(
        incompatible
        + textwrap.dedent(
            r"""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings(
                "ignore",
                message=r"cuda\.coop automatic DSL registration:",
                category=UserWarning,
            )
            from cuda import coop

        assert not caught
        assert coop.__name__ == "cuda.coop"
        """
        )
    )


def test_pythonwarnings_environment_suppresses_the_stable_warning_prefix() -> None:
    incompatible = _COMPATIBLE_CUTLASS.replace(
        "    def register_trace_finalize_hook(self, hook):\n"
        "        self.hook = hook\n\n",
        "",
    )
    script = (
        incompatible + "from cuda import coop\nassert coop.__name__ == 'cuda.coop'\n"
    )
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment["PYTHONPATH"] = os.fspath(SOURCE_ROOT)
    environment["PYTHONWARNINGS"] = (
        "ignore:cuda.coop automatic DSL registration:UserWarning"
    )

    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_auto_registration_does_not_swallow_base_exceptions() -> None:
    _assert_source_passes(
        """
        import importlib.abc
        import sys

        class InterruptCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "cutlass":
                    raise KeyboardInterrupt("injected interrupt")
                return None

        sys.meta_path.insert(0, InterruptCutlass())
        try:
            from cuda import coop
        except KeyboardInterrupt as error:
            assert str(error) == "injected interrupt"
        else:
            raise AssertionError("automatic registration swallowed KeyboardInterrupt")
        """
    )
