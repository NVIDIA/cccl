# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import textwrap

import pytest

from ._imports import (
    SOURCE_ROOT,
    optional_dependencies,
    run_python_with_source,
)


def test_numba_mlir_module_ships_in_the_source_tree():
    assert (SOURCE_ROOT / "cuda" / "coop" / "numba_mlir" / "__init__.py").is_file()


def test_numba_mlir_scope_packages_are_private_in_the_source_tree():
    root = SOURCE_ROOT / "cuda" / "coop" / "numba_mlir"

    assert (root / "_block" / "__init__.py").is_file()
    assert (root / "_warp" / "__init__.py").is_file()
    assert not (root / "block").exists()
    assert not (root / "warp").exists()


def test_root_import_is_cold_and_qualified_numba_import_reports_missing_runtime():
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        class BlockNumbaRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "numba_cuda_mlir" or fullname.startswith(
                    "numba_cuda_mlir."
                ):
                    raise ImportError("blocked Numba runtime", name=fullname)
                return None

        sys.meta_path.insert(0, BlockNumbaRuntime())

        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert "numba_cuda_mlir" not in sys.modules
        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "numba-cuda-mlir"
            assert exc.reason_code == "backend-runtime-missing"
            assert exc.details["missing"] == "numba_cuda_mlir"
            assert isinstance(exc.__cause__, ImportError)
            message = str(exc).lower()
            assert "numba" in message
            assert "install" in message
        else:
            raise AssertionError("qualified Numba import unexpectedly succeeded")
        """
    )
    result = run_python_with_source(
        script,
        inherit_pythonpath=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("setup", "reason_code", "details"),
    [
        pytest.param(
            """
class BrokenNumbaRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "numba_cuda_mlir":
            raise ImportError("broken transitive import", name="numba_dependency")
        return None

sys.meta_path.insert(0, BrokenNumbaRuntime())
""",
            "transitive-runtime-import-failed",
            {"missing": "numba_dependency"},
            id="transitive-import",
        ),
    ],
)
def test_qualified_numba_import_errors_are_structured(setup, reason_code, details):
    indented_setup = textwrap.indent(textwrap.dedent(setup).strip(), "        ")
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys
        import types

{indented_setup}

        try:
            import cuda.coop.numba_mlir  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "numba-cuda-mlir"
            assert exc.reason_code == {reason_code!r}
            for key, value in {details!r}.items():
                assert exc.details[key] == value
            assert isinstance(exc.__cause__, BaseException)
            assert "numba" in str(exc).lower()
        else:
            raise AssertionError("qualified Numba import unexpectedly succeeded")
        """
    )
    result = run_python_with_source(script, inherit_pythonpath=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "qualified_imports",
    (
        """import cuda.coop.cutlass
import cuda.coop.numba_mlir as numba_coop""",
        """import cuda.coop.numba_mlir as numba_coop
import cuda.coop.cutlass""",
    ),
)
def test_qualified_import_orders_activate_each_backend(qualified_imports: str):
    script = textwrap.dedent(
        """
        import sys
        import threading
        import types

        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
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
                self.trace_finalize_hook = hook

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

        runtime = types.ModuleType("numba_cuda_mlir")
        runtime.__path__ = []
        runtime.__version__ = "0.5.0"
        cuda = types.ModuleType("numba_cuda_mlir.cuda")
        extending = types.ModuleType("numba_cuda_mlir.extending")
        extending.WholeFunctionPlanner = type("WholeFunctionPlanner", (), {})
        extending.refresh_registries = lambda: None
        extending.register_planner = lambda planner: planner
        extending.require_launch_config = lambda state: {}
        extending.set_required_dynamic_shared_memory = lambda state, size: None
        rewrites = types.ModuleType("numba_cuda_mlir.numba_cuda.core.rewrites")
        rewrites.Rewrite = type("Rewrite", (), {})
        rewrites.register_rewrite = lambda phase: lambda rewrite: rewrite
        group_planner_type = type("CoopGroupHierarchyPlanner", (), {})
        whole_planner_type = type("CoopWholeFunctionPlanner", (), {})
        rewrite_type = type("CoopSinglePhaseRewrite", (), {})
        rewrites.rewrite_registry = types.SimpleNamespace(
            rewrites={"before-inference": [rewrite_type]},
        )
        planners = types.ModuleType("numba_cuda_mlir._whole_function_planners")
        planners._planner_registry = types.SimpleNamespace(
            _lock=threading.RLock(),
            _planners=[group_planner_type, whole_planner_type],
        )
        single_phase = types.ModuleType(
            "cuda.coop.numba_mlir._single_phase_rewrites"
        )
        single_phase.CoopWholeFunctionPlanner = whole_planner_type
        single_phase.CoopSinglePhaseRewrite = rewrite_type
        group_rewrites = types.ModuleType(
            "cuda.coop.numba_mlir._group_rewrites"
        )
        group_rewrites.CoopGroupHierarchyPlanner = group_planner_type
        runtime.cuda = cuda
        sys.modules["numba_cuda_mlir"] = runtime
        sys.modules["numba_cuda_mlir.cuda"] = cuda
        sys.modules["numba_cuda_mlir.extending"] = extending
        sys.modules["numba_cuda_mlir.numba_cuda.core.rewrites"] = rewrites
        sys.modules["numba_cuda_mlir._whole_function_planners"] = planners
        sys.modules["cuda.coop.numba_mlir._single_phase_rewrites"] = single_phase
        sys.modules["cuda.coop.numba_mlir._group_rewrites"] = group_rewrites

        __QUALIFIED_IMPORTS__
        from cuda import coop
        from cuda.coop._core import root_api

        assert coop.__name__ == "cuda.coop"
        assert numba_coop.__name__ == "cuda.coop.numba_mlir"
        assert root_api._backend_module_name() == "cuda.coop.cutlass"
        """
    ).replace("__QUALIFIED_IMPORTS__", qualified_imports)
    result = run_python_with_source(script, inherit_pythonpath=False)

    assert result.returncode == 0, result.stderr


def test_numba_mlir_extras_select_compatible_runtime_versions():
    extras = optional_dependencies()

    assert "pytest-xdist" in extras["test"]
    assert "pyright==1.1.411" in extras["test"]
    assert "simt-cu12" not in extras
    assert "simt-cu13" not in extras
    assert extras["numba-cuda-mlir"] == [
        "cuda-coop[cu13]",
        "numba-cuda-mlir[cu13]>=0.5.0",
    ]
    assert extras["numba-cuda-mlir-cu12"] == [
        "cuda-coop[cu12]",
        "numba-cuda-mlir[cu12]>=0.5.0",
    ]
    assert extras["numba-cuda-mlir-cu13"] == [
        "cuda-coop[cu13]",
        "numba-cuda-mlir[cu13]>=0.5.0",
    ]
    assert {
        "sysctk12",
        "sysctk13",
        "test-cu12",
        "test-cu13",
        "test-sysctk12",
        "test-sysctk13",
    }.isdisjoint(extras)
