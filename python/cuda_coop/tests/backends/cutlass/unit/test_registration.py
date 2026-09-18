# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional CUTLASS activation in fresh interpreters and both import orders."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]
_CUTLASS_AVAILABLE = importlib.util.find_spec("cutlass") is not None
_NUMBA_AVAILABLE = importlib.util.find_spec("numba_cuda_mlir") is not None


def _run_import_probe(script: str) -> None:
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(script)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_import_does_not_load_cutlass_or_cuda_bindings():
    _run_import_probe(
        """
        import importlib.abc
        import sys

        class RejectCompilerImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise AssertionError("root imported CUTLASS")
                if fullname.startswith("cuda.bindings"):
                    raise AssertionError("root imported CUDA bindings")
                return None

        sys.meta_path.insert(0, RejectCompilerImport())
        from cuda import coop

        assert callable(coop.load)
        assert "cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in sys.modules
        """
    )


def test_missing_cutlass_preserves_root_and_has_actionable_qualified_error():
    _run_import_probe(
        """
        import importlib.abc
        import sys

        class MissingCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass":
                    raise ModuleNotFoundError("CUTLASS unavailable", name="cutlass")
                return None

        sys.meta_path.insert(0, MissingCutlass())
        from cuda import coop
        from cuda.coop._core.api import _dispatch

        try:
            import cuda.coop.cutlass
        except ImportError as error:
            assert error.reason_code == "backend-runtime-missing", error
            assert "compatible CUTLASS DSL runtime" in str(error)
        else:
            raise AssertionError("missing runtime was accepted")
        assert callable(coop.load)
        assert "cuda.coop.cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in _dispatch._COMPILER_CONTEXT_PROBES
        """
    )


@pytest.mark.skipif(not _CUTLASS_AVAILABLE, reason="requires CUTLASS DSL")
def test_cutlass_first_root_import_activates_backend():
    _run_import_probe(
        """
        import sys
        import cutlass
        from cuda import coop
        from cuda.coop._core.api import _dispatch

        assert "cuda.coop.cutlass" in sys.modules
        assert "cuda.coop.cutlass" in _dispatch._COMPILER_CONTEXT_PROBES
        assert _dispatch._backend_module_name() is None
        assert callable(coop.load)
        """
    )


@pytest.mark.skipif(not _CUTLASS_AVAILABLE, reason="requires CUTLASS DSL")
def test_root_first_qualified_import_activates_backend():
    _run_import_probe(
        """
        import sys
        from cuda import coop

        assert "cutlass" not in sys.modules
        import cuda.coop.cutlass as cutlass_coop
        from cutlass.base_dsl.common import active_env_manager
        from cutlass.cutlass_dsl import CuTeDSL
        from cuda.coop._core.api import _dispatch

        assert _dispatch._backend_module_name() is None
        with active_env_manager(CuTeDSL._get_dsl().envar):
            assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
        assert _dispatch._backend_module_name() is None
        assert callable(cutlass_coop.load)
        """
    )


@pytest.mark.skipif(not _CUTLASS_AVAILABLE, reason="requires CUTLASS DSL")
def test_broken_cutlass_activation_can_recover_without_reimporting_root():
    _run_import_probe(
        """
        import sys
        import warnings
        import cutlass.cute as cute

        get_launch_facts = cute._get_launch_facts
        del cute._get_launch_facts
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            from cuda import coop
        assert len(captured) == 1, captured
        assert "exact launch facts" in str(captured[0].message)
        assert "cuda.coop.cutlass" not in sys.modules
        from cuda.coop._core.api import _dispatch
        assert "cuda.coop.cutlass" not in _dispatch._COMPILER_CONTEXT_PROBES

        cute._get_launch_facts = get_launch_facts
        import cuda.coop.cutlass as cutlass_coop
        assert "cuda.coop.cutlass" in _dispatch._COMPILER_CONTEXT_PROBES
        assert callable(coop.load) and callable(cutlass_coop.load)
        """
    )


@pytest.mark.skipif(
    not (_CUTLASS_AVAILABLE and _NUMBA_AVAILABLE),
    reason="requires CUTLASS DSL and Numba-CUDA-MLIR",
)
@pytest.mark.parametrize("first", ("cutlass", "numba_cuda_mlir"))
def test_compiler_backends_coexist(first):
    second = "numba_cuda_mlir" if first == "cutlass" else "cutlass"
    _run_import_probe(
        f"""
        import importlib
        import sys
        importlib.import_module({first!r})
        importlib.import_module({second!r})
        from cuda import coop
        from cuda.coop._core.api import _dispatch
        from cutlass.base_dsl.common import active_env_manager
        from cutlass.cutlass_dsl import CuTeDSL

        assert "cuda.coop.cutlass" in sys.modules
        assert "cuda.coop.numba_mlir" in sys.modules
        assert _dispatch._backend_module_name() is None
        with active_env_manager(CuTeDSL._get_dsl().envar):
            assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
            with _dispatch._compiler_scope("cuda.coop.numba_mlir"):
                assert _dispatch._backend_module_name() == "cuda.coop.numba_mlir"
            assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
        assert _dispatch._backend_module_name() is None
        """
    )
