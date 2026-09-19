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
        assert callable(coop.register)
        assert "cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in sys.modules
        """
    )


@pytest.mark.parametrize(
    "activate", ("import cuda.coop.cutlass", 'coop.register("cutlass")')
)
def test_missing_cutlass_preserves_root_and_has_actionable_qualified_error(activate):
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
            ACTIVATE_CUTLASS
        except ImportError as error:
            assert error.reason_code == "backend-runtime-missing", error
            assert "compatible CUTLASS DSL runtime" in str(error)
        else:
            raise AssertionError("missing runtime was accepted")
        assert callable(coop.load)
        assert "cuda.coop.cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in _dispatch._COMPILER_CONTEXT_PROBES
        """.replace("ACTIVATE_CUTLASS", activate)
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
@pytest.mark.parametrize("registration", (False, True), ids=("qualified", "register"))
@pytest.mark.parametrize("disable_auto", (False, True))
def test_root_first_qualified_import_activates_backend(registration, disable_auto):
    _run_import_probe(
        f"""
        import os
        import sys
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1" if {disable_auto!r} else "0"
        from cuda import coop

        assert "cutlass" not in sys.modules
        if {registration!r}:
            assert coop.register("cutlass") is None
            cutlass_coop = sys.modules["cuda.coop.cutlass"]
        else:
            import cuda.coop.cutlass as cutlass_coop
        from cutlass.base_dsl.common import active_env_manager
        from cutlass.cutlass_dsl import CuTeDSL
        from cuda.coop._core.api import _dispatch

        assert _dispatch._backend_module_name() is None
        with active_env_manager(CuTeDSL._get_dsl().envar):
            assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
        assert _dispatch._backend_module_name() is None
        assert callable(cutlass_coop.load)
        probe = _dispatch._COMPILER_CONTEXT_PROBES["cuda.coop.cutlass"]
        assert coop.register("cutlass") is None
        assert coop.register("cutlass") is None
        assert _dispatch._COMPILER_CONTEXT_PROBES["cuda.coop.cutlass"] is probe
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
@pytest.mark.parametrize("registration", (False, True), ids=("automatic", "register"))
def test_compiler_backends_coexist(first, registration):
    second = "numba_cuda_mlir" if first == "cutlass" else "cutlass"
    _run_import_probe(
        f"""
        import importlib
        import os
        import sys
        if {registration!r}:
            os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
            from cuda import coop
        importlib.import_module({first!r})
        importlib.import_module({second!r})
        from cuda import coop
        if {registration!r}:
            assert coop.register({first!r}) is None
            assert coop.register({second!r}) is None
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


@pytest.mark.skipif(not _CUTLASS_AVAILABLE, reason="requires CUTLASS DSL")
@pytest.mark.parametrize("failure", ("missing", "incompatible"))
def test_register_retry(failure):
    _run_import_probe(
        f"""
        import importlib.abc
        import os
        import sys
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        from cuda import coop
        from cuda.coop._core.api import _dispatch

        class MissingCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass":
                    raise ModuleNotFoundError("CUTLASS unavailable", name="cutlass")
                return None

        if {failure!r} == "missing":
            finder = MissingCutlass()
            sys.meta_path.insert(0, finder)
        else:
            import cutlass.cute as cute
            launch_facts = cute._get_launch_facts
            del cute._get_launch_facts
        try:
            coop.register("cutlass")
        except ImportError as error:
            expected = "backend-runtime-missing" if {failure!r} == "missing" else "backend-runtime-incompatible"
            assert error.reason_code == expected, error
        else:
            raise AssertionError("invalid CUTLASS runtime was accepted")
        assert "cuda.coop.cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in _dispatch._COMPILER_CONTEXT_PROBES
        assert callable(coop.load)
        if {failure!r} == "missing":
            sys.meta_path.remove(finder)
        else:
            cute._get_launch_facts = launch_facts
        assert coop.register("cutlass") is None
        assert "cuda.coop.cutlass" in _dispatch._COMPILER_CONTEXT_PROBES
        assert _dispatch._backend_module_name() is None
        """
    )


@pytest.mark.skipif(not _CUTLASS_AVAILABLE, reason="requires CUTLASS DSL")
@pytest.mark.parametrize("operation", ("topk_min_keys",))
def test_unimplemented_family(operation):
    from cuda import coop
    from cuda.coop._core.api._dispatch import (
        UnsupportedCoopBackendOperationError,
        _compiler_scope,
    )

    coop.register("cutlass")
    with _compiler_scope("cuda.coop.cutlass"):
        group = coop.this_block()
        values = coop.ThreadData(2, dtype=int)
        values[0], values[1] = 1, 2
        with pytest.raises(UnsupportedCoopBackendOperationError) as caught:
            getattr(coop, operation)(group, values, k=1)
    assert caught.value.operation == operation
    assert caught.value.backend_module == "cuda.coop.cutlass"
    assert caught.value.reason_code == "cuda-coop-backend-operation-unavailable"
