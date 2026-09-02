# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cuda import coop
from cuda.coop._core import _auto_registration, root_api

_SOURCE_ROOT = Path(__file__).parents[3]
_STUB = _SOURCE_ROOT / "cuda" / "coop" / "__init__.pyi"
_FIRST_SENTENCES = {
    "ThreadGroup": "Descriptor for the current CUDA thread block.",
    "this_block": "Return a descriptor for the current CUDA thread block.",
    "reduce": "Reduce one scalar per block thread and return the root result.",
    "sum": "Sum one scalar per block thread and return the root result.",
}


def _first_sentence(docstring: str) -> str:
    return inspect.cleandoc(docstring).splitlines()[0]


def _stub_docstrings() -> dict[str, str]:
    tree = ast.parse(_STUB.read_text(encoding="utf-8"))
    result = {}
    for statement in tree.body:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
            docstring = ast.get_docstring(statement, clean=False)
            if docstring is not None:
                result[statement.name] = docstring
    return result


def test_root_exports_exact_initial_surface() -> None:
    assert coop.__all__ == [
        "__version__",
        "ThreadGroup",
        "this_block",
        "reduce",
        "sum",
    ]
    assert root_api.__all__ == ["ThreadGroup", "this_block", "reduce", "sum"]


def test_root_signatures_match_scalar_reduction_contract() -> None:
    assert tuple(inspect.signature(root_api.this_block).parameters) == ()
    assert tuple(inspect.signature(root_api.reduce).parameters) == (
        "group",
        "value",
        "binary_op",
        "valid_items",
        "algorithm",
    )
    assert tuple(inspect.signature(root_api.sum).parameters) == (
        "group",
        "value",
        "valid_items",
        "algorithm",
    )


def test_runtime_and_stub_docstrings_share_locked_summaries() -> None:
    stub_docstrings = _stub_docstrings()
    for name, expected in _FIRST_SENTENCES.items():
        runtime_docstring = inspect.getdoc(getattr(coop, name))
        stub_docstring = stub_docstrings[name]
        assert runtime_docstring is not None
        assert _first_sentence(runtime_docstring) == expected
        assert _first_sentence(stub_docstring) == expected
        for section in ("Raises:", "Example:"):
            assert section in runtime_docstring
            assert section in stub_docstring


def test_root_callables_have_explicit_backend_markers() -> None:
    for name in ("this_block", "reduce", "sum"):
        assert getattr(coop, name).__cuda_coop_backend_member__ == name


def test_root_import_succeeds_when_numba_cuda_mlir_is_absent() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockNumbaCudaMlir(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "numba_cuda_mlir" or fullname.startswith(
                    "numba_cuda_mlir."
                ):
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}",
                        name="numba_cuda_mlir",
                    )
                return None

        sys.meta_path.insert(0, BlockNumbaCudaMlir())
        from cuda import coop

        assert coop.this_block().kind == "block"
        assert not any(
            name == "numba_cuda_mlir" or name.startswith("numba_cuda_mlir.")
            for name in sys.modules
        )
        """
    )
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment["PYTHONPATH"] = str(_SOURCE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_missing_backend_reports_structured_context_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match=(
            r"cuda\.coop\.sum requires an active compiler backend; "
            r"install a compatible backend or import cuda\.coop\.numba_mlir"
        ),
    ) as raised:
        root_api.sum(root_api.this_block(), 1)

    error = raised.value
    assert error.reason_code == "compiler-context-required"
    assert error.backend is None
    assert error.__cause__ is None


def test_compiler_context_error_retains_incompatible_backend_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError("missing compiler pass registration")
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    root_api._record_backend_activation_failure(
        "numba_mlir",
        "backend-runtime-incompatible",
        cause,
    )

    with pytest.raises(root_api.CoopCompilerContextRequiredError) as raised:
        root_api.sum(root_api.this_block(), 1)

    error = raised.value
    assert error.reason_code == "backend-runtime-incompatible"
    assert error.backend == "numba_mlir"
    assert error.feature == "sum"
    assert error.details == {
        "feature": "sum",
        "backend": "numba_mlir",
        "cause_type": "ImportError",
        "cause_message": "missing compiler pass registration",
        "activation_details": None,
    }
    assert error.cause is cause
    assert error.__cause__ is cause


def test_auto_registration_preserves_structured_runtime_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StructuredRuntimeError(ImportError):
        reason_code = "backend-runtime-too-old"

    cause = StructuredRuntimeError("detected numba-cuda-mlir 0.4")

    def fail_activation() -> None:
        raise cause

    monkeypatch.delenv("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", raising=False)
    monkeypatch.setattr(
        _auto_registration,
        "_activate_numba_mlir",
        fail_activation,
    )
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)

    with pytest.warns(_auto_registration.CudaCoopAutoRegistrationWarning):
        assert not _auto_registration._auto_register_numba_mlir()
    with pytest.raises(root_api.CoopCompilerContextRequiredError) as raised:
        root_api.sum(root_api.this_block(), 1)

    assert raised.value.reason_code == "backend-runtime-too-old"
    assert raised.value.__cause__ is cause


def test_auto_registration_honors_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_activation() -> None:
        raise AssertionError("backend activation must remain disabled")

    monkeypatch.setenv("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", "1")
    monkeypatch.setattr(
        _auto_registration,
        "_activate_numba_mlir",
        unexpected_activation,
    )

    assert not _auto_registration._auto_register_numba_mlir()


def test_thread_group_is_an_opaque_descriptor() -> None:
    with pytest.raises(TypeError, match="opaque"):
        coop.ThreadGroup()


def test_registered_backend_receives_canonical_root_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, tuple[Any, ...], dict[str, Any], str | None]] = []
    backend_name = "cuda.coop.testing_backend"
    backend = ModuleType(backend_name)

    def record(name: str):
        def implementation(*args: Any, **kwargs: Any) -> Any:
            observed.append(
                (name, args, kwargs, root_api._common_root_operation_name())
            )
            return args[1]

        return implementation

    backend.reduce = record("reduce")  # type: ignore[attr-defined]
    backend.sum = record("sum")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, backend_name, backend)
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)

    root_api._register_qualified_backend(backend_name)
    block = root_api.this_block()
    maximum = root_api.reduce(
        block,
        7,
        binary_op="maximum",
        valid_items=61,
        algorithm="raking-commutative-only",
    )
    total = root_api.sum(block, 9, algorithm="raking")

    assert maximum == 7
    assert total == 9
    assert observed == [
        (
            "reduce",
            (block, 7),
            {
                "binary_op": "max",
                "valid_items": 61,
                "algorithm": "raking_commutative_only",
            },
            "reduce",
        ),
        (
            "sum",
            (block, 9),
            {"valid_items": None, "algorithm": "raking"},
            "sum",
        ),
    ]


def test_compiler_scope_selects_backend_without_persisting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_name = "cuda.coop.scoped_backend"
    backend = ModuleType(backend_name)
    backend.sum = lambda group, value, **kwargs: value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, backend_name, backend)
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)

    with root_api._compiler_scope(backend_name):
        assert root_api.sum(root_api.this_block(), 3) == 3
    with pytest.raises(root_api.CoopCompilerContextRequiredError):
        root_api.sum(root_api.this_block(), 3)


def test_backend_registration_is_idempotent_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    root_api._register_qualified_backend("cuda.coop.numba_mlir")
    root_api._register_qualified_backend("cuda.coop.numba_mlir")
    with pytest.raises(RuntimeError, match="already activated"):
        root_api._register_qualified_backend("cuda.coop.other")


@pytest.mark.parametrize("valid_items", (True, 0, -1, 1.5, "one", object()))
def test_root_rejects_invalid_static_valid_items(valid_items) -> None:
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        root_api.sum(root_api.this_block(), 1, valid_items=valid_items)


def test_root_accepts_a_structural_compiler_integer() -> None:
    class CompilerInteger:
        width = 32
        signed = True
        dtype = "int32"

        @staticmethod
        def ir_value() -> object:
            return object()

    valid_items = CompilerInteger()

    assert root_api._normalize_valid_items("sum", valid_items) is valid_items


def test_root_rejects_callbacks_and_nondeterministic_algorithm() -> None:
    block = root_api.this_block()
    with pytest.raises(ValueError, match="block reduction operator"):
        root_api.reduce(block, 1, binary_op=lambda left, right: left + right)
    with pytest.raises(ValueError, match="BlockReduce algorithm"):
        root_api.sum(block, 1, algorithm="warp_reductions_nondeterministic")
