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
_CUTLASS_STUBS = (
    _SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_thread_data.pyi",
    _SOURCE_ROOT / "cuda" / "coop" / "cutlass" / "_load_store.pyi",
)
_FIRST_SENTENCES = {
    "ThreadData": "Create an uninitialized per-thread register payload.",
    "ThreadGroup": "Descriptor for the current CUDA thread block.",
    "this_block": "Return a descriptor for the current CUDA thread block.",
    "load": "Collectively load one block tile into a per-thread payload.",
    "store": "Collectively store one per-thread payload as one block tile.",
}


def _first_sentence(docstring: str) -> str:
    return inspect.cleandoc(docstring).splitlines()[0]


def _stub_docstrings(*paths: Path) -> dict[str, str]:
    result = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for statement in tree.body:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
                docstring = ast.get_docstring(statement, clean=False)
                if docstring is not None:
                    result[statement.name] = docstring
    return result


def test_root_exports_exact_initial_surface() -> None:
    assert coop.__all__ == [
        "__version__",
        "ThreadData",
        "ThreadGroup",
        "this_block",
        "load",
        "store",
    ]
    assert root_api.__all__ == [
        "ThreadData",
        "ThreadGroup",
        "this_block",
        "load",
        "store",
    ]


def test_root_signatures_use_one_items_vocabulary() -> None:
    assert tuple(inspect.signature(root_api.ThreadData).parameters) == (
        "items_per_thread",
        "dtype",
    )
    assert tuple(inspect.signature(root_api.this_block).parameters) == ()
    assert tuple(inspect.signature(root_api.load).parameters) == (
        "group",
        "source",
        "items",
        "valid_items",
        "oob_default",
        "offset",
    )
    assert tuple(inspect.signature(root_api.store).parameters) == (
        "group",
        "destination",
        "items",
        "valid_items",
        "offset",
    )


def test_runtime_and_stub_docstrings_share_locked_summaries() -> None:
    stub_docstrings = _stub_docstrings(_STUB)
    for name, expected in _FIRST_SENTENCES.items():
        runtime_docstring = inspect.getdoc(getattr(coop, name))
        stub_docstring = stub_docstrings[name]
        assert runtime_docstring is not None
        assert _first_sentence(runtime_docstring) == expected
        assert _first_sentence(stub_docstring) == expected
        assert runtime_docstring == inspect.cleandoc(stub_docstring)
        for section in ("Raises:", "Example:"):
            assert section in runtime_docstring
            assert section in stub_docstring


def test_qualified_stubs_share_portable_docstrings() -> None:
    root_docstrings = _stub_docstrings(_STUB)
    qualified_docstrings = _stub_docstrings(*_CUTLASS_STUBS)

    for name in ("ThreadData", "load", "store"):
        assert inspect.cleandoc(qualified_docstrings[name]) == inspect.cleandoc(
            root_docstrings[name]
        )


def test_root_import_succeeds_when_cutlass_is_absent() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ModuleNotFoundError(
                        f"No module named {fullname!r}",
                        name="cutlass",
                    )
                return None

        sys.meta_path.insert(0, BlockCutlass())
        from cuda import coop

        assert coop.this_block().kind == "block"
        assert not any(
            name == "cutlass" or name.startswith("cutlass.")
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
            r"cuda\.coop\.ThreadData requires an active compiler backend; "
            r"install or import a compatible backend before compiling a kernel"
        ),
    ) as raised:
        root_api.ThreadData(2)

    error = raised.value
    assert error.reason_code == "compiler-context-required"
    assert error.backend is None
    assert error.__cause__ is None


@pytest.mark.parametrize("items_per_thread", (0, -1, True, 1.5, "two"))
def test_thread_data_validates_item_count_before_backend_lookup(
    items_per_thread,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    with pytest.raises(ValueError, match="items_per_thread must be a positive"):
        root_api.ThreadData(items_per_thread)


def test_compiler_context_error_retains_incompatible_backend_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError("missing register_trace_finalize_hook")
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    root_api._record_backend_activation_failure(
        "cutlass",
        "backend-runtime-incompatible",
        cause,
    )

    with pytest.raises(root_api.CoopCompilerContextRequiredError) as raised:
        root_api.ThreadData(1)

    error = raised.value
    assert error.reason_code == "backend-runtime-incompatible"
    assert error.backend == "cutlass"
    assert error.feature == "ThreadData"
    assert error.details == {
        "feature": "ThreadData",
        "backend": "cutlass",
        "cause_type": "ImportError",
        "cause_message": "missing register_trace_finalize_hook",
        "activation_details": None,
    }
    assert error.cause is cause
    assert error.__cause__ is cause


@pytest.mark.parametrize(
    "reason_code",
    ("backend-runtime-missing", "backend-runtime-import-failed"),
)
def test_compiler_context_error_preserves_activation_reason(
    reason_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError(reason_code)
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    root_api._record_backend_activation_failure("cutlass", reason_code, cause)

    with pytest.raises(root_api.CoopCompilerContextRequiredError) as raised:
        root_api.ThreadData(1)

    assert raised.value.reason_code == reason_code
    assert raised.value.backend == "cutlass"
    assert raised.value.__cause__ is cause


def test_auto_registration_preserves_structured_runtime_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StructuredRuntimeError(ImportError):
        reason_code = "backend-runtime-import-failed"

    cause = StructuredRuntimeError("CUTLASS compiler import failed")

    def fail_activation() -> None:
        raise cause

    monkeypatch.delenv(
        "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION",
        raising=False,
    )
    monkeypatch.setattr(_auto_registration, "_activate_cutlass", fail_activation)
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)

    with pytest.warns(_auto_registration.CudaCoopAutoRegistrationWarning):
        assert not _auto_registration._auto_register_cutlass()
    with pytest.raises(root_api.CoopCompilerContextRequiredError) as raised:
        root_api.ThreadData(1)

    assert raised.value.reason_code == "backend-runtime-import-failed"
    assert raised.value.__cause__ is cause


def test_auto_registration_honors_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_activation() -> None:
        raise AssertionError("CUTLASS activation must remain disabled")

    monkeypatch.setenv("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", "1")
    monkeypatch.setattr(
        _auto_registration,
        "_activate_cutlass",
        unexpected_activation,
    )

    assert not _auto_registration._auto_register_cutlass()


def test_thread_group_is_an_opaque_descriptor() -> None:
    with pytest.raises(TypeError, match="opaque"):
        coop.ThreadGroup()


def test_registered_backend_receives_payload_load_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, tuple[Any, ...], dict[str, Any], str | None]] = []
    backend_name = "cuda.coop.testing_backend"
    backend = ModuleType(backend_name)

    def record(name: str, result: Any):
        def implementation(*args: Any, **kwargs: Any) -> Any:
            observed.append(
                (
                    name,
                    args,
                    kwargs,
                    root_api._common_root_operation_name(),
                )
            )
            return result

        return implementation

    payload = object()
    backend.ThreadData = record("ThreadData", payload)  # type: ignore[attr-defined]
    backend.load = record("load", payload)  # type: ignore[attr-defined]
    backend.store = record("store", None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, backend_name, backend)
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)

    root_api._register_qualified_backend(backend_name)
    block = root_api.this_block()
    items = root_api.ThreadData(2, dtype="int32")
    loaded = root_api.load(
        block,
        "source",
        items,
        valid_items=61,
        oob_default=-1,
        offset=3,
    )
    result = root_api.store(
        block,
        "destination",
        loaded,
        valid_items=61,
        offset=5,
    )

    assert items is payload
    assert loaded is payload
    assert result is None
    assert observed == [
        ("ThreadData", (2,), {"dtype": "int32"}, "ThreadData"),
        (
            "load",
            (block, "source", payload),
            {"valid_items": 61, "oob_default": -1, "offset": 3},
            "load",
        ),
        (
            "store",
            (block, "destination", payload),
            {"valid_items": 61, "offset": 5},
            "store",
        ),
    ]


def test_backend_registration_is_idempotent_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(root_api, "_QUALIFIED_BACKEND_MODULE", None)
    monkeypatch.setattr(root_api, "_BACKEND_ACTIVATION_FAILURE", None)
    root_api._register_qualified_backend("cuda.coop.cutlass")
    root_api._register_qualified_backend("cuda.coop.cutlass")
    with pytest.raises(RuntimeError, match="already activated"):
        root_api._register_qualified_backend("cuda.coop.other")


def test_load_rejects_oob_default_without_partial_tile() -> None:
    with pytest.raises(TypeError, match="oob_default requires valid_items"):
        root_api.load(root_api.this_block(), object(), object(), oob_default=-1)
