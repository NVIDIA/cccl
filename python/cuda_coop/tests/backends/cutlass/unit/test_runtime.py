# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional-runtime diagnostics and compiler-environment activation."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from cuda.coop._core.api import _dispatch
from tests.support.paths import PACKAGE_ROOT

pytestmark = pytest.mark.unit


@pytest.fixture
def compiler_modules(monkeypatch):
    # Load the runtime validator without importing the qualified facade: its
    # missing-runtime diagnostics must also be testable without CUTLASS.
    modules = []
    for name in ("_runtime", "_activation"):
        module_name = f"{__package__}.{name}"
        path = PACKAGE_ROOT / "cuda/coop/cutlass/_compiler" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    monkeypatch.setattr(_dispatch, "_COMPILER_CONTEXT_PROBES", {})
    monkeypatch.setattr(modules[0], "_runtime_requirement", lambda: "Test runtime.")
    return modules


def _compatible_modules():
    environment = object()

    class CuTeDSL:
        @staticmethod
        def _get_dsl():
            return SimpleNamespace(envar=environment)

        def register_trace_finalize_hook(self, hook):
            pass

        def trace_finalize_hooks(self, hooks):
            pass

    class LinkLibraries:
        _option_name = "link-libraries"

    return {
        "cutlass.cutlass_dsl": SimpleNamespace(CuTeDSL=CuTeDSL),
        "cutlass.cute": SimpleNamespace(_get_launch_facts=lambda: None),
        "cutlass.base_dsl.compiler": SimpleNamespace(
            LinkLibraries=LinkLibraries, GPUArch=lambda value: value
        ),
        "cutlass.base_dsl.common": SimpleNamespace(
            get_current_env_manager=lambda: None
        ),
    }


def test_runtime_accepts_current_capabilities_without_trace_factory(
    monkeypatch, compiler_modules
):
    runtime, _ = compiler_modules
    modules = _compatible_modules()
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    validated = runtime.validate_cutlass_runtime()
    assert validated.dsl_type is modules["cutlass.cutlass_dsl"].CuTeDSL
    assert validated.common is modules["cutlass.base_dsl.common"]
    assert not hasattr(validated.dsl_type, "register_trace_context_factory")
    assert runtime.validate_cutlass_runtime() is validated


@pytest.mark.parametrize(
    "module,capability",
    [
        ("cutlass.cutlass_dsl", "trace_finalize_hooks"),
        ("cutlass.cutlass_dsl", "register_trace_finalize_hook"),
        ("cutlass.base_dsl.common", "get_current_env_manager"),
        ("cutlass.cute", "_get_launch_facts"),
        ("cutlass.base_dsl.compiler", "GPUArch"),
        ("cutlass.base_dsl.compiler", "LinkLibraries"),
    ],
)
def test_runtime_rejects_missing_capability(
    monkeypatch, compiler_modules, module, capability
):
    runtime, _ = compiler_modules
    modules = _compatible_modules()
    owner = modules[module]
    if module == "cutlass.cutlass_dsl":
        owner = owner.CuTeDSL
    delattr(owner, capability)
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    with pytest.raises(runtime.CutlassRuntimeDependencyError) as caught:
        runtime.validate_cutlass_runtime()
    assert caught.value.reason_code == "backend-runtime-incompatible"
    assert capability in str(caught.value)


def test_runtime_rejects_unrecognized_link_option(monkeypatch, compiler_modules):
    runtime, _ = compiler_modules
    modules = _compatible_modules()
    modules["cutlass.base_dsl.compiler"].LinkLibraries._option_name = "libraries"
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    with pytest.raises(runtime.CutlassRuntimeDependencyError, match="link-libraries"):
        runtime.validate_cutlass_runtime()


@pytest.mark.parametrize(
    "missing,reason",
    [
        ("cutlass", "backend-runtime-missing"),
        ("cutlass.cute", "conflicting-backend-runtime"),
        ("cuda.bindings.driver", "transitive-runtime-import-failed"),
    ],
)
def test_import_failure_preserves_cause_and_can_retry(
    monkeypatch, compiler_modules, missing, reason
):
    runtime, _ = compiler_modules
    cause = ModuleNotFoundError("missing dependency", name=missing)

    def fail(name):
        raise cause

    monkeypatch.setattr(runtime.importlib, "import_module", fail)
    with pytest.raises(runtime.CutlassRuntimeDependencyError) as caught:
        runtime.validate_cutlass_runtime()
    assert caught.value.reason_code == reason
    assert caught.value.__cause__ is cause
    assert caught.value.details["missing"] == missing
    modules = _compatible_modules()
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    assert (
        runtime.validate_cutlass_runtime().dsl_type
        is modules["cutlass.cutlass_dsl"].CuTeDSL
    )


def test_activation_tracks_environment_and_respects_explicit_scope(
    monkeypatch, compiler_modules
):
    runtime, activation = compiler_modules
    modules = _compatible_modules()
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    active = [None]
    modules["cutlass.base_dsl.common"].get_current_env_manager = lambda: active[0]
    activation.register_trace_context()
    assert _dispatch._backend_module_name() is None
    active[0] = object()
    assert _dispatch._backend_module_name() is None
    active[0] = modules["cutlass.cutlass_dsl"].CuTeDSL._get_dsl().envar
    assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
    with _dispatch._compiler_scope("cuda.coop.numba_mlir"):
        assert _dispatch._backend_module_name() == "cuda.coop.numba_mlir"
    assert _dispatch._backend_module_name() == "cuda.coop.cutlass"
    active[0] = None
    assert _dispatch._backend_module_name() is None


def test_failed_initialization_does_not_register_a_probe(monkeypatch, compiler_modules):
    runtime, activation = compiler_modules
    modules = _compatible_modules()
    modules["cutlass.cutlass_dsl"].CuTeDSL._get_dsl = lambda: SimpleNamespace()
    monkeypatch.setattr(runtime.importlib, "import_module", modules.__getitem__)
    with pytest.raises(runtime.CutlassRuntimeDependencyError, match="environment"):
        activation.register_trace_context()
    assert _dispatch._COMPILER_CONTEXT_PROBES == {}
