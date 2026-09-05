# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Connect the portable ``cuda.coop`` API to compatible Python kernel DSLs.

Importing ``cuda.coop`` probes an explicit allowlist of separately installed
DSL integrations. Each probe verifies the compiler capabilities that its
adapter needs before installing compiler-owned activation: a trace context for
CUTLASS or whole-function rewrites for Numba-CUDA-MLIR. One incompatible DSL
does not prevent another compatible DSL from registering. Set
``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` to a truthy value to skip every
automatic probe and register backends explicitly instead.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

_DISABLE_ENV = "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})
_WARNING_PREFIX = "cuda.coop automatic DSL registration:"

# Keep this an explicit allowlist so installing an unrelated compiler package
# cannot change the cuda.coop root import.
_AUTO_DSL_CANDIDATES = ("cutlass", "numba_mlir")


class CudaCoopAutoRegistrationWarning(UserWarning):
    """A detected optional DSL could not activate the portable API."""


class _BackendUnavailable(ImportError):
    """The optional backend's top-level runtime is genuinely absent."""


class _IncompatibleBackend(ImportError):
    """A detected backend is missing capabilities required by cuda.coop."""

    def __init__(self, message: str, *, missing_capabilities: tuple[str, ...]):
        super().__init__(message)
        self.missing_capabilities = missing_capabilities


@dataclass(frozen=True)
class _Candidate:
    display_name: str
    runtime_module: str
    distributions: tuple[str, ...]
    install_hint: str
    activate: Callable[[], ModuleType]


def _auto_registration_disabled(value: str | None = None) -> bool:
    """Return whether automatic probing is disabled by the environment."""

    if value is None:
        value = os.environ.get(_DISABLE_ENV)
    if value is None:
        return False
    return value.strip().lower() not in _FALSE_ENV_VALUES


def _import_optional(module_name: str, *, top_level: str) -> ModuleType:
    """Import one optional module and distinguish absence from breakage."""

    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if getattr(error, "name", None) == top_level:
            raise _BackendUnavailable(top_level) from error
        raise


def _require_callables(
    modules: dict[str, object],
    requirements: dict[str, tuple[str, ...]],
) -> None:
    missing = tuple(
        f"{module_name}.{name}"
        for module_name, names in requirements.items()
        for name in names
        if not callable(getattr(modules[module_name], name, None))
    )
    if missing:
        raise _IncompatibleBackend(
            "missing required callable features: " + ", ".join(missing),
            missing_capabilities=missing,
        )


def _activate_cutlass() -> ModuleType:
    """Let the qualified CUTLASS adapter validate and register its hooks."""

    _import_optional("cutlass.cutlass_dsl", top_level="cutlass")

    # cuda.coop.cutlass registers the common-root fallback as its final import
    # action, after validation, compiler-hook setup, and wrapper construction.
    return importlib.import_module("cuda.coop.cutlass")


def _activate_numba_mlir() -> ModuleType:
    """Preflight compiler APIs, then transactionally register coop rewrites."""

    _import_optional("numba_cuda_mlir", top_level="numba_cuda_mlir")
    importlib.import_module("numba_cuda_mlir.cuda")
    extending = importlib.import_module("numba_cuda_mlir.extending")
    rewrites = importlib.import_module("numba_cuda_mlir.numba_cuda.core.rewrites")
    _require_callables(
        {
            "numba_cuda_mlir.extending": extending,
            "numba_cuda_mlir.numba_cuda.core.rewrites": rewrites,
        },
        {
            "numba_cuda_mlir.extending": (
                "WholeFunctionPlanner",
                "refresh_registries",
                "register_planner",
                "require_launch_config",
                "set_required_dynamic_shared_memory",
            ),
            "numba_cuda_mlir.numba_cuda.core.rewrites": (
                "Rewrite",
                "register_rewrite",
            ),
        },
    )
    return importlib.import_module("cuda.coop.numba_mlir")


_CANDIDATES = {
    "cutlass": _Candidate(
        display_name="CUTLASS",
        runtime_module="cutlass",
        distributions=("nvidia-cutlass-dsl",),
        install_hint="cuda-coop[cutlass]",
        activate=_activate_cutlass,
    ),
    "numba_mlir": _Candidate(
        display_name="Numba-CUDA-MLIR",
        runtime_module="numba_cuda_mlir",
        distributions=("numba-cuda-mlir",),
        install_hint=(
            "cuda-coop[numba-cuda-mlir-cu12] for CUDA 12 or "
            "cuda-coop[numba-cuda-mlir-cu13] for CUDA 13"
        ),
        activate=_activate_numba_mlir,
    ),
}


def _detected_version(candidate: _Candidate) -> str | None:
    runtime = sys.modules.get(candidate.runtime_module)
    version = getattr(runtime, "__version__", None)
    if isinstance(version, str) and version:
        return version
    for distribution in candidate.distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _remove_failed_backend_modules(prefix: str, before: frozenset[str]) -> None:
    for module_name in tuple(sys.modules):
        if (
            module_name == prefix or module_name.startswith(f"{prefix}.")
        ) and module_name not in before:
            sys.modules.pop(module_name, None)


def _warn_incompatible(candidate: _Candidate, error: Exception) -> None:
    version = _detected_version(candidate)
    detected = f" (detected version {version})" if version is not None else ""
    reason = str(error).strip() or type(error).__name__
    missing = getattr(error, "name", None)
    if isinstance(missing, str) and missing and missing not in reason:
        reason = f"dependency {missing!r} failed to import: {reason}"
    warnings.warn(
        f"{_WARNING_PREFIX} {candidate.display_name}{detected} was detected "
        f"but was not enabled because {reason}. The cuda.coop root import "
        "continued and other DSL backends were unaffected. Install a compatible "
        f"{candidate.install_hint}. "
        f"Set {_DISABLE_ENV}=1 to disable automatic DSL probing.",
        CudaCoopAutoRegistrationWarning,
        stacklevel=2,
    )


def _auto_register_known_dsls() -> tuple[str, ...]:
    """Enable each compatible known DSL without coupling their failures."""

    if _auto_registration_disabled():
        return ()

    registered = []
    for name in _AUTO_DSL_CANDIDATES:
        candidate = _CANDIDATES[name]
        package_prefix = f"cuda.coop.{name}"
        if package_prefix in sys.modules:
            registered.append(name)
            continue
        before = frozenset(sys.modules)
        try:
            candidate.activate()
        except _BackendUnavailable:
            _remove_failed_backend_modules(package_prefix, before)
        except Exception as error:
            _remove_failed_backend_modules(package_prefix, before)
            _warn_incompatible(candidate, error)
        else:
            registered.append(name)
    return tuple(registered)


__all__: list[str] = []
