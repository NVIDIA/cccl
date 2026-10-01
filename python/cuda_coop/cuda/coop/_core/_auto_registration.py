# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register compatible compiler runtimes already imported by the application.

The ``cuda.coop`` root import calls this module's allowlisted registration
probe. A runtime must already be in ``sys.modules`` before its backend is
considered; merely installing an optional compiler does not cause the root
import to load it. This makes compiler-first imports convenient while keeping
the common API available without a compiler. Root-first applications can call
``cuda.coop.register("numba-cuda-mlir")`` or import the qualified backend.

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` controls this probe. Unset, empty,
``"0"``, ``"false"``, ``"no"``, and ``"off"`` leave automatic registration
enabled; surrounding whitespace and letter case are ignored. Any other value,
including ``"1"``, disables it. The value is read on each probe, normally
during the first ``cuda.coop`` import. Changing it does not unregister an
active backend or trigger another probe. Explicit registration and qualified
backend imports remain available when automatic registration is disabled.

An absent optional runtime is skipped silently. A detected runtime that fails
activation produces ``CudaCoopAutoRegistrationWarning`` and leaves the common
API import usable under normal warning handling. The probe removes newly
imported modules for the failed backend; its activation code owns registry
rollback. Warning filters may promote the warning to an exception.
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
_AUTO_DSL_CANDIDATES = ("numba_mlir",)


class CudaCoopAutoRegistrationWarning(UserWarning):
    """A detected optional DSL could not activate the common API."""


class _BackendUnavailable(ImportError):
    """The optional backend's top-level runtime is genuinely absent."""


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
    """Import a backend dependency while preserving evidence of broken installs.

    Automatic registration may silently skip an absent optional runtime, but
    a dependency failure inside an installed runtime should be reported. Use
    the missing module recorded on ``ImportError`` to distinguish these cases;
    catching every import failure as absence would hide incompatible installs.

    Parameters
    ----------
    module_name : str
        Fully qualified module to import.
    top_level : str
        Runtime module name whose absence is an expected optional dependency.
        Only an exact match with the exception's ``name`` is treated as absent.

    Returns
    -------
    ModuleType
        Imported module.

    Raises
    ------
    _BackendUnavailable
        The import reports that ``top_level`` itself is missing.
    ImportError
        Any other import failure, propagated unchanged for diagnostics.
    """

    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if getattr(error, "name", None) == top_level:
            raise _BackendUnavailable(top_level) from error
        raise


def _activate_numba_mlir() -> ModuleType:
    """Load the runtime, then let the qualified backend validate and
    activate.
    """

    _import_optional("numba_cuda_mlir", top_level="numba_cuda_mlir")
    return importlib.import_module("cuda.coop.numba_mlir")


_CANDIDATES = {
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
        "continued and other DSL backends were unaffected. "
        "Install a compatible "
        f"{candidate.install_hint}. "
        f"Set {_DISABLE_ENV}=1 to disable automatic DSL probing.",
        CudaCoopAutoRegistrationWarning,
        stacklevel=2,
    )


def _auto_register_known_dsls() -> tuple[str, ...]:
    """Activate allowlisted runtimes already imported by the application.

    The root package import must remain usable without loading an optional
    compiler or CUDA bindings. Inspect ``sys.modules`` first: installing a
    runtime is insufficient to activate it. Compiler-first imports get this
    automatic activation; root-first callers can use
    ``cuda.coop.register("numba-cuda-mlir")`` explicitly.

    Respect ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` and reuse qualified
    backends already in ``sys.modules``. For a new attempt, snapshot loaded
    modules so failure cleanup removes only newly imported backend modules.
    An absent optional runtime is skipped silently; other activation
    exceptions produce ``CudaCoopAutoRegistrationWarning`` and allow probing
    to continue. Registry rollback is the qualified backend's responsibility.

    Returns
    -------
    tuple of str
        Internal backend names successfully activated or already loaded, in
        candidate order. Empty when probing is disabled or no candidate
        qualifies. Does not include missing or unsuccessfully activated DSLs.
    """

    if _auto_registration_disabled():
        return ()

    registered = []
    for name in _AUTO_DSL_CANDIDATES:
        candidate = _CANDIDATES[name]
        if candidate.runtime_module not in sys.modules:
            continue
        package_prefix = f"cuda.coop.{name}"
        if package_prefix in sys.modules:
            registered.append(name)
            continue
        before = frozenset(sys.modules)
        try:
            candidate.activate()
        except _BackendUnavailable:
            _remove_failed_backend_modules(package_prefix, before)
        except Exception as error:  # noqa: BLE001 - optional activation must not break the root import.
            _remove_failed_backend_modules(package_prefix, before)
            _warn_incompatible(candidate, error)
        else:
            registered.append(name)
    return tuple(registered)


__all__: list[str] = []
