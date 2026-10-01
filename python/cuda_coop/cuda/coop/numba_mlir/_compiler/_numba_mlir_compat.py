# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate the optional Numba backend's version before compiler integration.

Why package requirements and an activation check both exist
---------------------------------------------------------
The Numba extras in ``pyproject.toml`` declare the supported
``numba-cuda-mlir`` version range. When an application installs
``cuda-coop[numba-cuda-mlir-cu12]`` or ``cuda-coop[numba-cuda-mlir-cu13]``,
its package installer can select a compatible compiler or report a dependency
conflict before the application runs. The lower bound must name a released
compiler containing every API and behavior that this backend requires. Raise
that bound when adopting a newer compiler API, rather than adding a fallback
for older releases.

The base ``cuda-coop`` distribution deliberately has no compiler dependencies.
Applications may install it with ``pip install cuda-coop`` and manage their
compiler separately. Installing an extra applies additional requirements;
merely having Numba installed does not activate those requirements during a
base-package installation. An independently installed compiler can therefore
be outside our supported range, even though installing ``cuda-coop`` succeeds.
The activation check gives that application a useful error containing the
required range, detected version, and installation instructions.

Why validation belongs to backend activation
-------------------------------------------
An application using another backend, such as CUTLASS, should not need to
upgrade an unrelated Numba installation. We check only when activating
``cuda.coop.numba_mlir``, through its qualified import or
``coop.register("numba-cuda-mlir")``. Activation checks the version before
importing the CUDA compiler integration and registering compiler hooks. A failed
check therefore reports an environment problem before kernel compilation,
without adding checks to individual operations or the kernel launch path.

The common ``cuda.coop`` import does not inspect every installed compiler. Its
automatic registration considers runtimes the application already imported.
If that optional activation fails, the common import emits a warning and
continues; explicit activation raises the error. Merely installing an old
Numba version has no effect on an application using another backend.

Why there is no feature-by-feature compatibility layer
-----------------------------------------------------
The supported version range is the compiler contract. Checking that an
attribute exists cannot establish the semantics of launch specialization,
inlining, or generated code. Accepting older releases based on selected
attributes would create a second, less precise support policy and defer
failures to whichever operation first needs an unavailable feature. We use
the required compiler APIs directly after checking the version.

Likewise, catching AttributeError or TypeError around compiler operations and
turning every failure into an upgrade instruction can hide bugs in our own
integration or in a supported compiler. Import failures retain their original
cause, and compilation errors retain their specific meaning. In particular,
a missing launch-metadata API is different from a supported API reporting
that a particular compilation has no configured launch metadata. The latter
is a compilation/usage error, not evidence of an old compiler.

Maintenance
-----------
Keep the range below aligned with both Numba extras in ``pyproject.toml``.
Use packaging's version semantics, including local and development versions,
instead of a string prefix check. Already-installed prereleases within the
range are accepted; prereleases below the minimum or at the excluded next
series remain outside it. A missing or unparsable version cannot establish
support and is reported explicitly. Compiler API loading, planner registration,
and any registration failure handling belong to the activation code, rather
than to a registry of capabilities in this module.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

_REQUIRED_RUNTIME_VERSION = ">=0.5.0,<0.6"
_RUNTIME_INSTALL_HINT = (
    "python -m pip install --upgrade 'cuda-coop[numba-cuda-mlir-cu12]' "
    "for CUDA 12 or 'cuda-coop[numba-cuda-mlir-cu13]' for CUDA 13"
)


class _NumbaMlirBackendImportError(ImportError):
    """Qualified-backend import failure with its original cause preserved."""

    def __init__(self, reason_code, message, *, cause=None, **details):
        super().__init__(message)
        self.backend = "numba-cuda-mlir"
        self.reason_code = reason_code
        self.details = details
        if cause is not None:
            self.__cause__ = cause


def _detected_version(runtime: Any) -> str | None:
    version = getattr(runtime, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version("numba-cuda-mlir")
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_requirement(runtime: Any = None) -> str:
    """Describe the supported range and detected compiler installation."""

    version = _detected_version(runtime)
    detected = "the installed numba-cuda-mlir version could not be determined"
    if version is not None:
        detected = f"detected numba-cuda-mlir=={version}"
    return (
        f"cuda.coop.numba_mlir requires numba-cuda-mlir"
        f"{_REQUIRED_RUNTIME_VERSION}; {detected}. Install with "
        f"{_RUNTIME_INSTALL_HINT}."
    )


def _require_numba_mlir_version(runtime: Any) -> None:
    """Reject unsupported installations before accessing compiler APIs."""

    # Packaging is a backend dependency. Import it here so a missing runtime
    # still gets its own diagnostic, including in a base-only installation.
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ModuleNotFoundError as exc:
        if exc.name != "packaging":
            raise
        raise _NumbaMlirBackendImportError(
            "backend-dependency-missing",
            "cuda.coop.numba_mlir requires packaging to validate its compiler "
            f"version. Install with {_RUNTIME_INSTALL_HINT}.",
            cause=exc,
            missing=exc.name,
        ) from exc

    version = _detected_version(runtime)
    try:
        supported = version is not None and SpecifierSet(
            _REQUIRED_RUNTIME_VERSION
        ).contains(Version(version), prereleases=True)
    except InvalidVersion:
        supported = False
    if not supported:
        raise _NumbaMlirBackendImportError(
            "unsupported-runtime-version",
            _runtime_requirement(runtime),
            detected_version=version,
            required_version=_REQUIRED_RUNTIME_VERSION,
        )


__all__: tuple[str, ...] = ()
