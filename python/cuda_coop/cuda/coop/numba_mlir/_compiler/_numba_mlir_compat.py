# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the optional Numba backend's version before compiler integration."""

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
