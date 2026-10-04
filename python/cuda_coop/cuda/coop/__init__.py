# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose cooperative operations through a common CUDA Python interface.

Compiler adapters translate these calls into device code. Importing the common
interface does not load a compiler; :func:`register` loads a selected adapter
when explicit registration is needed.
"""

import importlib
import importlib.metadata
from pkgutil import extend_path

from ._registration import register

__path__ = extend_path(__path__, __name__)

_PORTABLE_API_MODULE = f"{__name__}._core.api"
_portable_api = importlib.import_module(_PORTABLE_API_MODULE)
_portable_exports: tuple[str, ...] = _portable_api.__all__
globals().update(
    {name: getattr(_portable_api, name) for name in _portable_exports}
)


def _package_version() -> str:
    """Return the installed version or ``0+unknown`` without metadata."""

    try:
        return importlib.metadata.version("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()

__all__ = ["__version__", "register"]
__all__.extend(_portable_exports)


def __dir__() -> list[str]:
    """List public names for interactive completion."""

    return sorted(__all__)
