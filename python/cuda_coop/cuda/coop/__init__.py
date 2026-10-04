# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose cooperative operations through a common CUDA Python interface.

Compiler adapters translate these calls into device code. The root import
probes supported compiler runtimes already loaded by the application and can
activate their adapters. Use :func:`register` or a qualified backend import
when importing the common interface before the compiler runtime.
"""

import importlib.metadata
from pkgutil import extend_path

from ._registration import register

__path__ = extend_path(__path__, __name__)

from ._core import api as _common_api
from ._core._auto_registration import _auto_register_known_dsls

globals().update(
    {name: getattr(_common_api, name) for name in _common_api.__all__}
)


def _package_version() -> str:
    """Return the installed version or ``0+unknown`` without metadata."""

    try:
        return importlib.metadata.version("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()

__all__ = ["__version__", "register"]
__all__.extend(_common_api.__all__)


def __dir__() -> list[str]:
    """List public names for interactive completion."""

    return sorted(__all__)


_auto_register_known_dsls()
del _auto_register_known_dsls
