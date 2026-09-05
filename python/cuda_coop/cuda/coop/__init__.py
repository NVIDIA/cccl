# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative algorithm backends for CUDA Python DSLs."""

import importlib
import importlib.metadata
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_PORTABLE_API_MODULE = f"{__name__}._core.api"
_portable_api = importlib.import_module(_PORTABLE_API_MODULE)
_portable_exports: tuple[str, ...] = _portable_api.__all__
globals().update({name: getattr(_portable_api, name) for name in _portable_exports})


def _package_version() -> str:
    try:
        return importlib.metadata.version("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _package_version()

__all__ = ["__version__", *_portable_exports]


def __dir__() -> list[str]:
    """Return only the documented backend-neutral completion surface."""

    return sorted(__all__)


if _portable_exports:
    from ._core._auto_registration import _auto_register_known_dsls

    _auto_register_known_dsls()
    del _auto_register_known_dsls
