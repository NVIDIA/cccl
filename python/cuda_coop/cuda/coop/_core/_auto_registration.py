# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capability-based activation of an installed CUTLASS DSL backend."""

from __future__ import annotations

import importlib
import os
import warnings

_DISABLE_ENV = "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


class CudaCoopAutoRegistrationWarning(UserWarning):
    """An installed CUTLASS runtime could not activate the common root."""


class _BackendUnavailable(ImportError):
    """The optional CUTLASS runtime is absent."""


def _auto_registration_disabled() -> bool:
    value = os.environ.get(_DISABLE_ENV)
    return value is not None and value.strip().lower() not in _FALSE_ENV_VALUES


def _import_cutlass_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if getattr(error, "name", None) == "cutlass":
            raise _BackendUnavailable("cutlass") from error
        raise


def _activate_cutlass() -> None:
    _import_cutlass_module("cutlass.cutlass_dsl")
    importlib.import_module("cuda.coop.cutlass")


def _auto_register_cutlass() -> bool:
    """Activate a compatible CUTLASS backend without making it mandatory."""

    if _auto_registration_disabled():
        return False
    try:
        _activate_cutlass()
    except _BackendUnavailable as error:
        from .root_api import _record_backend_activation_failure

        _record_backend_activation_failure(
            "cutlass",
            "backend-runtime-missing",
            error.__cause__ or error,
        )
        return False
    except Exception as error:
        from .root_api import _record_backend_activation_failure

        reason_code = getattr(
            error,
            "reason_code",
            "backend-runtime-incompatible",
        )
        if not isinstance(reason_code, str) or not reason_code:
            reason_code = "backend-runtime-incompatible"
        _record_backend_activation_failure(
            "cutlass",
            reason_code,
            error,
        )
        warnings.warn(
            "cuda.coop automatic CUTLASS activation was skipped: "
            f"{type(error).__name__}: {error}",
            CudaCoopAutoRegistrationWarning,
            stacklevel=2,
        )
        return False
    return True


__all__: list[str] = []
