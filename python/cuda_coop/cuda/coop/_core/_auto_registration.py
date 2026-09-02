# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capability-based activation of an installed Numba-CUDA-MLIR backend."""

from __future__ import annotations

import importlib
import os
import warnings

_DISABLE_ENV = "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


class CudaCoopAutoRegistrationWarning(UserWarning):
    """An installed compiler runtime could not activate the common root."""


class _BackendUnavailable(ImportError):
    """The optional compiler runtime or integration is absent."""


def _auto_registration_disabled() -> bool:
    value = os.environ.get(_DISABLE_ENV)
    return value is not None and value.strip().lower() not in _FALSE_ENV_VALUES


def _import_numba_mlir_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if getattr(error, "name", None) in {
            "numba_cuda_mlir",
            "cuda.coop.numba_mlir",
        }:
            raise _BackendUnavailable(module_name) from error
        raise


def _activate_numba_mlir() -> None:
    _import_numba_mlir_module("numba_cuda_mlir")
    _import_numba_mlir_module("cuda.coop.numba_mlir")


def _auto_register_numba_mlir() -> bool:
    """Activate a compatible Numba-CUDA-MLIR backend when installed."""

    if _auto_registration_disabled():
        return False
    try:
        _activate_numba_mlir()
    except _BackendUnavailable as error:
        from .root_api import _record_backend_activation_failure

        _record_backend_activation_failure(
            "numba_mlir",
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
            "numba_mlir",
            reason_code,
            error,
        )
        warnings.warn(
            "cuda.coop automatic Numba-CUDA-MLIR activation was skipped: "
            f"{type(error).__name__}: {error}",
            CudaCoopAutoRegistrationWarning,
            stacklevel=2,
        )
        return False
    return True


__all__: list[str] = []
