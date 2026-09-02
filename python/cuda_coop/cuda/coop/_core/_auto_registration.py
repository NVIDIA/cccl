# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Automatically connect ``cuda.coop`` to compatible Python kernel DSLs.

The portable ``from cuda import coop`` API does not select a compiler backend.
At import time, this module probes the installed CUTLASS DSL, verifies the
capabilities needed by ``cuda.coop``, and registers it as a conditional
candidate. The common root selects CUTLASS only while its exact compiler
environment manager is current.

Set ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` to a truthy value to skip these
probes. Activate a backend explicitly by importing its ``cuda.coop.<dsl>``
subpackage directly.
"""

from __future__ import annotations

import importlib
import os
import warnings

# A truthy value disables all automatic DSL probes and registrations.
_DISABLE_ENV = "CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


class CudaCoopAutoRegistrationWarning(UserWarning):
    """An installed DSL runtime could not activate the portable API."""


class _BackendUnavailable(ImportError):
    """An optional backend runtime is absent."""


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
