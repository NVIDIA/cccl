# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._algorithm import AlgorithmSpec
from ._types import (
    TempStorageParameter,
)


@runtime_checkable
class CoreBackendAdapter(Protocol):
    """Operations a backend supplies to materialize a core algorithm spec.

    The protocol deliberately stops before tracing, linking, caching, launch
    integration, and compiler hook registration. Those remain backend concerns.
    """

    def normalize_dtype(self, dtype: Any) -> Any:
        """Return the backend's canonical representation of ``dtype``."""
        ...

    def cpp_type(self, dtype: Any) -> str:
        """Render a canonical backend dtype as a C++ type spelling."""
        ...

    def lower_parameter(
        self,
        parameter: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        """Lower a value, pointer, reference, or array parameter."""
        ...

    def lower_temp_storage(
        self,
        parameter: TempStorageParameter,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        """Lower a backend-managed or explicitly supplied storage parameter."""
        ...

    def materialize(self, specialization: AlgorithmSpec, **kwargs: Any) -> Any:
        """Materialize ``specialization`` through the backend's call path."""
        ...


def lower_method_parameters(
    adapter: CoreBackendAdapter,
    specialization: AlgorithmSpec,
    method: tuple[Any, ...],
    *,
    include_temp_storage: bool,
) -> tuple[Any, ...]:
    """Dispatch one core method signature through an adapter's lowering
    hooks.
    """

    lowered = []
    for parameter in method:
        if isinstance(parameter, TempStorageParameter):
            if include_temp_storage:
                lowered.append(
                    adapter.lower_temp_storage(
                        parameter,
                        specialization=specialization,
                    )
                )
        else:
            lowered.append(
                adapter.lower_parameter(
                    parameter,
                    specialization=specialization,
                )
            )
    return tuple(lowered)
