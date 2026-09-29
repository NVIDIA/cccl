# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Translate shared parameter descriptions into a backend's compiler objects.

Core factories describe C++ calls without depending on a Python GPU compiler.
An adapter supplies the conversion rules for that compiler. The shared helper
preserves parameter order and can omit temporary storage for backends that
supply it another way.
"""

from __future__ import annotations

from typing import Any, Protocol

from ._algorithm import Algorithm
from ._types import (
    CxxOperator,
    PythonOperator,
    StatefulOperator,
    TempStorageParameter,
)


class CoreBackendAdapter(Protocol):
    """Define how a backend turns an ``Algorithm`` into a callable object.

    Lowering means converting a shared descriptor into the backend's type or
    parameter representation. Materialization uses those representations to
    build the backend's callable algorithm. The backend also owns tracing,
    linking, caches, launch integration, and compiler hook registration.
    """

    def normalize_dtype(self, dtype: Any) -> Any:
        """Convert a core dtype into the representation this backend uses.

        Built-in core tokens let a factory request types such as signed
        32-bit integers without importing the backend's type system.
        """
        ...

    def cpp_type(self, dtype: Any) -> str:
        """Spell a dtype as C++ source for generated declarations or calls."""
        ...

    def lower_parameter(
        self,
        parameter: Any,
        *,
        specialization: Algorithm,
    ) -> Any:
        """Convert one non-storage descriptor into a backend parameter.

        ``specialization`` supplies the bound values needed to resolve a
        descriptor's ``Dependency`` entries. The returned object belongs to
        the backend's signature representation.
        """
        ...

    def lower_cxx_operator(
        self,
        operator: CxxOperator,
        *,
        specialization: Algorithm,
    ) -> Any:
        """Lower a statically described C++ operator."""
        ...

    def lower_python_operator(
        self,
        operator: PythonOperator,
        *,
        specialization: Algorithm,
    ) -> Any:
        """Lower a stateless Python callable through backend compilation."""
        ...

    def lower_stateful_operator(
        self,
        operator: StatefulOperator,
        *,
        specialization: Algorithm,
    ) -> Any:
        """Lower a Python callable whose captured state is runtime data."""
        ...

    def lower_temp_storage(
        self,
        parameter: TempStorageParameter,
        *,
        specialization: Algorithm,
    ) -> Any:
        """Describe a temporary-storage argument in the backend's signature.

        The separate hook lets the backend apply its storage conventions.
        ``specialization`` provides the primitive and its bound arguments.
        """
        ...

    def materialize(self, specialization: Algorithm, **kwargs: Any) -> Any:
        """Build a backend callable from the fully bound core description.

        ``kwargs`` carries options specific to that backend's construction
        path. The protocol does not prescribe the returned object's type.
        """
        ...


def lower_method_parameters(
    adapter: CoreBackendAdapter,
    specialization: Algorithm,
    method: tuple[Any, ...],
    *,
    include_temp_storage: bool,
) -> tuple[Any, ...]:
    """Convert one signature in order, optionally omitting temporary storage.

    Some backends supply storage outside the ordinary argument list. The
    ``include_temp_storage`` flag lets them skip its descriptor while keeping
    the remaining arguments in the factory's order.

    Parameters
    ----------
    adapter : CoreBackendAdapter
        Backend that converts each retained descriptor.
    specialization : Algorithm
        Bound primitive passed to each conversion hook.
    method : tuple
        Parameter descriptors for one method signature.
    include_temp_storage : bool
        If true, send storage descriptors to ``lower_temp_storage``. If
        false, omit them. Send all other descriptors to ``lower_parameter``.

    Returns
    -------
    tuple
        Backend parameter objects in retained call order.
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
        elif isinstance(parameter, CxxOperator):
            lowered.append(
                adapter.lower_cxx_operator(
                    parameter,
                    specialization=specialization,
                )
            )
        elif isinstance(parameter, PythonOperator):
            lowered.append(
                adapter.lower_python_operator(
                    parameter,
                    specialization=specialization,
                )
            )
        elif isinstance(parameter, StatefulOperator):
            lowered.append(
                adapter.lower_stateful_operator(
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
