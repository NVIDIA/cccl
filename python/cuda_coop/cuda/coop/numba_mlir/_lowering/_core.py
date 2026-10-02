# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Translate compiler-neutral operation specifications into Numba descriptors.

The common core describes C++ algorithms, parameters, and template arguments
without depending on a Python compiler. This adapter supplies the Numba types
and calling conventions needed to turn those descriptions into an
``Algorithm``. For example, an array parameter needs a pointer-based wrapper,
and a runtime valid-item count can require checked integer narrowing.

The adapter also carries scratch and synchronization requirements into source
generation. It creates a specialized algorithm description; compilation and
creation of a callable happen later in ``make_invocable_from_specialization``
or after several descriptions have been collected for a shared compilation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import (
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    AlgorithmSpec,
    ArgumentBinding,
    Array,
    BuiltinDType,
    Constant,
    CoreBackendAdapter,
    CxxFunction,
    CxxOperator,
    Dependency,
    Pointer,
    PointerOffset,
    PythonOperator,
    Reference,
    StatefulOperator,
    SynchronizationScope,
    TempStorageParameter,
    Value,
    lower_method_parameters,
)

from .. import _types as backend
from .._compiler._operations import StorageABI
from .._semantic import _normalize_numba_callable


@dataclass(frozen=True)
class NumbaMlirArrayInputTransform:
    """Describe one elementwise input conversion in a generated CUB wrapper."""

    source_dtype: Any
    cpp_expression: str

    def __post_init__(self) -> None:
        if "{value}" not in self.cpp_expression:
            raise ValueError("array input transform must reference {value}")


def _optional_binding(value: object) -> ArgumentBinding:
    """Translate legacy presence markers without making their values static.

    Lowering factories accept both explicit binding descriptors and older
    arguments whose mere presence requested a runtime overload. Preserve an
    ``ArgumentBinding`` as supplied; otherwise ``None`` means omitted and every
    other object means runtime. In particular, a plain integer here is not a
    compile-time value. Callers must use ``ArgumentBinding.static`` to embed it.

    Parameters
    ----------
    value : object
        Binding descriptor, omitted sentinel, or legacy presence marker.

    Returns
    -------
    ArgumentBinding
        Existing descriptor or a new omitted/runtime binding.
    """

    if isinstance(value, ArgumentBinding):
        return value
    if value is None:
        return ArgumentBinding.omitted()
    return ArgumentBinding.runtime()


class NumbaMlirCoreAdapter(CoreBackendAdapter):
    """Build Numba calling-convention descriptors from common core parameters.

    Optional named overrides control scalar argument checks and elementwise
    array conversions. The resulting ``Algorithm`` uses the backend's normal
    source generation, compilation cache, and linking path.
    """

    _BUILTIN_DTYPES: ClassVar[dict[BuiltinDType, numba_types.Type]] = {
        INT8: numba_types.int8,
        UINT8: numba_types.uint8,
        INT16: numba_types.int16,
        UINT16: numba_types.uint16,
        INT32: numba_types.int32,
        UINT32: numba_types.uint32,
        INT64: numba_types.int64,
        UINT64: numba_types.uint64,
        FLOAT32: numba_types.float32,
        FLOAT64: numba_types.float64,
    }
    _CORE_DTYPES: ClassVar[dict[numba_types.Type, BuiltinDType]] = {
        value: key for key, value in _BUILTIN_DTYPES.items()
    }

    def __init__(
        self,
        *,
        input_transforms: Mapping[str, NumbaMlirArrayInputTransform]
        | None = None,
        value_abis: Mapping[str, backend.Value] | None = None,
    ) -> None:
        self._input_transforms = dict(input_transforms or {})
        self._value_abis = dict(value_abis or {})

    def normalize_dtype(self, dtype: Any) -> Any:
        if isinstance(dtype, BuiltinDType):
            try:
                return self._BUILTIN_DTYPES[dtype]
            except KeyError as exc:
                raise TypeError(
                    f"unsupported core dtype {dtype.name!r}"
                ) from exc
        return dtype

    def core_dtype(self, dtype: Any) -> Any:
        """Return a backend-neutral builtin token when one exists."""

        return self._CORE_DTYPES.get(dtype, dtype)

    def cpp_type(self, dtype: Any) -> str:
        return backend.numba_type_to_cpp(self.normalize_dtype(dtype))

    def _resolvable(self, value: Any) -> Any:
        if isinstance(value, Dependency):
            return backend.Dependency(value.name)
        if isinstance(value, Constant):
            return backend.Constant(value.value)
        return backend.Constant(self.normalize_dtype(value))

    @staticmethod
    def _is_backend_output(parameter: Any) -> bool:
        if parameter.is_return is None:
            return parameter.is_output
        return parameter.is_return

    def lower_parameter(
        self,
        parameter: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> backend.Parameter:
        """Translate one core parameter into the Numba provider ABI description.

        Preserve template dependencies for arrays, pointers, and references
        until ``Algorithm.specialize`` resolves them. Scalar values require a
        concrete dtype; a named scalar ABI override can impose stricter runtime
        typing or checked narrowing. Output ownership follows ``is_return`` when
        explicitly set, otherwise ``is_output`` determines the backend return
        value.

        Named input transforms are different from ordinary arrays: resolve their
        extent and target dtype now so source generation can emit a fixed local
        array and per-element C++ conversions. Only input-only arrays may use
        this path. Dependent C++ functors substitute bracketed type placeholders
        only, leaving unrelated bare tokens unchanged.

        Parameters
        ----------
        parameter : object
            Core pointer offset, array, pointer, reference, value, or C++
            functor descriptor. Temporary storage is handled by
            ``lower_temp_storage``.
        specialization : AlgorithmSpec
            Core specification providing template arguments for eager dependency
            resolution in transforms and C++ functors.

        Returns
        -------
        Parameter
            Backend descriptor; a scalar override may be the adapter's existing
            descriptor. ``materialize`` copies descriptors before attaching
            names.

        Raises
        ------
        TypeError
            The parameter kind is unsupported or a scalar value has a dependent
            dtype.
        ValueError
            An input transform targets an output/inout array or lacks a positive
            specialized integer extent.
        """

        if isinstance(parameter, PointerOffset):
            return backend.PointerOffset(
                self.normalize_dtype(parameter.dtype),
                parameter.pointer_arg_index,
                static_value=parameter.static_value,
            )
        if isinstance(parameter, Array):
            transform = (
                self._input_transforms.get(parameter.name)
                if parameter.name is not None
                else None
            )
            if transform is not None:
                if parameter.is_output or parameter.is_inout:
                    raise ValueError(
                        f"input transform {parameter.name!r}"
                        f" targets an output array"
                    )
                target_dtype = parameter.dtype
                if isinstance(target_dtype, Dependency):
                    target_dtype = target_dtype.resolve(
                        specialization.template_arguments
                    )
                elif isinstance(target_dtype, Constant):
                    target_dtype = target_dtype.value
                size = parameter.size
                if isinstance(size, Dependency):
                    size = size.resolve(specialization.template_arguments)
                elif isinstance(size, Constant):
                    size = size.value
                if (
                    not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 1
                ):
                    raise ValueError(
                        "Numba-CUDA-MLIR input transforms require a positive "
                        "specialized array extent"
                    )
                return backend.TransformedArray(
                    self.normalize_dtype(transform.source_dtype),
                    self.normalize_dtype(target_dtype),
                    size,
                    transform.cpp_expression,
                )
            if isinstance(
                parameter.dtype, (Dependency, Constant)
            ) or isinstance(parameter.size, (Dependency, Constant)):
                return backend.DependentArray(
                    self._resolvable(parameter.dtype),
                    self._resolvable(parameter.size),
                    is_output=self._is_backend_output(parameter),
                )
            return backend.Array(
                self.normalize_dtype(parameter.dtype),
                parameter.size,
                is_output=self._is_backend_output(parameter),
            )
        if isinstance(parameter, Pointer):
            dtype = parameter.dtype
            if isinstance(dtype, Dependency):
                pointer_type = (
                    backend.DependentPointerReference
                    if parameter.deref_on_call
                    else backend.DependentPointer
                )
                return pointer_type(
                    backend.Dependency(dtype.name),
                    is_output=self._is_backend_output(parameter),
                )
            pointer_type = (
                backend.PointerReference
                if parameter.deref_on_call
                else backend.Pointer
            )
            return pointer_type(
                self.normalize_dtype(dtype),
                is_output=self._is_backend_output(parameter),
            )
        if isinstance(parameter, Reference):
            dtype = parameter.dtype
            if isinstance(dtype, Dependency):
                return backend.DependentReference(
                    backend.Dependency(dtype.name),
                    is_output=self._is_backend_output(parameter),
                )
            return backend.Reference(
                self.normalize_dtype(dtype),
                is_output=self._is_backend_output(parameter),
            )
        if isinstance(parameter, Value):
            dtype = parameter.dtype
            if isinstance(dtype, Dependency):
                raise TypeError(
                    "Numba-CUDA-MLIR does not support dependent scalar values"
                )
            normalized_dtype = self.normalize_dtype(dtype)
            value_abi = (
                self._value_abis.get(parameter.name)
                if parameter.name is not None
                else None
            )
            if value_abi is not None:
                return value_abi
            return backend.Value(
                normalized_dtype,
                is_output=self._is_backend_output(parameter),
            )
        if isinstance(parameter, CxxFunction):
            dtype = parameter.dtype
            cpp = parameter.cpp
            if isinstance(dtype, Dependency):
                dependency = dtype
                dtype = dependency.resolve(specialization.template_arguments)
                # CxxFunction supports a dedicated type-expression placeholder
                # in addition to the bracketed operator-template convention.
                # Bare tokens are deliberately not replaced.
                cpp = cpp.replace(
                    f"{{{dependency.name}}}",
                    self.cpp_type(dtype),
                )
                cpp = cpp.replace(
                    f"<{dependency.name}>",
                    f"<{self.cpp_type(dtype)}>",
                )
            return backend.CxxFunction(
                cpp,
                self.normalize_dtype(dtype),
            )
        raise TypeError(
            f"unsupported Numba-CUDA-MLIR core parameter {parameter!r}"
        )

    def lower_cxx_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        del specialization
        if not isinstance(operator, CxxOperator):
            raise TypeError(f"expected CxxOperator, got {operator!r}")
        if not isinstance(operator.dtype, Dependency):
            return backend.CxxFunction(
                f"{operator.cpp}{{}}",
                self.normalize_dtype(operator.dtype),
            )
        return backend.DependentCxxOperator(
            backend.Dependency(operator.dtype.name),
            operator.cpp,
        )

    def lower_python_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        del specialization
        if not isinstance(operator, PythonOperator):
            raise TypeError(f"expected PythonOperator, got {operator!r}")
        return backend.DependentPythonOperator(
            self._resolvable(operator.ret_dtype),
            tuple(self._resolvable(dtype) for dtype in operator.arg_dtypes),
            backend.Constant(_normalize_numba_callable(operator.op)),
        )

    def lower_stateful_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        del specialization
        if not isinstance(operator, StatefulOperator):
            raise TypeError(f"expected StatefulOperator, got {operator!r}")
        return backend.DependentStatefulOperator(
            self._resolvable(operator.state_dtype),
            self._resolvable(operator.ret_dtype),
            tuple(self._resolvable(dtype) for dtype in operator.arg_dtypes),
            backend.Constant(_normalize_numba_callable(operator.op)),
            name=operator.name,
        )

    def lower_temp_storage(
        self,
        parameter: TempStorageParameter,
        *,
        specialization: AlgorithmSpec,
    ) -> backend.Pointer:
        del specialization
        return backend.Pointer(self.normalize_dtype(parameter.dtype))

    def materialize(
        self,
        specialization: AlgorithmSpec,
        *,
        storage_abi: StorageABI,
        execution_scope: SynchronizationScope,
        synchronization_scope: SynchronizationScope,
        extra_type_definitions: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> backend.Algorithm:
        """Build a backend algorithm from a specialized core specification.

        Validate named scalar ABI overrides and array input transforms against
        all matching core parameters before lowering. Scalar overrides must
        preserve provider dtype and output role; transforms apply only to
        input-only arrays. Copy each lowered descriptor before attaching its
        parameter name because an override may be reused by several parameters
        or specifications.

        Keep checked scalar overloads ahead of runtime pointer-offset overloads.
        Numba-CUDA-MLIR selects the first convertible signature, and the offset
        integer domain is intentionally broader than an exact scalar ABI. The
        stable ordering otherwise preserves the core's method order. Include the
        leading scratch parameter only for ``LEADING_POINTER`` storage,
        translate type declarations, and finish template substitution without
        compiling LTO.

        Parameters
        ----------
        specialization : AlgorithmSpec
            Core specification with ordered template arguments and overloads.
        storage_abi : StorageABI
            Whether the backend receives a leading scratch pointer or no
            scratch.
        execution_scope : SynchronizationScope
            Participating scope used by the source emitter to allocate scratch.
        synchronization_scope : SynchronizationScope
            Post-call synchronization for allocating wrappers; must be ``NONE``
            or match ``execution_scope``.
        extra_type_definitions : tuple, optional
            Backend type definitions prepended to the core's declarations,
            including any supporting link images.
        **kwargs : dict
            Reserved for the adapter interface; additional options are rejected.

        Returns
        -------
        Algorithm
            Specialized backend provider with copied, named parameter
            descriptors and explicit storage/execution contracts.

        Raises
        ------
        TypeError
            Options or scalar ABI declarations are unsupported, or parameter
            lowering encounters an unsupported descriptor.
        ValueError
            Named overrides/transforms do not match eligible core parameters,
            their dtypes or output roles conflict, or scope/ABI values are
            invalid.
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"unexpected Numba-CUDA-MLIR "
                f"materialization options: {unexpected}"
            )

        storage_abi = StorageABI(storage_abi)
        execution_scope = SynchronizationScope(execution_scope)
        synchronization_scope = SynchronizationScope(synchronization_scope)
        include_temp_storage = storage_abi is StorageABI.LEADING_POINTER

        invalid_value_abis = {
            name
            for name, value_abi in self._value_abis.items()
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value_abi, backend.Value)
            )
        }
        if invalid_value_abis:
            names = ", ".join(sorted(repr(name) for name in invalid_value_abis))
            raise TypeError(
                "Numba-CUDA-MLIR value ABI declarations require non-empty "
                f"parameter names and backend Value instances: {names}"
            )
        value_parameters_by_name = {
            name: tuple(
                parameter
                for method in specialization.parameters
                for parameter in method
                if parameter.name == name
            )
            for name in self._value_abis
        }
        unknown_value_abis = {
            name
            for name, parameters in value_parameters_by_name.items()
            if not parameters
        }
        if unknown_value_abis:
            names = ", ".join(sorted(unknown_value_abis))
            raise ValueError(f"unknown Numba-CUDA-MLIR value ABI(s): {names}")
        invalid_value_targets = {
            name
            for name, parameters in value_parameters_by_name.items()
            if any(
                not isinstance(parameter, Value)
                or isinstance(parameter, PointerOffset)
                or parameter.is_inout
                for parameter in parameters
            )
        }
        if invalid_value_targets:
            names = ", ".join(sorted(invalid_value_targets))
            raise ValueError(
                f"Numba-CUDA-MLIR value ABIs require "
                f"scalar Value parameters: {names}"
            )
        dtype_mismatches = {
            name
            for name, parameters in value_parameters_by_name.items()
            if any(
                self.normalize_dtype(parameter.dtype)
                != self._value_abis[name].provider_dtype
                for parameter in parameters
            )
        }
        if dtype_mismatches:
            names = ", ".join(sorted(dtype_mismatches))
            raise ValueError(
                "Numba-CUDA-MLIR value ABI provider dtypes do not match "
                f"core Value dtypes: {names}"
            )
        output_mismatches = {
            name
            for name, parameters in value_parameters_by_name.items()
            if any(
                self._is_backend_output(parameter)
                != self._value_abis[name].is_output
                for parameter in parameters
            )
        }
        if output_mismatches:
            names = ", ".join(sorted(output_mismatches))
            raise ValueError(
                "Numba-CUDA-MLIR value ABI output roles do not match core "
                f"Value parameters: {names}"
            )

        parameters_by_name = {
            name: tuple(
                parameter
                for method in specialization.parameters
                for parameter in method
                if parameter.name == name
            )
            for name in self._input_transforms
        }
        unknown_transforms = {
            name
            for name, parameters in parameters_by_name.items()
            if not parameters
        }
        if unknown_transforms:
            names = ", ".join(sorted(unknown_transforms))
            raise ValueError(
                f"unknown Numba-CUDA-MLIR input transform(s): {names}"
            )
        invalid_transforms = {
            name
            for name, parameters in parameters_by_name.items()
            if any(
                not isinstance(parameter, Array)
                or parameter.is_output
                or parameter.is_inout
                for parameter in parameters
            )
        }
        if invalid_transforms:
            names = ", ".join(sorted(invalid_transforms))
            raise ValueError(
                "Numba-CUDA-MLIR input transforms require input-only array "
                f"parameters: {names}"
            )

        # Pointer-offset overloads deliberately accept a broadly convertible
        # integer in the same position where ABI-checked scalar overloads
        # require an exact dtype. Numba-CUDA-MLIR selects the first convertible
        # overload, so keep the checked forms ahead of pointer-offset forms.
        # The stable sort otherwise preserves the canonical core ordering.
        ordered_parameters = sorted(
            specialization.parameters,
            key=lambda method: any(
                isinstance(parameter, PointerOffset)
                and parameter.static_value is None
                for parameter in method
            ),
        )
        methods = []
        for method in ordered_parameters:
            core_parameters = [
                parameter
                for parameter in method
                if include_temp_storage
                or not isinstance(parameter, TempStorageParameter)
            ]
            lowered_parameters = lower_method_parameters(
                self,
                specialization,
                method,
                include_temp_storage=include_temp_storage,
            )
            named_parameters = []
            for parameter, lowered in zip(core_parameters, lowered_parameters):
                # Scalar ABI overrides may be reused by several parameters
                # or specs.
                named = copy(lowered)
                named.parameter_name = parameter.name
                named_parameters.append(named)
            methods.append(named_parameters)

        definitions = [*extra_type_definitions]
        definitions.extend(
            SimpleNamespace(code=item.code, lto_irs=[])
            for item in specialization.type_definitions
        )
        algorithm = backend.Algorithm(
            specialization.struct_name,
            specialization.method_name,
            specialization.c_name,
            list(specialization.includes),
            [
                backend.TemplateParameter(parameter.name)
                for parameter in specialization.template_parameters
            ],
            methods,
            storage_abi=storage_abi,
            execution_scope=execution_scope,
            synchronization_scope=synchronization_scope,
            type_definitions=definitions,
            fake_return=specialization.fake_return,
            output_by_reference=specialization.output_by_reference,
        )
        template_arguments = {
            name: self.normalize_dtype(value)
            for name, value in specialization.ordered_specialization_arguments
        }
        return algorithm.specialize(template_arguments)


__all__ = ["NumbaMlirArrayInputTransform", "NumbaMlirCoreAdapter"]
