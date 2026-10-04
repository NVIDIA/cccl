# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Translate shared algorithm descriptions to Numba descriptors.

The common core describes C++ algorithms, parameters, and template arguments
without depending on a Python compiler. This adapter supplies the Numba types
and calling conventions needed to turn those descriptions into an
``Algorithm``. For example, an array parameter needs a pointer-based wrapper,
and a runtime valid-item count can require checked integer narrowing.

The adapter also carries scratch and synchronization requirements into source
generation. It creates a specialized algorithm description. Python callback
operators compile their LTO IR during specialization. The provider wrappers
compile later in ``make_invocable_from_specialization`` or after several
descriptions have been collected for a shared compilation.
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
    Algorithm,
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
    """Describe the element conversion a provider wrapper must generate.

    An input array can have a different element type from the CUB parameter.
    The adapter resolves the parameter's target type and fixed extent.
    Source generation then emits a local array of converted elements and
    passes it to the CUB call. Only input-only arrays support this conversion.

    Parameters
    ----------
    source_dtype : object
        Element type of the array supplied by the kernel.
    cpp_expression : str
        C++ expression for one converted element. It must contain ``{value}``,
        which source generation replaces with the input element expression.
        This record checks the placeholder; the C++ compiler checks the
        generated expression.
    """

    source_dtype: Any
    cpp_expression: str

    def __post_init__(self) -> None:
        if "{value}" not in self.cpp_expression:
            raise ValueError("array input transform must reference {value}")


def _optional_binding(value: object) -> ArgumentBinding:
    """Translate legacy presence markers without making their values static.

    Lowering factories accept both explicit binding descriptors and older
    arguments whose mere presence requested a runtime overload. Preserve an
    ``ArgumentBinding`` as supplied; otherwise ``None`` means omitted and
    every other object means runtime. In particular, a plain integer here is
    not a compile-time value. Callers must use ``ArgumentBinding.static`` to
    embed it.

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
    """Translate shared algorithm parameters to the Numba calling convention.

    The core describes the C++ operation without Numba types. This adapter
    maps its types and parameters to the descriptors used by Numba source
    generation. ``materialize`` produces a specialized backend ``Algorithm``;
    Python callbacks may compile during specialization. Provider-wrapper
    compilation and linking happen later.

    Named overrides cover cases the C++ signature alone cannot express. For
    example, a runtime item count needs a checked integer ABI, and an input
    conversion needs a local array with the target element type. The adapter
    checks overrides against every matching core parameter when it
    materializes an algorithm.

    Parameters
    ----------
    input_transforms : Mapping[str, NumbaMlirArrayInputTransform], optional
        Conversions keyed by core parameter name. Each target must be an
        input-only array with a positive specialized extent.
    value_abis : mapping of str to backend.Value, optional
        Scalar ABI descriptors keyed by core parameter name. An override must
        preserve the provider dtype and output role of each matching scalar.
        Construction copies the mappings; materialization copies parameter
        descriptors before attaching their names.
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
        """Map a shared builtin dtype to its matching Numba numeric type.

        Keep already backend-specific values unchanged. Reject a shared
        builtin that the backend mapping does not support.
        """

        if isinstance(dtype, BuiltinDType):
            try:
                return self._BUILTIN_DTYPES[dtype]
            except KeyError as exc:
                raise TypeError(
                    f"unsupported core dtype {dtype.name!r}"
                ) from exc
        return dtype

    def core_dtype(self, dtype: Any) -> Any:
        """Translate a Numba dtype for a shared core specialization.

        Provider factories call this before specializing the common algorithm.
        Return the matching ``BuiltinDType`` token, or pass ``dtype`` through
        when it has no builtin mapping.
        """

        return self._CORE_DTYPES.get(dtype, dtype)

    def cpp_type(self, dtype: Any) -> str:
        """Spell a specialization's dtype for generated C++ declarations.

        Normalize a shared builtin or accept a backend dtype, then return
        its builtin C++ spelling or ``storage_t`` for an opaque payload.
        """

        return backend.numba_type_to_cpp(self.normalize_dtype(dtype))

    def _resolvable(self, value: Any) -> Any:
        """Preserve a parameter dependency until template arguments are known.

        ``lower_parameter`` uses this for element dtypes and array lengths.
        ``value`` may be a shared ``Dependency``, a ``Constant``, or a direct
        value. Return the corresponding backend descriptor for substitution.
        Translate shared dependencies and constants to backend descriptors.
        Normalize dtype before wrapping a value as a backend constant.
        """

        if isinstance(value, Dependency):
            return backend.Dependency(value.name)
        if isinstance(value, Constant):
            return backend.Constant(value.value)
        return backend.Constant(self.normalize_dtype(value))

    @staticmethod
    def _is_backend_output(parameter: Any) -> bool:
        """Choose whether a C++ output becomes a backend return value.

        An explicit ``is_return`` overrides ``is_output``.
        """

        if parameter.is_return is None:
            return parameter.is_output
        return parameter.is_return

    def lower_parameter(
        self,
        parameter: Any,
        *,
        specialization: Algorithm,
    ) -> backend.Parameter:
        """Describe one core parameter in the Numba provider ABI.

        Preserve template dependencies for arrays, pointers, and references
        until provider construction resolves them. Scalar values require a
        concrete dtype; a named scalar ABI override can impose stricter
        runtime typing or checked narrowing. Output ownership follows
        ``is_return`` when explicitly set, otherwise ``is_output`` determines
        the backend return value.

        Named input transforms are different from ordinary arrays: resolve
        their extent and target dtype now so source generation can emit a
        fixed local array and per-element C++ conversions. Only input-only
        arrays may use this path. Dependent C++ functors substitute bracketed
        type placeholders only, leaving unrelated bare tokens unchanged.

        Parameters
        ----------
        parameter : object
            Core pointer offset, array, pointer, reference, value, or C++
            functor descriptor. Temporary storage is handled by
            ``lower_temp_storage``.
        specialization : Algorithm
            Core specialization providing template arguments for eager
            dependency resolution in transforms and C++ functors.

        Returns
        -------
        Parameter
            Backend descriptor; a scalar override may be the adapter's
            existing descriptor. ``materialize`` copies descriptors before
            attaching names.

        Raises
        ------
        TypeError
            The parameter kind is unsupported or a scalar value has a
            dependent dtype.
        ValueError
            An input transform targets an output/inout array or lacks a
            positive specialized integer extent.
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
                # Replace {T} in type expressions and <T> in templates. Leave
                # bare names intact so replacing a dtype cannot alter an
                # unrelated C++ identifier in the supplied expression.
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
        specialization: Algorithm,
    ) -> Any:
        """Translate a shared C++ functor into a backend operator descriptor.

        A concrete dtype creates a value-initialized ``CxxFunction``. A dtype
        dependency creates ``DependentCxxOperator`` so its bracketed
        placeholder is replaced only after template arguments are known.
        Neither path compiles a Python function or adds a runtime argument.
        """

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
        specialization: Algorithm,
    ) -> Any:
        """Defer a Python callback until its dtype dependencies are resolved.

        Translate return and argument types to backend resolvables and unwrap
        the outer device dispatcher to its Python function. The resulting
        ``DependentPythonOperator`` compiles a concrete device ABI during
        specialization; this adapter step does not compile the callback yet.
        """

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
        specialization: Algorithm,
    ) -> Any:
        """Translate a shared stateful operator into deferred Numba inputs.

        Adapt dtype dependencies and normalize the callback. Return a
        ``DependentStatefulOperator`` for later specialization, which compiles
        callback LTO with a leading typed state pointer. The state array
        remains a runtime provider operand.
        """

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
        specialization: Algorithm,
    ) -> backend.Pointer:
        """Describe scratch as a pointer in the provider's calling convention.

        The shared adapter traversal calls this for a ``TempStorageParameter``
        when the storage ABI includes a leading pointer. ``parameter.dtype``
        supplies its element type; ``specialization`` is part of the adapter
        interface but is unused here. Return a backend pointer descriptor.
        Later stages handle allocation, byte layout checks, and reuse
        barriers.
        """

        del specialization
        return backend.Pointer(self.normalize_dtype(parameter.dtype))

    def materialize(
        self,
        specialization: Algorithm,
        *,
        storage_abi: StorageABI,
        execution_scope: SynchronizationScope,
        synchronization_scope: SynchronizationScope,
        extra_type_definitions: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> backend.Algorithm:
        """Build a backend algorithm from a core specialization.

        Validate named scalar ABI overrides and array input transforms against
        all matching core parameters before lowering. Scalar overrides must
        preserve provider dtype and output role; transforms apply only to
        input-only arrays. Copy each lowered descriptor before attaching its
        parameter name because an override may be reused by several parameters
        or specializations.

        Keep checked scalar overloads ahead of runtime pointer-offset
        overloads. Numba-CUDA-MLIR selects the first convertible signature,
        and the offset integer domain is intentionally broader than an exact
        scalar ABI. The stable ordering otherwise preserves the core's method
        order. Include the leading scratch parameter only for
        ``LEADING_POINTER`` storage, translate type declarations, and finish
        template substitution. Specializing a Python operator compiles its
        callback LTO IR here. Provider-wrapper compilation and linking occur
        later when an invocable or shared compilation is materialized.

        Parameters
        ----------
        specialization : Algorithm
            Core specialization with ordered template arguments and overloads.
        storage_abi : StorageABI
            Select a leading scratch pointer or an ABI without scratch.
        execution_scope : SynchronizationScope
            Participating scope for scratch allocation by the source emitter.
        synchronization_scope : SynchronizationScope
            Post-call synchronization for allocating wrappers; must be
            ``NONE`` or match ``execution_scope``.
        extra_type_definitions : tuple, optional
            Backend type definitions prepended to the core's declarations,
            including any supporting link images.
        **kwargs : dict
            Reserved for the adapter interface; extra options are rejected.

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
            Named overrides/transforms do not match eligible core parameters
            or conflict in dtype or output role. Scope/ABI values are invalid.
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
        # require an exact dtype. Numba-CUDA-MLIR selects the first
        # convertible overload, so keep checked forms ahead of offset forms.
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
                # or specializations.
                named = copy(lowered)
                named.parameter_name = parameter.name
                named_parameters.append(named)
            methods.append(named_parameters)

        definitions = [*extra_type_definitions]
        definitions.extend(
            SimpleNamespace(code=item.code, lto_irs=[])
            for item in specialization.type_definitions
        )
        template_arguments = {
            name: self.normalize_dtype(value)
            for name, value in specialization.ordered_specialization_arguments
        }
        return backend.Algorithm(
            specialization.struct_name,
            specialization.method_name,
            specialization.c_name,
            list(specialization.includes),
            [
                backend.TemplateParameter(parameter.name)
                for parameter in specialization.template_parameters
            ],
            methods,
            template_arguments=template_arguments,
            storage_abi=storage_abi,
            execution_scope=execution_scope,
            synchronization_scope=synchronization_scope,
            type_definitions=definitions,
            fake_return=specialization.fake_return,
            output_by_reference=specialization.output_by_reference,
        )


__all__ = [
    "NumbaMlirArrayInputTransform",
    "NumbaMlirCoreAdapter",
    "_optional_binding",
]
