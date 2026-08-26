# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower backend-neutral cooperative specs into Numba-CUDA-MLIR wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

from numba_cuda_mlir import types

from cuda.coop._core import (
    INT8,
    INT32,
    INT64,
    UINT8,
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
    TempStorageParameter,
    Value,
    lower_method_parameters,
)

from .. import _types as backend


@dataclass(frozen=True)
class NumbaMlirArrayInputTransform:
    """Describe one elementwise input conversion in a generated CUB wrapper."""

    source_dtype: Any
    cpp_expression: str

    def __post_init__(self) -> None:
        if "{value}" not in self.cpp_expression:
            raise ValueError("array input transform must reference {value}")


def _optional_binding(value: Any) -> ArgumentBinding:
    """Preserve explicit bindings while retaining legacy presence sentinels."""

    if isinstance(value, ArgumentBinding):
        return value
    if value is None:
        return ArgumentBinding.omitted()
    return ArgumentBinding.runtime()


class NumbaMlirCoreAdapter(CoreBackendAdapter):
    """Translate core descriptors while retaining Numba-CUDA-MLIR linking and caching."""

    _BUILTIN_DTYPES = {
        INT8: types.int8,
        UINT8: types.uint8,
        INT32: types.int32,
        INT64: types.int64,
    }

    def __init__(
        self,
        *,
        input_transforms: Mapping[str, NumbaMlirArrayInputTransform] | None = None,
    ) -> None:
        self._input_transforms = dict(input_transforms or {})

    def normalize_dtype(self, dtype: Any) -> Any:
        if isinstance(dtype, BuiltinDType):
            try:
                return self._BUILTIN_DTYPES[dtype]
            except KeyError as exc:
                raise TypeError(f"unsupported core dtype {dtype.name!r}") from exc
        return dtype

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
    ) -> Any:
        if isinstance(parameter, PointerOffset):
            return backend.PointerOffset(
                self.normalize_dtype(parameter.dtype),
                parameter.pointer_arg_index,
                static_value=parameter.static_value,
            )
        if isinstance(parameter, Array):
            transform = self._input_transforms.get(parameter.name)
            if transform is not None:
                if parameter.is_output or parameter.is_inout:
                    raise ValueError(
                        f"input transform {parameter.name!r} targets an output array"
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
                if not isinstance(size, int) or isinstance(size, bool) or size < 1:
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
            if isinstance(parameter.dtype, (Dependency, Constant)) or isinstance(
                parameter.size, (Dependency, Constant)
            ):
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
                backend.PointerReference if parameter.deref_on_call else backend.Pointer
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
            return backend.Value(
                self.normalize_dtype(dtype),
                is_output=self._is_backend_output(parameter),
            )
        if isinstance(parameter, CxxFunction):
            dtype = parameter.dtype
            cpp = parameter.cpp
            if isinstance(dtype, Dependency):
                dependency = dtype
                dtype = dependency.resolve(specialization.template_arguments)
                # CxxFunction dependencies use the same bracketed placeholder
                # convention as DependentCxxOperator; bare tokens are not replaced.
                cpp = cpp.replace(
                    f"<{dependency.name}>",
                    f"<{self.cpp_type(dtype)}>",
                )
            return backend.CxxFunction(
                cpp,
                self.normalize_dtype(dtype),
            )
        raise TypeError(f"unsupported Numba-CUDA-MLIR core parameter {parameter!r}")

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
            [self._resolvable(dtype) for dtype in operator.arg_dtypes],
            self._resolvable(operator.op),
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
        return backend.DependentPythonOperator(
            self._resolvable(operator.ret_dtype),
            [self._resolvable(dtype) for dtype in operator.arg_dtypes],
            self._resolvable(operator.op),
        )

    def lower_temp_storage(
        self,
        parameter: TempStorageParameter,
        *,
        specialization: AlgorithmSpec,
    ) -> Any:
        del specialization
        return backend.Pointer(self.normalize_dtype(parameter.dtype))

    def materialize(
        self,
        specialization: AlgorithmSpec,
        *,
        include_temp_storage: bool = True,
        extra_type_definitions: tuple[Any, ...] = (),
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"unexpected Numba-CUDA-MLIR materialization options: {unexpected}"
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
            name for name, parameters in parameters_by_name.items() if not parameters
        }
        if unknown_transforms:
            names = ", ".join(sorted(unknown_transforms))
            raise ValueError(f"unknown Numba-CUDA-MLIR input transform(s): {names}")
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

        # Pointer-offset overloads deliberately accept an integer in the same
        # position where partial-tile overloads accept an exact int32 count.
        # Numba-CUDA-MLIR selects the first convertible overload, so keep the
        # more specific partial-tile forms ahead of pointer-offset forms.  The
        # stable sort otherwise preserves the canonical core ordering.
        ordered_parameters = sorted(
            specialization.parameters,
            key=lambda method: any(
                isinstance(parameter, PointerOffset) and parameter.static_value is None
                for parameter in method
            ),
        )
        methods = [
            list(
                lower_method_parameters(
                    self,
                    specialization,
                    method,
                    include_temp_storage=include_temp_storage,
                )
            )
            for method in ordered_parameters
        ]

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
