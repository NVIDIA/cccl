# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower shared-core algorithm specs into public-CUB CUTLASS shims."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int8, Int32, Int64, Uint8, Uint32

from cuda.coop._core import (
    INT8,
    INT32,
    INT64,
    UINT8,
    AlgorithmSpec,
    Array,
    BuiltinDType,
    Constant,
    CoreBackendAdapter,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupLoweringPlan,
    GroupLoweringTarget,
    Pointer,
    PointerOffset,
    PythonOperator,
    Reference,
    StatefulOperator,
    StorageOwnership,
    SynchronizationScope,
    TempStorageParameter,
    Value,
    lower_method_parameters,
)

from .._compiler import _rendering as _provider_rendering
from .._compiler._types import TYPE_SPECS

_ROOT_SCOPE = "cuda.coop.cutlass"


@dataclasses.dataclass(frozen=True)
class CutlassAbiParameter:
    """One flattened FFI parameter in a generated CUTLASS provider shim."""

    logical_name: str
    cpp_name: str
    cpp_type: str
    ffi_type: Any | None
    source: str
    item_index: int | None = None
    is_pointer: bool = False


@dataclasses.dataclass(frozen=True)
class CutlassLoweredParameter:
    """C++ wrapper fragments for one shared-core method parameter."""

    name: str
    call_expression: str | None
    abi_parameters: tuple[CutlassAbiParameter, ...] = ()
    declarations: tuple[str, ...] = ()
    epilogue: tuple[str, ...] = ()
    is_temp_storage: bool = False


@dataclasses.dataclass(frozen=True)
class CutlassArrayInputTransform:
    """Adapt one flattened input array before invoking the CUB collective."""

    source_dtype: Any
    cpp_expression: str

    def __post_init__(self) -> None:
        if "{value}" not in self.cpp_expression:
            raise ValueError("CUTLASS input transforms require a {value} placeholder")


@dataclasses.dataclass(frozen=True)
class CutlassRuntimeIntRange:
    """Inclusive and optional relational policy for a runtime ``int``."""

    logical_name: str
    minimum: int
    maximum: int
    clamp: bool = False
    modulus: int | None = None
    less_than_parameter: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.logical_name
            or self.logical_name[0].isdigit()
            or not self.logical_name.replace("_", "a").isalnum()
        ):
            raise ValueError("CUTLASS runtime range requires a C identifier")
        if any(
            not isinstance(bound, int) or isinstance(bound, bool)
            for bound in (self.minimum, self.maximum)
        ):
            raise TypeError("CUTLASS runtime range bounds must be integers")
        if self.minimum > self.maximum:
            raise ValueError("CUTLASS runtime range minimum exceeds maximum")
        if not isinstance(self.clamp, bool):
            raise TypeError("CUTLASS runtime range clamp must be a bool")
        if self.less_than_parameter is not None:
            if (
                not self.less_than_parameter
                or self.less_than_parameter[0].isdigit()
                or not self.less_than_parameter.replace("_", "a").isalnum()
            ):
                raise ValueError(
                    "CUTLASS runtime range relation requires a C identifier"
                )
            if self.less_than_parameter == self.logical_name:
                raise ValueError(
                    "CUTLASS runtime range relation requires distinct parameters"
                )
            if self.clamp or self.modulus is not None:
                raise ValueError(
                    "CUTLASS runtime range relations cannot clamp or normalize"
                )
        if self.modulus is not None:
            if (
                not isinstance(self.modulus, int)
                or isinstance(self.modulus, bool)
                or self.modulus < 1
            ):
                raise TypeError("CUTLASS runtime range modulus must be positive int")
            if self.clamp:
                raise ValueError("CUTLASS runtime range cannot clamp and normalize")
            if self.minimum != 0 or self.maximum != self.modulus - 1:
                raise ValueError(
                    "CUTLASS modulo range must span zero through modulus minus one"
                )


@dataclasses.dataclass(frozen=True, eq=False)
class CutlassCoreArtifact:
    """Plan-owned, runtime-value-free description of one CUB wrapper."""

    plan: GroupLoweringPlan
    specialization: AlgorithmSpec
    parameters: tuple[CutlassLoweredParameter, ...]
    kind: str
    symbol_name: str
    method_index: int = 0
    input_transforms: tuple[tuple[str, CutlassArrayInputTransform], ...] = ()
    output_initializers: tuple[tuple[str, str], ...] = ()
    output_value_initializers: tuple[tuple[str, str], ...] = ()
    runtime_int_ranges: tuple[CutlassRuntimeIntRange, ...] = ()
    external_scratch: bool = False

    @property
    def _base_semantic_key(self) -> tuple[Any, ...]:
        assert self.plan.artifact_key is not None
        return (
            self.plan.artifact_key,
            self.kind,
            self.method_index,
            self.input_transforms,
            self.output_initializers,
            self.output_value_initializers,
            self.runtime_int_ranges,
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        if not self.external_scratch:
            return self._base_semantic_key
        return "external_scratch", self._base_semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CutlassCoreArtifact):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def items_per_thread(self) -> int:
        value = self.specialization.template_arguments.get("ITEMS_PER_THREAD", 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("ITEMS_PER_THREAD must be a positive integer")
        return value

    @property
    def abi_parameters(self) -> tuple[CutlassAbiParameter, ...]:
        return tuple(
            abi_parameter
            for parameter in self.parameters
            for abi_parameter in parameter.abi_parameters
        )

    @property
    def ffi_param_types(self) -> tuple[Any, ...]:
        parameter_types = tuple(
            llvm.PointerType.get(0) if parameter.is_pointer else parameter.ffi_type
            for parameter in self.abi_parameters
        )
        if self.external_scratch:
            return *parameter_types, Uint32, Int32, Int32
        return parameter_types

    def bind_ffi_arguments(
        self,
        runtime_values: Mapping[str, Any],
        output_values: Mapping[str, Any],
        *,
        scratch_values: Sequence[Any] = (),
    ) -> tuple[Any, ...]:
        """Flatten logical runtime values in the generated ABI order."""

        arguments = []
        for parameter in self.abi_parameters:
            values = output_values if parameter.source == "output" else runtime_values
            if parameter.logical_name not in values:
                raise KeyError(
                    f"missing {parameter.source} value for {parameter.logical_name!r}"
                )
            value = values[parameter.logical_name]
            if parameter.item_index is not None:
                if not isinstance(value, Sequence):
                    raise TypeError(
                        f"{parameter.logical_name} must be a sequence for an "
                        "array parameter"
                    )
                try:
                    value = value[parameter.item_index]
                except IndexError as exc:
                    raise ValueError(
                        f"{parameter.logical_name} does not contain enough items"
                    ) from exc
            arguments.append(value)
        scratch_values = tuple(scratch_values)
        expected_scratch_values = 3 if self.external_scratch else 0
        if len(scratch_values) != expected_scratch_values:
            raise ValueError(
                "CUTLASS core artifact expected "
                f"{expected_scratch_values} scratch ABI values, got "
                f"{len(scratch_values)}"
            )
        return *arguments, *scratch_values

    @property
    def scratch_requirement_key(self) -> tuple[Any, ...]:
        """Identity of the instantiated CUB class whose layout is required."""

        if not self.external_scratch:
            raise ValueError("implementation-owned artifacts have no scratch probe")
        provenance = self.plan.provenance
        if provenance is None or provenance.library != "CUB":
            raise ValueError("external core scratch requires public-CUB provenance")
        adapter = CutlassCoreAdapter()
        return (
            "cub_temp_storage_layout",
            self.specialization.struct_name,
            tuple(
                (name, _render_template_argument(adapter, value))
                for name, value in self.specialization.ordered_template_arguments
            ),
        )

    @property
    def scratch_cpp_type(self) -> str:
        """Fully instantiated public-CUB temporary-storage type."""

        provenance = self.plan.provenance
        if provenance is None or provenance.library != "CUB":
            raise ValueError("external core scratch requires public-CUB provenance")
        cpp_class = provenance.cpp_class
        if not cpp_class.startswith("::"):
            cpp_class = f"::{cpp_class}"
        adapter = CutlassCoreAdapter()
        template_arguments = ", ".join(
            _render_template_argument(adapter, value)
            for _, value in self.specialization.ordered_template_arguments
        )
        return f"typename {cpp_class}<{template_arguments}>::TempStorage"


class CutlassCoreAdapter(CoreBackendAdapter):
    """Render backend-neutral descriptors through the CUTLASS bundle JIT."""

    _BUILTIN_DTYPES = {
        INT8: Int8,
        UINT8: Uint8,
        INT32: Int32,
        INT64: Int64,
    }
    _BUILTIN_CPP_TYPES = {
        Int8: "signed char",
        Uint8: "unsigned char",
        Int32: "int",
        Int64: "long long",
    }

    def normalize_dtype(self, dtype: Any) -> Any:
        if isinstance(dtype, BuiltinDType):
            try:
                return self._BUILTIN_DTYPES[dtype]
            except KeyError as exc:
                raise TypeError(f"unsupported core dtype {dtype.name!r}") from exc
        return dtype

    def cpp_type(self, dtype: Any) -> str:
        dtype = self.normalize_dtype(dtype)
        if dtype in TYPE_SPECS:
            return TYPE_SPECS[dtype].cpp_type
        try:
            return self._BUILTIN_CPP_TYPES[dtype]
        except KeyError as exc:
            raise TypeError(f"unsupported CUTLASS dtype {dtype!r}") from exc

    def _resolve(self, value: Any, specialization: AlgorithmSpec) -> Any:
        if isinstance(value, (Dependency, Constant)):
            value = value.resolve(specialization.template_arguments)
        return value

    def _resolved_dtype(self, value: Any, specialization: AlgorithmSpec) -> Any:
        return self.normalize_dtype(self._resolve(value, specialization))

    @staticmethod
    def _name(parameter: Any) -> str:
        name = getattr(parameter, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("CUTLASS core parameters require stable names")
        return name

    @staticmethod
    def _output_pointer(name: str, cpp_type: str) -> CutlassAbiParameter:
        return CutlassAbiParameter(
            logical_name=name,
            cpp_name=f"{name}_result",
            cpp_type=f"{cpp_type}*",
            ffi_type=None,
            source="output",
            is_pointer=True,
        )

    def lower_parameter(
        self,
        parameter: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> CutlassLoweredParameter:
        if isinstance(parameter, Array):
            name = self._name(parameter)
            dtype = self._resolved_dtype(parameter.dtype, specialization)
            cpp_type = self.cpp_type(dtype)
            size = self._resolve(parameter.size, specialization)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{name} array extent must be a positive integer")
            abi_parameters = []
            declarations = []
            epilogue = []
            if not parameter.is_output or parameter.is_inout:
                abi_parameters.extend(
                    CutlassAbiParameter(
                        logical_name=name,
                        cpp_name=f"{name}_{index}",
                        cpp_type=cpp_type,
                        ffi_type=dtype,
                        source="runtime",
                        item_index=index,
                    )
                    for index in range(size)
                )
                values = ", ".join(f"{name}_{index}" for index in range(size))
                declarations.append(f"  {cpp_type} {name}[{size}] = {{{values}}};")
            else:
                declarations.append(f"  {cpp_type} {name}[{size}];")
            if parameter.is_output or parameter.is_inout:
                abi_parameters.append(self._output_pointer(name, cpp_type))
                epilogue.extend(
                    f"  {name}_result[{index}] = {name}[{index}];"
                    for index in range(size)
                )
            return CutlassLoweredParameter(
                name=name,
                call_expression=name,
                abi_parameters=tuple(abi_parameters),
                declarations=tuple(declarations),
                epilogue=tuple(epilogue),
            )

        if isinstance(parameter, PointerOffset):
            name = self._name(parameter)
            dtype = self._resolved_dtype(parameter.dtype, specialization)
            cpp_type = self.cpp_type(dtype)
            return CutlassLoweredParameter(
                name=name,
                call_expression=name,
                abi_parameters=(
                    CutlassAbiParameter(
                        logical_name=name,
                        cpp_name=name,
                        cpp_type=cpp_type,
                        ffi_type=dtype,
                        source="runtime",
                    ),
                ),
            )

        if isinstance(parameter, Pointer):
            name = self._name(parameter)
            dtype = self._resolved_dtype(parameter.dtype, specialization)
            cpp_type = self.cpp_type(dtype)
            qualifiers = " __restrict__" if parameter.restrict else ""
            expression = f"*{name}" if parameter.deref_on_call else name
            if getattr(parameter, "is_inout", False):
                raise NotImplementedError(
                    "CUTLASS core pointer parameters do not support inout values"
                )
            return CutlassLoweredParameter(
                name=name,
                call_expression=expression,
                abi_parameters=(
                    CutlassAbiParameter(
                        logical_name=name,
                        cpp_name=name,
                        cpp_type=f"{cpp_type}*{qualifiers}",
                        ffi_type=None,
                        source="output" if parameter.is_output else "runtime",
                        is_pointer=True,
                    ),
                ),
            )

        if isinstance(parameter, (Reference, Value)):
            name = self._name(parameter)
            dtype = self._resolved_dtype(parameter.dtype, specialization)
            cpp_type = self.cpp_type(dtype)
            abi_parameters = []
            declarations = []
            epilogue = []
            if parameter.is_output or parameter.is_inout:
                input_name = f"{name}_input"
                if parameter.is_inout:
                    abi_parameters.append(
                        CutlassAbiParameter(
                            logical_name=name,
                            cpp_name=input_name,
                            cpp_type=cpp_type,
                            ffi_type=dtype,
                            source="runtime",
                        )
                    )
                    declarations.append(f"  {cpp_type} {name} = {input_name};")
                else:
                    declarations.append(f"  {cpp_type} {name}{{}};")
                abi_parameters.append(self._output_pointer(name, cpp_type))
                epilogue.append(f"  *{name}_result = {name};")
            else:
                abi_parameters.append(
                    CutlassAbiParameter(
                        logical_name=name,
                        cpp_name=name,
                        cpp_type=cpp_type,
                        ffi_type=dtype,
                        source="runtime",
                    )
                )
            return CutlassLoweredParameter(
                name=name,
                call_expression=name,
                abi_parameters=tuple(abi_parameters),
                declarations=tuple(declarations),
                epilogue=tuple(epilogue),
            )

        if isinstance(parameter, CxxFunction):
            name = self._name(parameter)
            cpp = parameter.cpp
            if isinstance(parameter.dtype, Dependency):
                dtype = self._resolved_dtype(parameter.dtype, specialization)
                cpp = cpp.replace(
                    f"<{parameter.dtype.name}>",
                    f"<{self.cpp_type(dtype)}>",
                )
            return CutlassLoweredParameter(name=name, call_expression=cpp)

        raise TypeError(f"unsupported CUTLASS core parameter {parameter!r}")

    def lower_cxx_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> CutlassLoweredParameter:
        if not isinstance(operator, CxxOperator):
            raise TypeError(f"expected CxxOperator, got {operator!r}")
        name = self._name(operator)
        cpp = operator.cpp
        if isinstance(operator.dtype, Dependency):
            dtype = self._resolved_dtype(operator.dtype, specialization)
            cpp = cpp.replace(
                f"<{operator.dtype.name}>",
                f"<{self.cpp_type(dtype)}>",
            )
        return CutlassLoweredParameter(
            name=name,
            call_expression=f"{cpp}{{}}",
        )

    def lower_python_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> CutlassLoweredParameter:
        del specialization
        if not isinstance(operator, PythonOperator):
            raise TypeError(f"expected PythonOperator, got {operator!r}")
        raise NotImplementedError(
            "CUTLASS public-CUB lowering does not support Python operators"
        )

    def lower_stateful_operator(
        self,
        operator: Any,
        *,
        specialization: AlgorithmSpec,
    ) -> CutlassLoweredParameter:
        del specialization
        if not isinstance(operator, StatefulOperator):
            raise TypeError(f"expected StatefulOperator, got {operator!r}")
        raise NotImplementedError(
            "CUTLASS public-CUB lowering does not support stateful Python operators"
        )

    def lower_temp_storage(
        self,
        parameter: TempStorageParameter,
        *,
        specialization: AlgorithmSpec,
    ) -> CutlassLoweredParameter:
        del specialization
        return CutlassLoweredParameter(
            name=self._name(parameter),
            call_expression=None,
            is_temp_storage=True,
        )

    def materialize(
        self,
        specialization: AlgorithmSpec,
        *,
        plan: GroupLoweringPlan,
        kind: str,
        symbol_name: str | None = None,
        method_index: int = 0,
        input_transforms: Mapping[str, CutlassArrayInputTransform] | None = None,
        output_initializers: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        output_value_initializers: Mapping[str, str] | None = None,
        runtime_int_ranges: Sequence[CutlassRuntimeIntRange] = (),
        external_scratch: bool = False,
        **kwargs: Any,
    ) -> CutlassCoreArtifact:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected CUTLASS materialization options: {unexpected}")
        plan.require_supported()
        if plan.target not in {
            GroupLoweringTarget.CUB_BLOCK,
            GroupLoweringTarget.CUB_WARP,
        }:
            raise ValueError("CUTLASS core artifacts require a CUB lowering plan")
        if not isinstance(plan.implementation, AlgorithmSpec):
            raise TypeError("CUTLASS core artifacts require an AlgorithmSpec")
        if plan.implementation.semantic_key != specialization.semantic_key:
            raise ValueError("CUTLASS specialization does not match its plan")
        if not isinstance(external_scratch, bool):
            raise TypeError("external_scratch must be a bool")
        expected_storage_ownership = (
            StorageOwnership.CALLER
            if external_scratch
            else StorageOwnership.IMPLEMENTATION
        )
        if plan.temp_storage is None or (
            plan.temp_storage.ownership is not expected_storage_ownership
        ):
            raise ValueError(
                "CUTLASS core artifact storage ownership does not match its "
                "external-scratch mode"
            )
        if external_scratch:
            temp_storage = plan.temp_storage
            if plan.target is not GroupLoweringTarget.CUB_BLOCK:
                raise ValueError("external CUTLASS core scratch is block-scoped only")
            if (
                temp_storage.address_space != "shared"
                or temp_storage.instances != 1
                or temp_storage.instance_index != "cta"
                or not temp_storage.exact_layout_required
            ):
                raise ValueError(
                    "external CUTLASS core scratch requires exact caller-owned "
                    "CTA shared storage"
                )
        if not isinstance(kind, str) or not kind:
            raise ValueError("CUTLASS core artifact kind must not be empty")
        if method_index < 0:
            raise ValueError("CUTLASS method_index is out of range")
        try:
            method = specialization.parameters[method_index]
        except IndexError as exc:
            raise ValueError("CUTLASS method_index is out of range") from exc
        if isinstance(output_initializers, Mapping):
            if output_value_initializers is not None:
                raise ValueError(
                    "output initializers may be supplied either as value mappings "
                    "or copy pairs, not both"
                )
            output_value_initializers = output_initializers
            output_initializers = ()
        output_initializers = tuple(tuple(pair) for pair in output_initializers)
        if any(
            len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
            for pair in output_initializers
        ):
            raise TypeError(
                "CUTLASS output initializers must be (output, input) name pairs"
            )
        output_names = [output_name for output_name, _ in output_initializers]
        if len(output_names) != len(set(output_names)):
            raise ValueError("CUTLASS output initializer targets must be unique")
        method_by_name = {
            getattr(parameter, "name", None): parameter for parameter in method
        }
        normalized_runtime_int_ranges = tuple(runtime_int_ranges)
        if any(
            not isinstance(runtime_range, CutlassRuntimeIntRange)
            for runtime_range in normalized_runtime_int_ranges
        ):
            raise TypeError(
                "CUTLASS runtime int ranges must contain CutlassRuntimeIntRange values"
            )
        range_names = [
            runtime_range.logical_name
            for runtime_range in normalized_runtime_int_ranges
        ]
        if len(range_names) != len(set(range_names)):
            raise ValueError("CUTLASS runtime int range parameters must be unique")
        for runtime_range in normalized_runtime_int_ranges:
            descriptor = method_by_name.get(runtime_range.logical_name)
            if not isinstance(descriptor, (Reference, Value)) or getattr(
                descriptor,
                "is_output",
                False,
            ):
                raise ValueError(
                    "CUTLASS runtime int ranges must name input scalar parameters"
                )
            if (
                self.cpp_type(self._resolved_dtype(descriptor.dtype, specialization))
                != "int"
            ):
                raise TypeError("CUTLASS runtime int ranges require int parameters")
            relation_name = runtime_range.less_than_parameter
            if relation_name is not None:
                relation_descriptor = method_by_name.get(relation_name)
                if not isinstance(relation_descriptor, (Reference, Value)) or getattr(
                    relation_descriptor,
                    "is_output",
                    False,
                ):
                    raise ValueError(
                        "CUTLASS runtime int range relations must name input "
                        "scalar parameters"
                    )
                if (
                    self.cpp_type(
                        self._resolved_dtype(
                            relation_descriptor.dtype,
                            specialization,
                        )
                    )
                    != "int"
                ):
                    raise TypeError(
                        "CUTLASS runtime int range relations require int parameters"
                    )
        for output_name, input_name in output_initializers:
            if output_name not in method_by_name or input_name not in method_by_name:
                raise ValueError(
                    "CUTLASS output initializers must name method parameters"
                )
            if not getattr(method_by_name[output_name], "is_output", False):
                raise ValueError(
                    f"CUTLASS output initializer target {output_name!r} is not output"
                )
            input_parameter = method_by_name[input_name]
            if getattr(input_parameter, "is_output", False) and not getattr(
                input_parameter,
                "is_inout",
                False,
            ):
                raise ValueError(
                    f"CUTLASS output initializer source {input_name!r} is output-only"
                )
        parameters = lower_method_parameters(
            self,
            specialization,
            method,
            include_temp_storage=True,
        )
        normalized_input_transforms = tuple(
            sorted((input_transforms or {}).items(), key=lambda item: item[0])
        )
        normalized_output_value_initializers = tuple(
            sorted((output_value_initializers or {}).items(), key=lambda item: item[0])
        )
        parameters = self._apply_parameter_adapters(
            specialization,
            method,
            parameters,
            input_transforms=dict(normalized_input_transforms),
            output_initializers=dict(normalized_output_value_initializers),
        )
        if sum(parameter.is_temp_storage for parameter in parameters) != 1:
            raise ValueError(
                "CUTLASS public-CUB wrappers require exactly one TempStorage parameter"
            )
        generated_symbol = symbol_name is None
        if generated_symbol:
            digest = hashlib.sha256(
                repr(
                    (
                        plan.artifact_key,
                        kind,
                        method_index,
                        normalized_input_transforms,
                        output_initializers,
                        normalized_output_value_initializers,
                        normalized_runtime_int_ranges,
                    )
                ).encode("utf-8", errors="backslashreplace")
            ).hexdigest()[:16]
            symbol_name = (
                f"cuda_coop_cutlass_{specialization.c_name}_"
                f"{specialization.method_name.lower()}_{digest}"
            )
        external_suffix = "_external_scratch"
        if external_scratch and symbol_name.endswith(external_suffix):
            symbol_name = symbol_name[: -len(external_suffix)]
        if (
            method_index
            and (external_scratch or not generated_symbol)
            and not symbol_name.endswith(f"_m{method_index}")
        ):
            symbol_name = f"{symbol_name}_m{method_index}"
        if external_scratch:
            symbol_name = f"{symbol_name}{external_suffix}"
        return CutlassCoreArtifact(
            plan=plan,
            specialization=specialization,
            parameters=parameters,
            kind=kind,
            symbol_name=symbol_name,
            method_index=method_index,
            output_initializers=output_initializers,
            input_transforms=normalized_input_transforms,
            output_value_initializers=normalized_output_value_initializers,
            runtime_int_ranges=normalized_runtime_int_ranges,
            external_scratch=external_scratch,
        )

    def _apply_parameter_adapters(
        self,
        specialization: AlgorithmSpec,
        method: tuple[Any, ...],
        parameters: tuple[CutlassLoweredParameter, ...],
        *,
        input_transforms: Mapping[str, CutlassArrayInputTransform],
        output_initializers: Mapping[str, str],
    ) -> tuple[CutlassLoweredParameter, ...]:
        remaining_transforms = dict(input_transforms)
        remaining_initializers = dict(output_initializers)
        adapted = []
        for descriptor, lowered in zip(method, parameters, strict=True):
            transform = remaining_transforms.pop(lowered.name, None)
            initializer = remaining_initializers.pop(lowered.name, None)
            if transform is not None:
                if not isinstance(descriptor, Array) or descriptor.is_output:
                    raise TypeError(
                        f"CUTLASS input transform {lowered.name!r} requires an "
                        "input-only Array parameter"
                    )
                source_dtype = self.normalize_dtype(transform.source_dtype)
                source_cpp_type = self.cpp_type(source_dtype)
                target_dtype = self._resolved_dtype(descriptor.dtype, specialization)
                target_cpp_type = self.cpp_type(target_dtype)
                size = self._resolve(descriptor.size, specialization)
                if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                    raise ValueError(
                        f"{lowered.name} array extent must be a positive integer"
                    )
                abi_parameters = tuple(
                    CutlassAbiParameter(
                        logical_name=lowered.name,
                        cpp_name=f"{lowered.name}_{index}",
                        cpp_type=source_cpp_type,
                        ffi_type=source_dtype,
                        source="runtime",
                        item_index=index,
                    )
                    for index in range(size)
                )
                values = ", ".join(
                    transform.cpp_expression.format(value=f"{lowered.name}_{index}")
                    for index in range(size)
                )
                lowered = dataclasses.replace(
                    lowered,
                    abi_parameters=abi_parameters,
                    declarations=(
                        f"  {target_cpp_type} {lowered.name}[{size}] = {{{values}}};",
                    ),
                )
            if initializer is not None:
                if (
                    not isinstance(descriptor, Array)
                    or not descriptor.is_output
                    or descriptor.is_inout
                ):
                    raise TypeError(
                        f"CUTLASS output initializer {lowered.name!r} requires "
                        "an output-only Array parameter"
                    )
                dtype = self._resolved_dtype(descriptor.dtype, specialization)
                cpp_type = self.cpp_type(dtype)
                size = self._resolve(descriptor.size, specialization)
                if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                    raise ValueError(
                        f"{lowered.name} array extent must be a positive integer"
                    )
                values = ", ".join(initializer for _ in range(size))
                lowered = dataclasses.replace(
                    lowered,
                    declarations=(
                        f"  {cpp_type} {lowered.name}[{size}] = {{{values}}};",
                    ),
                )
            adapted.append(lowered)
        if remaining_transforms:
            names = ", ".join(sorted(remaining_transforms))
            raise ValueError(f"unknown CUTLASS input transform parameter(s): {names}")
        if remaining_initializers:
            names = ", ".join(sorted(remaining_initializers))
            raise ValueError(
                f"unknown CUTLASS output initializer parameter(s): {names}"
            )
        return tuple(adapted)


def with_caller_owned_core_temp_storage(
    plan: GroupLoweringPlan,
) -> GroupLoweringPlan:
    """Return a block-CUB plan whose exact scratch is supplied by the caller."""

    if plan.target is not GroupLoweringTarget.CUB_BLOCK:
        raise ValueError("external CUTLASS core scratch is block-scoped only")
    temp_storage = plan.temp_storage
    if temp_storage is None:
        raise ValueError("CUTLASS core plan requires a temporary-storage contract")
    return dataclasses.replace(
        plan,
        temp_storage=dataclasses.replace(
            temp_storage,
            ownership=StorageOwnership.CALLER,
            address_space="shared",
            cpp_type="typename implementation_type::TempStorage",
            instances=1,
            instance_index="cta",
            exact_layout_required=True,
        ),
    )


def _render_template_argument(adapter: CutlassCoreAdapter, value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return value
    return adapter.cpp_type(value)


def _render_type_definitions(artifact: CutlassCoreArtifact) -> list[str]:
    lines = []
    for definition in artifact.specialization.type_definitions:
        digest = (
            hashlib.sha256(
                repr(definition.semantic_key).encode("utf-8", errors="backslashreplace")
            )
            .hexdigest()[:16]
            .upper()
        )
        guard = f"CUDA_COOP_CUTLASS_TYPE_{digest}"
        lines.extend(
            [
                f"#ifndef {guard}",
                f"#define {guard}",
                *definition.code.splitlines(),
                "#endif",
            ]
        )
    return lines


def _storage_lines(artifact: CutlassCoreArtifact) -> tuple[list[str], str]:
    if artifact.external_scratch:
        return (
            [
                "  constexpr unsigned long long required_temp_bytes =",
                "      (unsigned long long)sizeof(typename implementation_type::TempStorage);",
                "  constexpr unsigned long long required_temp_alignment =",
                "      (unsigned long long)alignof(typename implementation_type::TempStorage);",
                "  if (temp_storage_bytes <= 0 ||",
                "      (unsigned long long)temp_storage_bytes < required_temp_bytes ||",
                "      ((unsigned long long)temp_storage_smem_addr &",
                "       (required_temp_alignment - 1ull)) != 0ull) {",
                '    asm volatile("trap;");',
                "  }",
                "  void* temp_storage_ptr =",
                "      cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr);",
                "  auto* storage_ptr = reinterpret_cast<",
                "      typename implementation_type::TempStorage*>(temp_storage_ptr);",
            ],
            "*storage_ptr",
        )
    if artifact.plan.target is GroupLoweringTarget.CUB_BLOCK:
        return (
            ["  __shared__ typename implementation_type::TempStorage storage;"],
            "storage",
        )
    participation = artifact.plan.participation
    if participation is None or participation.exact_block_dim is None:
        raise ValueError("CUB warp artifacts require exact block dimensions")
    x, y, z = participation.exact_block_dim
    block_threads = x * y * z
    resolved_group = artifact.plan.resolved_group
    if resolved_group.kind == "warp":
        logical_width = 32
    elif resolved_group.kind == "threads_within_warp":
        logical_width = resolved_group.static_size
        assert logical_width is not None
    else:
        raise ValueError("CUB warp artifacts require a warp group")
    if block_threads < logical_width or block_threads % logical_width != 0:
        raise ValueError("CUB warp artifacts require complete logical-warp partitions")
    instances = block_threads // logical_width
    return (
        [
            "  __shared__ typename implementation_type::TempStorage "
            f"storage[{instances}];",
            "  unsigned int storage_instance =",
            f"      cuda_coop_cutlass_linear_tid() / {logical_width}u;",
        ],
        "storage[storage_instance]",
    )


def _storage_reuse_barrier(artifact: CutlassCoreArtifact) -> str | None:
    synchronization = artifact.plan.synchronization
    if synchronization is None:
        raise ValueError("CUTLASS core artifacts require synchronization metadata")
    barrier = synchronization.storage_reuse_barrier
    if barrier is SynchronizationScope.BLOCK:
        return "  cuda_coop_cutlass_block_sync();"
    if barrier is SynchronizationScope.WARP:
        group = artifact.plan.resolved_group
        logical_width = 32 if group.kind == "warp" else group.static_size
        if group.kind not in {"warp", "threads_within_warp"} or not isinstance(
            logical_width,
            int,
        ):
            raise ValueError("CUTLASS warp barrier requires a static warp group")
        return f"  cuda_coop_cutlass_warp_sync({logical_width}u);"
    if barrier is SynchronizationScope.NONE:
        return None
    raise ValueError("CUTLASS core artifact has an unsupported storage barrier")


def _render_output_initializers(
    adapter: CutlassCoreAdapter,
    artifact: CutlassCoreArtifact,
) -> list[str]:
    method = artifact.specialization.parameters[artifact.method_index]
    method_by_name = {
        getattr(parameter, "name", None): parameter for parameter in method
    }
    lines = []
    for output_name, input_name in artifact.output_initializers:
        output_parameter = method_by_name[output_name]
        input_parameter = method_by_name[input_name]
        if isinstance(output_parameter, Array) and isinstance(input_parameter, Array):
            output_size = adapter._resolve(
                output_parameter.size,
                artifact.specialization,
            )
            input_size = adapter._resolve(
                input_parameter.size,
                artifact.specialization,
            )
            if output_size != input_size:
                raise ValueError("CUTLASS array output initializer extents must match")
            lines.extend(
                f"  {output_name}[{index}] = {input_name}[{index}];"
                for index in range(output_size)
            )
            continue
        if isinstance(output_parameter, (Reference, Value)) and isinstance(
            input_parameter,
            (Reference, Value),
        ):
            lines.append(f"  {output_name} = {input_name};")
            continue
        raise TypeError(
            "CUTLASS output initializers require matching scalar or array parameters"
        )
    return lines


def _render_runtime_int_ranges(artifact: CutlassCoreArtifact) -> list[str]:
    lines = []
    for runtime_range in artifact.runtime_int_ranges:
        name = runtime_range.logical_name
        if runtime_range.modulus is not None:
            lines.extend(
                (
                    f"  if ({name} < 0) {{",
                    '    asm volatile("trap;");',
                    "  }",
                    f"  {name} %= {runtime_range.modulus};",
                )
            )
        elif runtime_range.clamp:
            lines.extend(
                (
                    f"  if ({name} < {runtime_range.minimum}) {{",
                    f"    {name} = {runtime_range.minimum};",
                    f"  }} else if ({name} > {runtime_range.maximum}) {{",
                    f"    {name} = {runtime_range.maximum};",
                    "  }",
                )
            )
        else:
            lines.extend(
                (
                    f"  if ({name} < {runtime_range.minimum} || "
                    f"{name} > {runtime_range.maximum}) {{",
                    '    asm volatile("trap;");',
                    "  }",
                )
            )
    for runtime_range in artifact.runtime_int_ranges:
        relation_name = runtime_range.less_than_parameter
        if relation_name is not None:
            lines.extend(
                (
                    f"  if ({runtime_range.logical_name} >= {relation_name}) {{",
                    '    asm volatile("trap;");',
                    "  }",
                )
            )
    return lines


def render_cutlass_core_artifact(artifact: CutlassCoreArtifact) -> list[str]:
    """Render one typed wrapper containing exactly one public-CUB call."""

    adapter = CutlassCoreAdapter()
    # Re-materialization validation keeps stale or manually forged requests from
    # bypassing the plan/spec invariant at source-generation time.
    validated = adapter.materialize(
        artifact.specialization,
        plan=artifact.plan,
        kind=artifact.kind,
        symbol_name=artifact.symbol_name,
        method_index=artifact.method_index,
        input_transforms=dict(artifact.input_transforms),
        output_initializers=artifact.output_initializers,
        output_value_initializers=dict(artifact.output_value_initializers),
        runtime_int_ranges=artifact.runtime_int_ranges,
        external_scratch=artifact.external_scratch,
    )
    abi_parameters = validated.abi_parameters
    signature_parameters = [
        f"{parameter.cpp_type} {parameter.cpp_name}" for parameter in abi_parameters
    ]
    if validated.external_scratch:
        signature_parameters.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    signature = ", ".join(signature_parameters)
    template_arguments = ", ".join(
        _render_template_argument(adapter, value)
        for _, value in validated.specialization.ordered_template_arguments
    )
    provenance = validated.plan.provenance
    if provenance is None or provenance.library != "CUB":
        raise ValueError("CUTLASS core artifacts require public-CUB provenance")
    cpp_class = provenance.cpp_class
    if not cpp_class.startswith("::"):
        cpp_class = f"::{cpp_class}"
    storage_lines, storage_expression = _storage_lines(validated)
    declarations = [
        line for parameter in validated.parameters for line in parameter.declarations
    ]
    output_initializers = _render_output_initializers(adapter, validated)
    runtime_int_ranges = _render_runtime_int_ranges(validated)
    call_arguments = [
        parameter.call_expression
        for parameter in validated.parameters
        if parameter.call_expression is not None
    ]
    epilogue = [
        line for parameter in validated.parameters for line in parameter.epilogue
    ]
    barrier = _storage_reuse_barrier(validated)
    barrier_lines = []
    if barrier is not None:
        if validated.external_scratch:
            barrier_lines = [
                "  if (temp_storage_auto_sync != 0) {",
                f"  {barrier}",
                "  }",
            ]
        else:
            barrier_lines = [barrier]
    type_definitions = _render_type_definitions(validated)
    linkage_safe_definitions = (
        ["}", *type_definitions, 'extern "C" {'] if type_definitions else []
    )
    return [
        *linkage_safe_definitions,
        f"void {validated.symbol_name}({signature}) {{",
        f"  using implementation_type = {cpp_class}<{template_arguments}>;",
        *storage_lines,
        *declarations,
        *output_initializers,
        *runtime_int_ranges,
        f"  implementation_type({storage_expression})."
        f"{validated.specialization.method_name}("
        f"{', '.join(call_arguments)});",
        *barrier_lines,
        *epilogue,
        "}",
    ]


def register_cutlass_core_renderer(
    kind: str,
    *,
    includes: tuple[str, ...],
) -> None:
    """Register the shared renderer for one AlgorithmSpec include family."""

    # Shared-core CxxOperator descriptors use cuda::std function objects. Keep
    # that dependency at the reusable renderer boundary instead of relying on
    # a primitive header to include it transitively.
    includes = tuple(dict.fromkeys((*includes, "cuda/std/functional")))
    include_lines = tuple(f"#include <{include}>" for include in includes)
    _provider_rendering.register_bundle_renderer(
        kind,
        render=render_cutlass_core_artifact,
        include_lines=include_lines,
        cccl_headers=tuple((f"#include <{include}>", include) for include in includes),
        scratch_layout_probe=lambda artifact: (
            _provider_rendering.make_scratch_layout_probe(
                artifact.scratch_requirement_key,
                artifact.scratch_cpp_type,
            )
            if artifact.external_scratch
            else None
        ),
    )


__all__ = [
    "CutlassAbiParameter",
    "CutlassArrayInputTransform",
    "CutlassCoreAdapter",
    "CutlassCoreArtifact",
    "CutlassLoweredParameter",
    "register_cutlass_core_renderer",
    "render_cutlass_core_artifact",
    "with_caller_owned_core_temp_storage",
]
