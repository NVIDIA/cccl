# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral primitive parameter and operator descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Mapping

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class ArgumentKind(str, Enum):
    """Whether a method argument is encoded statically or passed at runtime."""

    STATIC = "static"
    RUNTIME = "runtime"


class ParameterRole(str, Enum):
    """Semantic role of a parameter in a cooperative primitive call."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    CONSTANT = "constant"
    TEMP_STORAGE = "temp_storage"
    OPERATOR = "operator"
    STATE = "state"


@dataclass(frozen=True)
class ParameterClassification:
    name: str | None
    kind: ArgumentKind
    role: ParameterRole


@dataclass(frozen=True)
class BuiltinDType:
    """Backend-neutral scalar dtype used by generated primitive ABIs."""

    name: str


INT8 = BuiltinDType("int8")
UINT8 = BuiltinDType("uint8")
INT32 = BuiltinDType("int32")
INT64 = BuiltinDType("int64")


class SubstitutionFailure(ValueError):
    """A dependent signature cannot be formed for a specialization."""


@dataclass(frozen=True)
class TemplateParameter:
    name: str


@dataclass(frozen=True)
class Dependency:
    name: str

    @property
    def dep(self) -> str:
        """Compatibility spelling used by existing backend descriptors."""

        return self.name

    def resolve(self, template_arguments: Mapping[str, Any]) -> Any:
        if self.name not in template_arguments:
            raise SubstitutionFailure(f"Template argument {self.name} not provided")
        value = template_arguments[self.name]
        if value is None:
            raise SubstitutionFailure(f"Template argument {self.name} is None")
        return value


@dataclass(frozen=True)
class Constant:
    value: Any

    @property
    def val(self) -> Any:
        """Compatibility spelling used by existing backend descriptors."""

        return self.value

    def resolve(self, _template_arguments: Mapping[str, Any]) -> Any:
        return self.value


@dataclass(frozen=True)
class RuntimeValue:
    """Marker telling a builder that a scalar must remain runtime-provided."""

    name: str | None = None


class _RuntimeParameter:
    @property
    def argument_kind(self) -> ArgumentKind:
        return ArgumentKind.RUNTIME

    @property
    def role(self) -> ParameterRole:
        if getattr(self, "is_inout", False):
            return ParameterRole.INOUT
        if getattr(self, "is_output", False):
            return ParameterRole.OUTPUT
        return ParameterRole.INPUT


class _StaticParameter:
    @property
    def argument_kind(self) -> ArgumentKind:
        return ArgumentKind.STATIC

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.OPERATOR


@dataclass(frozen=True)
class Value(_RuntimeParameter):
    dtype: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None


@dataclass(frozen=True)
class PointerOffset(Value):
    """Scalar element offset applied to an earlier pointer argument.

    ``pointer_arg_index`` is the zero-based method-argument position after the
    leading :class:`TempStorageParameter` is removed. It must identify an
    earlier pointer in the same method signature; backends enforce that
    relationship when lowering the signature.
    """

    pointer_arg_index: int = 0
    static_value: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pointer_arg_index, int)
            or isinstance(self.pointer_arg_index, bool)
            or self.pointer_arg_index < 0
        ):
            raise ValueError("pointer_arg_index must be a non-negative integer")
        if self.static_value is not None and (
            not isinstance(self.static_value, Integral)
            or isinstance(self.static_value, bool)
        ):
            raise TypeError("static pointer offset must be an integer")
        if self.static_value is not None:
            normalized = int(self.static_value)
            if not _I64_MIN <= normalized <= _I64_MAX:
                raise ValueError(
                    "static pointer offset must fit a signed 64-bit integer"
                )
            object.__setattr__(self, "static_value", normalized)

    @property
    def argument_kind(self) -> ArgumentKind:
        if self.static_value is not None:
            return ArgumentKind.STATIC
        return ArgumentKind.RUNTIME

    @property
    def role(self) -> ParameterRole:
        if self.static_value is not None:
            return ParameterRole.CONSTANT
        return ParameterRole.INPUT


@dataclass(frozen=True)
class Pointer(_RuntimeParameter):
    dtype: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None
    is_array_pointer: bool = False
    restrict: bool = False
    deref_on_call: bool = False


@dataclass(frozen=True)
class Reference(_RuntimeParameter):
    dtype: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None


@dataclass(frozen=True)
class Array(_RuntimeParameter):
    dtype: Any
    size: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None


@dataclass(frozen=True)
class TempStorageParameter(_RuntimeParameter):
    dtype: Any = UINT8
    name: str | None = "temp_storage"
    is_output: bool = False
    is_array_pointer: bool = True

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.TEMP_STORAGE


@dataclass(frozen=True)
class CxxFunction(_StaticParameter):
    cpp: str
    dtype: Any
    name: str | None = None

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.CONSTANT


@dataclass(frozen=True)
class CxxOperator(_StaticParameter):
    cpp: str
    dtype: Any
    name: str | None = None


@dataclass(frozen=True)
class PythonOperator(_StaticParameter):
    ret_dtype: Any
    arg_dtypes: tuple[Any, ...]
    op: Any
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arg_dtypes", tuple(self.arg_dtypes))


@dataclass(frozen=True)
class StatefulOperator(_RuntimeParameter):
    """Python callable whose state is passed as a runtime operand."""

    op: Any
    state_dtype: Any
    ret_dtype: Any
    arg_dtypes: tuple[Any, ...]
    name: str | None = None
    is_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arg_dtypes", tuple(self.arg_dtypes))

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.STATE


def classify_parameter(parameter: Any) -> ParameterClassification:
    """Return the static/runtime and semantic-role classification."""

    try:
        kind = parameter.argument_kind
        role = parameter.role
    except AttributeError as exc:
        raise TypeError(f"Unsupported core parameter {parameter!r}") from exc
    return ParameterClassification(getattr(parameter, "name", None), kind, role)
