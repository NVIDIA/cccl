# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe primitive arguments without importing a backend type system.

A factory uses parameter descriptors such as ``Value``, ``Pointer``, and
``Array`` to describe a C++ method's signature. These records carry types,
directions, and dependencies; they do not hold the device values passed to a
call. For example, ``Array`` describes an array parameter's type and extent,
while ``ThreadData`` constructs the per-thread payload supplied to it. An
adapter turns the records into its compiler's parameter objects.

Argument kind and parameter role answer different questions. Kind says whether
the generated code embeds an argument or receives it at runtime. Role says
whether the primitive reads it, writes it, or uses it as temporary storage.
Return-value handling is a separate adapter choice. The ``is_return`` field
controls that choice where it is available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class ArgumentKind(str, Enum):
    """Distinguish embedded method arguments from runtime operands.

    ``STATIC`` arguments appear in generated C++ source. ``RUNTIME`` arguments
    need a value when the device call executes. This classification describes
    a selected signature; ``BindingKind`` also covers omitted options while a
    factory chooses that signature.
    """

    STATIC = "static"
    RUNTIME = "runtime"


class ParameterRole(str, Enum):
    """Describe how a primitive uses an argument, apart from its C++ type.

    ``INPUT`` supplies a value, ``OUTPUT`` receives a result, and ``INOUT``
    does both. ``CONSTANT`` describes an embedded non-operator value, such as
    a static count or offset. ``TEMP_STORAGE`` identifies scratch memory that
    the implementation needs during the call. An output role does not by
    itself make the argument a Python return value.

    ``OPERATOR`` identifies a C++ functor or compiled Python callback selected
    during specialization. ``STATE`` identifies an operand that supplies a
    callback's state at runtime.
    """

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"
    CONSTANT = "constant"
    TEMP_STORAGE = "temp_storage"
    OPERATOR = "operator"
    STATE = "state"


@dataclass(frozen=True)
class ParameterClassification:
    """Keep an argument's name, supply kind, and role in a group call plan.

    Planners use this smaller record when they need argument roles without
    the full type or array extent from a parameter descriptor. ``name`` may
    be ``None`` for an unnamed parameter.
    """

    name: str | None
    kind: ArgumentKind
    role: ParameterRole


@dataclass(frozen=True)
class BuiltinDType:
    """Name a scalar type without choosing a compiler's representation.

    Adapters map ``name`` to a supported backend dtype. For example, ``INT32``
    requests a signed 32-bit integer. The record itself does not validate
    arbitrary names or choose their size and alignment.
    """

    name: str


INT8 = BuiltinDType("int8")
UINT8 = BuiltinDType("uint8")
INT16 = BuiltinDType("int16")
UINT16 = BuiltinDType("uint16")
INT32 = BuiltinDType("int32")
UINT32 = BuiltinDType("uint32")
INT64 = BuiltinDType("int64")
UINT64 = BuiltinDType("uint64")
FLOAT32 = BuiltinDType("float32")
FLOAT64 = BuiltinDType("float64")


class SubstitutionFailure(ValueError):
    """Report that a named dependency cannot supply a signature's value.

    A missing or ``None`` binding can make one candidate overload unavailable.
    A backend can reject that candidate while considering other signatures.
    """


@dataclass(frozen=True)
class TemplateParameter:
    """Name one C++ template parameter in the declaration's argument order.

    ``Algorithm`` uses ``name`` to find the bound value and to distinguish
    template arguments from extra dependencies used only by its parameters.
    """

    name: str


@dataclass(frozen=True)
class Dependency:
    """Defer a parameter's type or extent to a named algorithm binding.

    For example, ``Array(Dependency("T"), Dependency("ITEMS_PER_THREAD"))``
    describes a signature that a factory can reuse for different element
    types and array sizes. ``resolve`` looks up each name in the algorithm's
    ``template_arguments`` mapping.
    """

    name: str

    def resolve(self, template_arguments: Mapping[str, Any]) -> Any:
        """Read the named binding, rejecting a missing or ``None`` value.

        Raise ``SubstitutionFailure`` when this dependency cannot form part
        of the selected signature. Leave value-type checks to the adapter.
        """

        if self.name not in template_arguments:
            raise SubstitutionFailure(
                f"Template argument {self.name} not provided"
            )
        value = template_arguments[self.name]
        if value is None:
            raise SubstitutionFailure(f"Template argument {self.name} is None")
        return value


@dataclass(frozen=True)
class Constant:
    """Supply a fixed value through the same interface as ``Dependency``.

    A parameter can use this record for a type or extent that does not depend
    on the algorithm's bindings. Resolution returns ``value`` unchanged.
    """

    value: Any

    def resolve(self, _template_arguments: Mapping[str, Any]) -> Any:
        return self.value


@dataclass(frozen=True)
class RuntimeValue:
    """Tell a factory that a scalar comes from the eventual device call.

    Frontends use this marker in place of a compiler expression when building
    shared semantics. ``name`` is an optional label; the ``binding`` helper
    discards it and records only that a runtime value is required.
    """

    name: str | None = None


class _RuntimeParameter:
    """Share runtime classification and direction rules among descriptors.

    An inout flag takes precedence over an output flag. A descriptor with
    neither flag has the input role. Subclasses can override these rules.
    """

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
    """Mark an embedded argument that needs no runtime operand."""

    @property
    def argument_kind(self) -> ArgumentKind:
        return ArgumentKind.STATIC

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.OPERATOR


@dataclass(frozen=True)
class Value(_RuntimeParameter):
    """Describe a scalar argument and how the primitive uses it.

    ``Pointer``, ``Reference``, and ``Array`` use the same name and direction
    fields. The direction describes the operation; ``is_return`` lets an
    adapter represent an output as a result or keep it as a caller argument.

    Attributes
    ----------
    dtype : object
        Scalar type understood by the consuming adapter.
    name : str or None
        Argument name for planning and generated signatures.
    is_output : bool
        Mark a value that the primitive writes.
    is_inout : bool
        Mark a value that the primitive reads and writes. This takes
        precedence over ``is_output`` when classifying its role.
    is_return : bool or None
        Select return-value handling independently of the output role.
        ``True`` asks the adapter to expose this argument as a result.
        ``None`` leaves the choice to the adapter; ``False`` keeps an output
        as an argument.
    """

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

    ``static_value`` embeds a fixed element offset. With ``None``, the
    wrapper receives the offset at runtime. In either case, the offset
    adjusts the pointer passed to the primitive instead of adding a C++
    method argument. Its unit is elements, not bytes.

    Construction checks the index and normalizes a static offset to a Python
    integer within signed 64-bit bounds. Operation factories impose any
    further bounds, such as requiring a nonnegative offset.
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
    """Describe an address passed to a primitive or its generated wrapper.

    The common dtype, name, and direction fields follow ``Value``. Extra
    fields record information that an adapter can use when building the call.

    Attributes
    ----------
    is_array_pointer : bool
        Identify a pointer to an array payload rather than a single value.
    restrict : bool
        Record the factory's non-aliasing intent for adapters that express
        this property in generated code.
    deref_on_call : bool
        Pass the pointed-to value to the C++ method. The wrapper still
        receives the address, which lets a method reference caller storage.
    """

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
    """Describe a C++ reference parameter that can access the caller's value.

    The fields follow ``Value``. The adapter chooses how its call interface
    supplies the address needed by the generated C++ reference.
    """

    dtype: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None


@dataclass(frozen=True)
class Array(_RuntimeParameter):
    """Describe an array parameter with a count known during compilation.

    ``dtype`` gives the element type and ``size`` gives the element count.
    Either can be a ``Dependency`` to resolve from the algorithm's bindings.
    In a Load/Store signature, this count is the items per thread, not the tile
    size. The name and direction fields follow ``Value``.
    """

    dtype: Any
    size: Any
    name: str | None = None
    is_output: bool = False
    is_inout: bool = False
    is_return: bool | None = None


@dataclass(frozen=True)
class TempStorageParameter(_RuntimeParameter):
    """Mark scratch storage that the backend supplies to the primitive.

    The default dtype describes bytes. Allocation size, alignment, and
    ownership come from storage planning. An adapter can retain this marker
    as an argument or omit it when it supplies storage through another path.
    """

    dtype: Any = UINT8
    name: str | None = "temp_storage"
    is_output: bool = False
    is_array_pointer: bool = True

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.TEMP_STORAGE


@dataclass(frozen=True)
class CxxFunction(_StaticParameter):
    """Embed a C++ expression as a method argument without a runtime operand.

    ``cpp`` can name a functor or contain a scalar expression such as a fixed
    item count. ``dtype`` describes that argument to the adapter. ``name``
    identifies the parameter in the shared signature. The factory supplies
    source text here; this record does not parse or validate it.
    """

    cpp: str
    dtype: Any
    name: str | None = None

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.CONSTANT


@dataclass(frozen=True)
class CxxOperator(_StaticParameter):
    """Describe a stateless C++ functor for a generated primitive call.

    The backend constructs the functor in generated code without a runtime
    operator argument. ``CxxFunction.cpp`` instead contains the expression to
    use in the call.

    Attributes
    ----------
    cpp : str
        C++ functor type spelling, for example ``::cuda::std::plus<T>``. The
        backend resolves any dtype placeholder and constructs the object.
    dtype : object
        Operand dtype, or a ``Dependency`` resolved from the algorithm's bound
        template arguments.
    name : str, optional
        Parameter name used to identify the operator in the core signature.
    """

    cpp: str
    dtype: Any
    name: str | None = None


@dataclass(frozen=True)
class PythonOperator(_StaticParameter):
    """Describe a Python callback compiled without a runtime state operand.

    The backend compiles ``op`` for the declared input and result dtypes. The
    callable may capture values in Python closures or defaults; those values
    affect compilation identity but add no runtime state argument.

    Attributes
    ----------
    ret_dtype : object
        Callback result dtype, or a dependency resolved by the backend.
    arg_dtypes : tuple
        Callback input dtypes in call order. Construction copies the supplied
        sequence to a tuple. Entries may be dependencies.
    op : object
        Python callable or callback wrapper understood by the backend.
    name : str, optional
        Parameter name used to identify the operator in the core signature.
    op_tokenizer : callable, optional
        Backend policy that describes ``op`` without traversing compiler
        implementation state. Core identity traversal calls this policy again
        on each query so changed callback dependencies remain visible. The
        returned token replaces the ordinary token for ``op``; the policy
        function itself is excluded from identity and dataclass comparison.
        With ``None``, the shared encoder inspects ``op`` directly.
    """

    ret_dtype: Any
    arg_dtypes: tuple[Any, ...]
    op: Any
    name: str | None = None
    op_tokenizer: Callable[[Any], Any] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "arg_dtypes", tuple(self.arg_dtypes))


@dataclass(frozen=True)
class StatefulOperator(_RuntimeParameter):
    """Describe a Python callback with state supplied as a runtime operand.

    The callback and its type signature are selected during specialization.
    The primitive caller supplies the state value at runtime. This record
    describes the requirement; each backend decides whether it can implement
    stateful callbacks. Its role stays ``STATE`` even for output state.

    Attributes
    ----------
    op : object
        Python callable or callback wrapper understood by the backend.
    state_dtype : object
        Dtype of the runtime state supplied to the callback.
    ret_dtype : object
        Callback result dtype, or a dependency resolved by the backend.
    arg_dtypes : tuple
        Callback input dtypes in call order, separate from ``state_dtype``.
        Construction copies the supplied sequence to a tuple.
    name : str, optional
        Name used to identify the state parameter in the core signature.
    is_output : bool
        Whether the descriptor requests output state. Backend lowering defines
        how that state is returned to the caller.
    op_tokenizer : callable, optional
        Backend callback-identity policy, with the same contract as
        ``PythonOperator.op_tokenizer``. The state value is a runtime operand
        and is not stored in this descriptor.
    """

    op: Any
    state_dtype: Any
    ret_dtype: Any
    arg_dtypes: tuple[Any, ...]
    name: str | None = None
    is_output: bool = False
    op_tokenizer: Callable[[Any], Any] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "arg_dtypes", tuple(self.arg_dtypes))

    @property
    def role(self) -> ParameterRole:
        return ParameterRole.STATE


def classify_parameter(parameter: Any) -> ParameterClassification:
    """Extract the argument facts needed by a group call plan.

    Read ``argument_kind`` and ``role`` from the descriptor, with an optional
    ``name``. Raise ``TypeError`` if either required attribute is absent.
    This preserves the descriptor's classification without resolving its
    dtype or other dependencies.
    """

    try:
        kind = parameter.argument_kind
        role = parameter.role
    except AttributeError as exc:
        raise TypeError(f"Unsupported core parameter {parameter!r}") from exc
    return ParameterClassification(getattr(parameter, "name", None), kind, role)
