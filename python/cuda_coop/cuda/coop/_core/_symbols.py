# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe specialization values and callback dependencies for cache keys.

An operator's name alone does not identify the code to compile. Its defaults,
captured values, and referenced helpers can change the result. Describe those
dependencies as nested tokens so algorithm comparison and callback symbols
can distinguish them. Backend normalization supplies identities for compiler
objects without making the shared core depend on a compiler's internals.

The value encoder also preserves distinctions that Python equality can lose,
such as ``True`` versus ``1`` and the sign of floating-point zero.
Cycle markers use distances within the traversal instead of object addresses.
These tokens describe supported Python inputs; they are not a serialization
format or a proof that arbitrary Python programs behave identically.
"""

from __future__ import annotations

import dataclasses
import dis
import hashlib
import inspect
import math
import os
import re
import struct
import sys
import sysconfig
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from functools import cache, partial
from types import CodeType, ModuleType
from typing import Any

from ._types import PythonOperator, StatefulOperator

_ADDRESS_IN_REPR = re.compile(r"(?<= at )0x[0-9a-fA-F]+")
_MISSING = object()
_PY_TPFLAGS_HEAPTYPE = 1 << 9
_TYPE_METADATA_NAMES = frozenset(
    {
        "__dict__",
        "__classcell__",
        "__doc__",
        "__firstlineno__",
        "__module__",
        "__qualname__",
        "__static_attributes__",
        "__weakref__",
    }
)


@cache
def _python_library_paths() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Locate standard-library roots separately from installed-package roots.

    Some Python layouts put site-packages inside a standard-library directory.
    Callers must exclude those package roots before using a module's location
    to justify a compact name-based token.
    """

    paths = sysconfig.get_paths()

    def normalized(*names: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                os.path.realpath(path)
                for name in names
                if isinstance((path := paths.get(name)), str) and path
            )
        )

    return normalized("stdlib", "platstdlib"), normalized("purelib", "platlib")


def _path_is_within(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def _is_verified_standard_library_module(module_name: Any) -> bool:
    """Check a loaded module's origin before using a standard-library name.

    A familiar name alone is insufficient: a user module can shadow it. Accept
    recognized built-in or frozen modules, or source under the interpreter's
    standard-library roots and outside its installed-package roots.
    """

    if not isinstance(module_name, str) or not module_name:
        return False

    module_root = module_name.partition(".")[0]
    stdlib_module_names = getattr(sys, "stdlib_module_names", frozenset())
    if (
        module_root not in stdlib_module_names
        and module_root not in sys.builtin_module_names
    ):
        return False

    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        return False

    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin in {"built-in", "frozen"}:
        return True

    source_path = origin if isinstance(origin, str) else None
    if not source_path or source_path in {"namespace", "unknown"}:
        source_path = getattr(module, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        return False

    source_path = os.path.realpath(source_path)
    stdlib_paths, package_paths = _python_library_paths()
    return any(
        _path_is_within(source_path, path) for path in stdlib_paths
    ) and not any(_path_is_within(source_path, path) for path in package_paths)


def _is_verified_standard_library_definition(value: Any) -> bool:
    """Check that a qualified name resolves to this library object.

    The module's origin check alone does not identify an object that claims
    its name. Resolve the path without invoking descriptors and require the
    resolved object to be ``value``.
    """

    module_name = _defined_module_name(value)
    if not _is_verified_standard_library_module(module_name):
        return False
    qualified_name = getattr(value, "__qualname__", None)
    if not isinstance(qualified_name, str) or "<locals>" in qualified_name:
        return False
    resolved: Any = sys.modules[module_name]
    for name in qualified_name.split("."):
        try:
            resolved = inspect.getattr_static(resolved, name)
        except AttributeError:
            return False
    return resolved is value


def _is_closure_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _defined_module_name(value: Any) -> str | None:
    """Use a module name only when the object supplies it as text."""

    module_name = getattr(value, "__module__", None)
    return module_name if isinstance(module_name, str) else None


@dataclasses.dataclass
class _TokenState:
    """Track recursion and reuse completed work within one token query.

    ``normalize`` lets a backend replace compiler objects with supported data.
    ``active`` maps object identities to their depth on the current path. This
    lets recursive helpers and self-referencing objects end in a cycle marker.
    ``completed`` keeps reusable tokens by traversal mode and object identity.
    Each entry retains its source object so Python cannot reuse that identity
    during this query. The modes keep ordinary values separate from callable
    dependencies, whose class definitions receive more detailed inspection.
    ``cycle_hits`` counts back-references. Store a result for reuse only if
    its own traversal found none, because cycle depths depend on the path.
    """

    normalize: Callable[[Any], Any] | None = None
    active: dict[int, int] = dataclasses.field(default_factory=dict)
    completed: dict[tuple[str, int], tuple[Any, Any]] = dataclasses.field(
        default_factory=dict
    )
    cycle_hits: int = 0


def _code_token(code: CodeType, state: _TokenState) -> tuple[Any, ...]:
    """Describe bytecode and nested constants without source-location data.

    Names, argument counts, and flags contribute to callback identity. Source
    filenames and line numbers do not, so moving otherwise identical code does
    not change this component. Nested code objects are described by contents
    rather than by a representation that includes an address.
    """

    constants = tuple(
        _code_token(value, state)
        if isinstance(value, CodeType)
        else _semantic_token(value, state)
        for value in code.co_consts
    )
    return (
        code.co_code.hex(),
        constants,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )


def _object_state_token(
    value: Any,
    state: _TokenState,
    *,
    dependency_values: bool = False,
) -> tuple[Any, ...] | None:
    """Describe stored instance attributes, including inherited private slots.

    Python mangles private slot names using the defining class. Resolve those
    names before reading values, and skip slots that have not been assigned.
    Return ``None`` when these attributes expose no stored state.

    With ``dependency_values=True``, inspect stored classes and descriptors as
    callable dependencies. Their definitions can affect a callback even when
    their names stay the same.
    """

    tokenize = _dependency_token if dependency_values else _semantic_token
    object_state: list[tuple[str, Any]] = []
    attributes = getattr(value, "__dict__", None)
    if attributes:
        object_state.append(("__dict__", tokenize(attributes, state)))

    seen_slots: set[str] = set()
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            storage_name = slot
            if slot.startswith("__") and not slot.endswith("__"):
                class_name = cls.__name__.lstrip("_")
                if class_name:
                    storage_name = f"_{class_name}{slot}"
            if storage_name in {"__dict__", "__weakref__"} or (
                storage_name in seen_slots
            ):
                continue
            seen_slots.add(storage_name)
            try:
                slot_value = getattr(value, storage_name)
            except AttributeError:
                continue
            token = (
                ("self",)
                if slot_value is value
                else tokenize(slot_value, state)
            )
            object_state.append((storage_name, token))

    return tuple(object_state) if object_state else None


def _referenced_paths_token(
    code: CodeType,
    namespace: Mapping[str, Any],
    state: _TokenState,
    *,
    load_opnames: frozenset[str],
    include_nested: bool,
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    """Describe names and immediate attribute paths loaded by the bytecode.

    Callback identity construction calls this for global and captured
    dependencies. For example, a callback using ``helpers.combine`` must
    change identity when that referenced helper changes, even if the
    callback bytecode and module name stay the same.

    ``load_opnames`` selects global-name or closure-cell loads. For each load,
    record the root name and each following attribute or method prefix. Read
    those paths from ``namespace`` with static attribute lookup so descriptors
    are inspected without calling their accessors. Missing roots are skipped;
    missing attributes receive an explicit unresolved marker.

    ``include_nested`` also visits code objects in the code's constants.
    This scan follows explicit bytecode paths, not computed lookups. It avoids
    including unrelated namespace entries while retaining the loaded root
    objects as well as their attributes.

    Parameters
    ----------
    code : types.CodeType
        Callback implementation whose explicit global or closure
        references are being inspected.
    namespace : mapping of str to object
        Values bound to the selected root names: globals for global
        loads, or named closure contents for closure loads.
    state : _TokenState
        Mutable traversal state shared with the caller: backend
        normalization, active recursion paths, and reusable
        completed tokens. It belongs to one semantic-token query.
    load_opnames : frozenset of str
        Bytecode load instructions that introduce the root names to
        inspect.
    include_nested : bool
        Whether to inspect code objects stored in constants, such as
        nested helper functions, using the same namespace.
    """

    paths: set[tuple[str, ...]] = set()

    def collect_paths(current: CodeType) -> None:
        instructions = tuple(dis.get_instructions(current))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in load_opnames or not isinstance(
                instruction.argval, str
            ):
                continue
            path = [instruction.argval]
            paths.add(tuple(path))
            for attribute in instructions[index + 1 :]:
                if attribute.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                    break
                if not isinstance(attribute.argval, str):
                    break
                path.append(attribute.argval)
                paths.add(tuple(path))
        if include_nested:
            for constant in current.co_consts:
                if isinstance(constant, CodeType):
                    collect_paths(constant)

    collect_paths(code)
    dependencies: list[tuple[tuple[str, ...], Any]] = []
    for path in sorted(paths):
        if path[0] not in namespace:
            continue
        value = namespace[path[0]]
        resolved = True
        for attribute in path[1:]:
            try:
                value = inspect.getattr_static(value, attribute)
            except AttributeError:
                resolved = False
                break
        token = (
            _dependency_token(value, state)
            if resolved
            else ("unresolved-attribute", path)
        )
        dependencies.append((path, token))
    return tuple(dependencies)


def _referenced_globals_token(
    code: CodeType,
    namespace: Mapping[str, Any] | None,
    state: _TokenState,
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    """Inspect global dependencies loaded by this code and its nested code.

    An unavailable globals mapping contributes no paths. Otherwise, inspect
    names loaded by the bytecode instead of every namespace entry.
    """

    if namespace is None:
        return ()
    return _referenced_paths_token(
        code,
        namespace,
        state,
        load_opnames=frozenset(
            {"LOAD_FROM_DICT_OR_GLOBALS", "LOAD_GLOBAL", "LOAD_NAME"}
        ),
        include_nested=True,
    )


def _referenced_closure_token(
    code: CodeType,
    closure_values: Mapping[str, Any],
    state: _TokenState,
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    """Inspect attribute paths reached through named closure cells.

    The caller already records each cell's value. These paths also expose
    dependencies behind values such as a captured module, whose ordinary token
    contains only its name.
    """

    return _referenced_paths_token(
        code,
        closure_values,
        state,
        load_opnames=frozenset({"LOAD_DEREF"}),
        include_nested=True,
    )


def _type_reference_token(value: type, state: _TokenState) -> Any:
    """Use library type names or inspect a user-defined class dependency.

    Verified standard-library definitions use a compact name token. Static
    extension types from verified standard-library modules use the same path.
    Other classes include their definitions so changed methods or class state
    can distinguish callbacks that refer to the same class name.
    """

    module_name = _defined_module_name(value)
    type_flags = getattr(value, "__flags__", None)
    is_static_type = (
        isinstance(type_flags, int) and not type_flags & _PY_TPFLAGS_HEAPTYPE
    )
    if _is_verified_standard_library_definition(value) or (
        is_static_type and _is_verified_standard_library_module(module_name)
    ):
        return (
            "type",
            module_name,
            getattr(value, "__qualname__", value.__name__),
        )
    return _type_dependency_token(value, state)


def _descriptor_token(value: Any, state: _TokenState) -> Any:
    """Describe attribute access code without invoking the descriptor.

    Static methods, class methods, and properties contribute their underlying
    functions. A custom descriptor contributes its type definition and stored
    state, since either can change what a callback obtains from an attribute.
    """

    if isinstance(value, staticmethod):
        return "staticmethod", _semantic_token(value.__func__, state)
    if isinstance(value, classmethod):
        return "classmethod", _semantic_token(value.__func__, state)
    if isinstance(value, property):
        return (
            "property",
            _semantic_token(value.fget, state),
            _semantic_token(value.fset, state),
            _semantic_token(value.fdel, state),
        )
    descriptor_type = type(value)
    if descriptor_type.__module__ != "builtins" and any(
        name in base.__dict__
        for base in descriptor_type.__mro__
        for name in ("__get__", "__set__", "__delete__")
    ):
        return (
            "descriptor",
            _type_reference_token(descriptor_type, state),
            _semantic_token(value, state),
        )
    return _semantic_token(value, state)


def _type_dependency_token(value: type, state: _TokenState) -> Any:
    """Describe a class's members, bases, and metaclass for a callback.

    Dependency traversal calls this when naming a class alone would hide
    changes to code or state used by a callback. It contributes to
    specialization identity; it does not instantiate the class or
    compile its methods.

    Definitions can refer back to their class or to other active dependencies.
    Use path-relative cycle markers for those references. Reuse completed work
    only when it contains no cycle marker; its depth would have a different
    meaning if reused from another traversal path.

    Parameters
    ----------
    value : type
        Class reached through a callback dependency, including its
        own bases and metaclass.
    state : _TokenState
        Mutable traversal state shared with the caller: backend
        normalization, active recursion paths, and reusable
        completed tokens. It belongs to one semantic-token query.
    """

    value_id = id(value)
    if value_id in state.active:
        state.cycle_hits += 1
        return _cycle_token(value, len(state.active) - state.active[value_id])

    memo_key = ("type-dependency", value_id)
    completed = state.completed.get(memo_key)
    if completed is not None:
        return completed[1]

    state.active[value_id] = len(state.active)
    cycle_hits_before = state.cycle_hits
    try:
        members = tuple(
            (name, _dependency_token(member, state))
            for name, member in sorted(vars(value).items())
            if name not in _TYPE_METADATA_NAMES
        )
        bases = tuple(
            _type_reference_token(base, state) for base in value.__bases__
        )
        metaclass = type(value)
        metaclass_token = _type_reference_token(metaclass, state)
        token = (
            "type-dependency",
            _defined_module_name(value),
            getattr(value, "__qualname__", value.__name__),
            metaclass_token,
            bases,
            members,
        )
    finally:
        del state.active[value_id]

    if state.cycle_hits == cycle_hits_before:
        state.completed[memo_key] = (value, token)
    return token


def _dataclass_fields_token(value, state, tokenize):
    """Describe record fields while honoring an operator's identity policy.

    ``op_tokenizer`` supplies the operator's ``op`` token. Exclude the policy
    function itself: it describes the callback but is not a dependency. All
    other fields use the caller's encoder. Evaluate the policy during each
    new traversal so captured changes remain visible when core planning
    compares an operator again.
    """

    operator = isinstance(value, (PythonOperator, StatefulOperator))
    fields = []
    for field in dataclasses.fields(value):
        if operator and field.name == "op_tokenizer":
            continue
        item = getattr(value, field.name)
        if operator and field.name == "op" and value.op_tokenizer is not None:
            # Backend policy travels with the operator through core planning.
            # Evaluate it here to observe changed callback dependencies.
            token = value.op_tokenizer(item)
        else:
            token = tokenize(item, state)
        fields.append((field.name, token))
    return tuple(fields)


def _dependency_token(value: Any, state: _TokenState) -> Any:
    """Describe a callable dependency, including nested class definitions.

    The callback encoder uses this traversal so a captured container of
    helper classes retains their definitions. Ordinary specialization
    values can use a narrower description; that distinction is why
    dependency traversal keeps a separate memoization mode.

    The ordinary value encoder names a type without inspecting its definition.
    Callbacks need more detail: methods, class attributes, and descriptors can
    change their behavior. Keep this dependency mode through containers and
    dataclass fields, with a separate memo from ordinary value tokens.

    Apply backend normalization first. It runs before cycle tracking, so
    replacements must stop changing or the recursion will not end.

    Parameters
    ----------
    value : object
        Value reached through callback globals, closure contents,
        defaults, stored state, or another dependency container.
    state : _TokenState
        Mutable traversal state shared with the caller: backend
        normalization, active recursion paths, and reusable
        completed tokens. It belongs to one semantic-token query.
    """

    if state.normalize is not None:
        normalized = state.normalize(value)
        if normalized is not value:
            return _dependency_token(normalized, state)
    if isinstance(value, type):
        return _type_reference_token(value, state)
    if not (
        isinstance(value, (Mapping, tuple, list, set, frozenset))
        or (dataclasses.is_dataclass(value) and not isinstance(value, type))
    ):
        return _descriptor_token(value, state)

    value_id = id(value)
    if value_id in state.active:
        state.cycle_hits += 1
        return _cycle_token(value, len(state.active) - state.active[value_id])

    memo_key = ("dependency-value", value_id)
    completed = state.completed.get(memo_key)
    if completed is not None:
        return completed[1]

    state.active[value_id] = len(state.active)
    cycle_hits_before = state.cycle_hits
    try:
        if isinstance(value, Mapping):
            items_snapshot = tuple(value.items())
            items = [
                (_dependency_token(key, state), _dependency_token(item, state))
                for key, item in items_snapshot
            ]
            token = (
                "mapping",
                _container_kind(value),
                _container_state_token(
                    value,
                    state,
                    dependency_values=True,
                ),
                tuple(sorted(items, key=lambda item: repr(item[0]))),
            )
        elif isinstance(value, (tuple, list)):
            token = (
                "sequence",
                _container_kind(value),
                _container_state_token(
                    value,
                    state,
                    dependency_values=True,
                ),
                tuple(_dependency_token(item, state) for item in value),
            )
        elif isinstance(value, (set, frozenset)):
            token = (
                "set",
                _container_kind(value),
                _container_state_token(
                    value,
                    state,
                    dependency_values=True,
                ),
                tuple(
                    sorted(
                        (_dependency_token(item, state) for item in value),
                        key=repr,
                    )
                ),
            )
        else:
            token = (
                type(value).__module__,
                type(value).__qualname__,
                _dataclass_fields_token(value, state, _dependency_token),
            )
    finally:
        del state.active[value_id]

    if state.cycle_hits == cycle_hits_before:
        state.completed[memo_key] = (value, token)
    return token


def _callable_token(value: Any, state: _TokenState) -> tuple[Any, ...]:
    """Identify a callable from its implementation and captured dependencies.

    ``_semantic_token`` calls this when it encounters a callable while
    building an algorithm or operator identity. The description must
    notice changes to captured inputs before a compiled callback or
    provider can be reused.

    Functions contribute bytecode, defaults, closure contents, and referenced
    globals. Bound methods also include the instance's stored state. Callable
    objects use their exposed function metadata or their ``__call__`` method;
    when they expose code directly, also include their actual call descriptor.
    This keeps wrapper metadata from hiding a different call implementation.

    Partial applications include their bound arguments. Verified Python
    standard-library functions use a qualified reference instead of traversing
    library internals. Other callable tokens include a digest of the inspected
    dependencies. Unexpected metadata is included as data rather than silently
    discarded, so it can still distinguish callable objects.

    Parameters
    ----------
    value : callable
        Function, bound method, partial application, or callable
        instance whose behavior contributes to a specialization.
    state : _TokenState
        Mutable traversal state shared with the caller: backend
        normalization, active recursion paths, and reusable
        completed tokens. It belongs to one semantic-token query.
    """

    if isinstance(value, partial):
        return (
            "partial",
            _semantic_token(value.func, state),
            _dependency_token(value.args, state),
            _dependency_token(value.keywords, state),
        )

    function_like = inspect.isfunction(value) or inspect.ismethod(value)
    module_name = _defined_module_name(value)
    qualified_name = getattr(value, "__qualname__", type(value).__qualname__)
    if inspect.isfunction(value) and _is_verified_standard_library_definition(
        value
    ):
        return "callable-reference", module_name, qualified_name

    callable_state = _object_state_token(value, state, dependency_values=True)
    malformed_attributes: list[tuple[str, Any]] = []
    implementation = value
    exposed_code = getattr(implementation, "__code__", _MISSING)
    exposed_namespace = getattr(implementation, "__globals__", _MISSING)
    exposed_closure = getattr(implementation, "__closure__", _MISSING)
    if exposed_code is not _MISSING and not isinstance(exposed_code, CodeType):
        malformed_attributes.append(
            ("__code__", _dependency_token(exposed_code, state))
        )
    code = exposed_code if isinstance(exposed_code, CodeType) else None
    exposed_object_code = code is not None and not function_like
    object_call = _MISSING
    if exposed_object_code:
        try:
            object_call = inspect.getattr_static(type(value), "__call__")
        except AttributeError:
            pass
    if code is None:
        if exposed_namespace is not _MISSING and not isinstance(
            exposed_namespace, Mapping
        ):
            malformed_attributes.append(
                ("__globals__", _dependency_token(exposed_namespace, state))
            )
        if (
            exposed_closure is not _MISSING
            and exposed_closure is not None
            and not _is_closure_sequence(exposed_closure)
        ):
            malformed_attributes.append(
                ("__closure__", _dependency_token(exposed_closure, state))
            )
        # Inspect the implementation itself, rather than test callability.
        implementation = getattr(value, "__call__", _MISSING)  # noqa: B004
        implementation_code = getattr(implementation, "__code__", _MISSING)
        if implementation_code is not _MISSING and not isinstance(
            implementation_code, CodeType
        ):
            malformed_attributes.append(
                (
                    "__call__.__code__",
                    _dependency_token(implementation_code, state),
                )
            )
        code = (
            implementation_code
            if isinstance(implementation_code, CodeType)
            else None
        )
        namespace_value = getattr(implementation, "__globals__", _MISSING)
        closure_value = getattr(implementation, "__closure__", _MISSING)
        attribute_prefix = "__call__."
    else:
        namespace_value = exposed_namespace
        closure_value = exposed_closure
        attribute_prefix = ""

    if isinstance(namespace_value, Mapping):
        namespace: Mapping[str, Any] | None = namespace_value
    else:
        namespace = None
        if namespace_value is not _MISSING:
            malformed_attributes.append(
                (
                    f"{attribute_prefix}__globals__",
                    _dependency_token(namespace_value, state),
                )
            )

    digest = hashlib.sha256()
    if code is not None:
        digest.update(
            repr(_code_token(code, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
        digest.update(
            repr(_referenced_globals_token(code, namespace, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
    digest.update(
        repr(
            _dependency_token(
                getattr(implementation, "__defaults__", None), state
            )
        ).encode("utf-8", errors="backslashreplace")
    )
    digest.update(
        repr(
            _dependency_token(
                getattr(implementation, "__kwdefaults__", None), state
            )
        ).encode("utf-8", errors="backslashreplace")
    )
    if closure_value is None or closure_value is _MISSING:
        closure: Sequence[Any] = ()
    elif _is_closure_sequence(closure_value):
        closure = closure_value
    else:
        closure = ()
        malformed_attributes.append(
            (
                f"{attribute_prefix}__closure__",
                _dependency_token(closure_value, state),
            )
        )
    closure_values: dict[str, Any] = {}
    freevar_names = getattr(code, "co_freevars", ())
    # Wrappers can expose mismatched code and closure metadata. Hash all cells
    # and resolve attribute paths for the name/value pairs that are available.
    for index, cell in enumerate(closure):
        name = freevar_names[index] if index < len(freevar_names) else None
        try:
            cell_value = cell.cell_contents
        except ValueError:
            cell_value = "<empty>"
        except AttributeError:
            cell_value = cell
        else:
            if name is not None:
                closure_values[name] = cell_value
        digest.update(
            repr(_dependency_token(cell_value, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
    if code is not None:
        digest.update(
            repr(_referenced_closure_token(code, closure_values, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
    bound_self = getattr(value, "__self__", None)
    if bound_self is not None:
        if isinstance(bound_self, ModuleType):
            bound_self_token = ("module", bound_self.__name__)
        else:
            object_state = _object_state_token(
                bound_self,
                state,
                dependency_values=True,
            )
            bound_self_token = (
                type(bound_self).__module__,
                type(bound_self).__qualname__,
                object_state,
            )
        digest.update(
            repr(bound_self_token).encode("utf-8", errors="backslashreplace")
        )
    if callable_state is not None:
        digest.update(
            repr(callable_state).encode("utf-8", errors="backslashreplace")
        )
    if object_call is not _MISSING:
        digest.update(
            repr(_descriptor_token(object_call, state)).encode(
                "utf-8", errors="backslashreplace"
            )
        )
    if malformed_attributes:
        digest.update(
            repr(tuple(malformed_attributes)).encode(
                "utf-8", errors="backslashreplace"
            )
        )

    return (
        "callable",
        module_name,
        qualified_name,
        digest.hexdigest()[:20],
    )


def _cycle_token(
    value: Any, back_reference_depth: int
) -> tuple[str, str, str, int]:
    """Identify a cycle by object kind and distance to its active ancestor."""

    return (
        "cycle",
        getattr(value, "__module__", type(value).__module__),
        getattr(value, "__qualname__", type(value).__qualname__),
        back_reference_depth,
    )


def _container_kind(value: Any) -> tuple[str | None, str]:
    """Keep a container subclass distinct from its built-in counterpart."""

    return _defined_module_name(type(value)), type(value).__qualname__


def _container_state_token(
    value: Any,
    state: _TokenState,
    *,
    dependency_values: bool = False,
) -> tuple[tuple[str, Any], ...] | None:
    """Capture container state that equal elements alone cannot describe.

    This includes instance attributes and a ``defaultdict``'s factory, which
    determines the value supplied for a missing key. ``dependency_values``
    keeps class and descriptor inspection active in callback dependencies.
    """

    tokenize = _dependency_token if dependency_values else _semantic_token
    container_state = list(
        _object_state_token(
            value,
            state,
            dependency_values=dependency_values,
        )
        or ()
    )
    if isinstance(value, defaultdict):
        default_factory = value.default_factory
        default_factory_token = (
            ("self",)
            if default_factory is value
            else tokenize(default_factory, state)
        )
        container_state.append(("default_factory", default_factory_token))
    return tuple(container_state) if container_state else None


def _semantic_token(value: Any, state: _TokenState) -> Any:
    """Describe a value while sharing cycle detection and memoized results.

    Scalar cases preserve type-sensitive distinctions before recursion starts.
    Container tokens include their type, stored state, and elements. Sort
    mapping entries by encoded key and set elements by encoded value, using
    their ``repr`` strings as sort keys. Sequences retain element order.
    Callables use the dependency-aware encoder; ordinary type or module
    values contribute only their qualified name or module name.
    """

    if state.normalize is not None:
        normalized = state.normalize(value)
        if normalized is not value:
            return _semantic_token(normalized, state)
    if isinstance(value, Enum):
        return type(value).__module__, type(value).__qualname__, value.value
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, float):
        if math.isnan(value):
            return "float", "nan", struct.pack(">d", value).hex()
        return "float", value.hex()
    if value is None or isinstance(value, (int, str, bytes)):
        return value
    if isinstance(value, ModuleType):
        return "module", value.__name__
    if isinstance(value, type):
        return "type", _defined_module_name(value), value.__qualname__

    value_id = id(value)
    if value_id in state.active:
        state.cycle_hits += 1
        return _cycle_token(value, len(state.active) - state.active[value_id])

    memo_key = ("value", value_id)
    completed = state.completed.get(memo_key)
    if completed is not None:
        return completed[1]

    state.active[value_id] = len(state.active)
    cycle_hits_before = state.cycle_hits
    try:
        if isinstance(value, Mapping):
            items_snapshot = tuple(value.items())
            items = [
                (_semantic_token(key, state), _semantic_token(item, state))
                for key, item in items_snapshot
            ]
            token = (
                "mapping",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(sorted(items, key=lambda item: repr(item[0]))),
            )
        elif isinstance(value, (tuple, list)):
            token = (
                "sequence",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(_semantic_token(item, state) for item in value),
            )
        elif isinstance(value, (set, frozenset)):
            token = (
                "set",
                _container_kind(value),
                _container_state_token(value, state),
                tuple(
                    sorted(
                        (_semantic_token(item, state) for item in value),
                        key=repr,
                    )
                ),
            )
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            token = (
                type(value).__module__,
                type(value).__qualname__,
                _dataclass_fields_token(value, state, _semantic_token),
            )
        elif callable(value):
            token = _callable_token(value, state)
        else:
            attributes = _object_state_token(value, state)
            if attributes is not None:
                token = (
                    type(value).__module__,
                    type(value).__qualname__,
                    attributes,
                )
            else:
                stable_repr = _ADDRESS_IN_REPR.sub("0x?", repr(value))
                token = (
                    type(value).__module__,
                    type(value).__qualname__,
                    stable_repr,
                )
    finally:
        del state.active[value_id]

    if state.cycle_hits == cycle_hits_before:
        # A cycle token contains path-relative depth, so reuse only results
        # that did not encounter a back-reference while visiting this value.
        state.completed[memo_key] = (value, token)
    return token


def semantic_token(
    value: Any, *, normalize: Callable[[Any], Any] | None = None
) -> Any:
    """Describe a specialization value for comparison and hashing.

    Algorithms use this token to compare bound values and planning metadata.
    Callback identities also include code and captured dependencies. Each call
    starts a fresh traversal so previous queries cannot supply stale tokens.
    Floating-point tokens preserve zero signs and NaN bits.

    Parameters
    ----------
    value : object
        Scalar, container, dataclass, type, module, callable, or state-bearing
        object used in a specialization. Other objects with no stored state
        use their type and ``repr``, with usual memory-address text removed.
    normalize : callable, optional
        Backend conversion applied to visited values, including referenced
        dependencies. Return the original object when no conversion is needed.
        A replacement is visited again and must eventually stop changing by
        identity. Describe compiler objects by their compilation inputs,
        without including compiler bookkeeping.

    Returns
    -------
    object
        A scalar or nested tuple description. Supported specialization values
        produce hashable tokens. Callable descriptions contain digests; the
        result as a whole is a token rather than a single hash digest.

    Notes
    -----
    Callable inspection follows stored state and explicit bytecode references.
    It cannot establish the behavior of arbitrary computed lookups or external
    state. A backend must normalize values whose relevant contents the shared
    encoder cannot describe, such as compiler-specific callback wrappers.
    """

    return _semantic_token(value, _TokenState(normalize=normalize))
