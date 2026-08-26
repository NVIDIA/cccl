# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import enum
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from types import ModuleType

import pytest

from ...support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

_REFERENCED_SEMANTIC_GLOBAL = 1
_UNRELATED_SEMANTIC_GLOBAL = 1


def _global_dependent_operator(left, right):
    return left + right + _REFERENCED_SEMANTIC_GLOBAL


from cuda.coop._core import (
    INT32,
    INT64,
    UINT8,
    Algorithm,
    ArgumentKind,
    Array,
    CoreBackendAdapter,
    CxxFunction,
    Dependency,
    ParameterRole,
    PointerOffset,
    PythonOperator,
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
    TypeDefinition,
    Value,
    semantic_token,
)
from cuda.coop._core._symbols import _TokenState, _type_reference_token

_TOKEN_GLOBAL_OFFSET = 1
offset = 100


class _TokenGlobalHolder:
    offset = 7


_TOKEN_GLOBAL_HOLDER = _TokenGlobalHolder()
_TOKEN_GLOBAL_MODULE = ModuleType("_cuda_coop_token_settings")
_TOKEN_GLOBAL_MODULE.offset = 11


class _TokenGlobalPolicy:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value + 1


class _TokenGlobalStaticPolicy:
    @staticmethod
    def apply(value):
        return value + 1


def _token_global_value_op(value):
    return value + _TOKEN_GLOBAL_OFFSET


def _token_attribute_op(value):
    return value + _TOKEN_GLOBAL_HOLDER.offset


def _token_module_attribute_op(value):
    return value + _TOKEN_GLOBAL_MODULE.offset


def _token_global_policy_op(value):
    return _TokenGlobalPolicy(value).result()


def _token_global_static_policy_op(value):
    return _TokenGlobalStaticPolicy.apply(value)


def _token_helper_add(value):
    return value + 1


def _token_helper_subtract(value):
    return value - 1


_TOKEN_GLOBAL_HELPER = _token_helper_add


def _token_global_helper_op(value):
    return _TOKEN_GLOBAL_HELPER(value)


def _token_recursive_op(value):
    if value == 0:
        return 0
    return _token_recursive_op(value - 1)


@pytest.mark.parametrize("pointer_arg_index", [-1, True, "zero"])
def test_pointer_offset_requires_non_negative_integer_index(pointer_arg_index):
    with pytest.raises(ValueError, match="non-negative integer"):
        PointerOffset(INT64, pointer_arg_index=pointer_arg_index)


def test_static_pointer_offset_is_not_a_runtime_abi_parameter():
    offset = PointerOffset(
        INT64,
        name="offset",
        pointer_arg_index=0,
        static_value=4,
    )

    assert offset.argument_kind is ArgumentKind.STATIC
    assert offset.role is ParameterRole.CONSTANT
    assert offset.static_value == 4


@pytest.mark.parametrize("static_value", [-(1 << 63), (1 << 63) - 1])
def test_static_pointer_offset_accepts_signed_i64_boundaries(static_value):
    offset = PointerOffset(INT64, static_value=static_value)

    assert offset.static_value == static_value


@pytest.mark.parametrize("static_value", [-(1 << 63) - 1, 1 << 63])
def test_static_pointer_offset_rejects_signed_i64_overflow(static_value):
    with pytest.raises(ValueError, match="fit a signed 64-bit integer"):
        PointerOffset(INT64, static_value=static_value)


def test_core_import_does_not_load_backend_runtimes():
    script = textwrap.dedent(
        """
        import sys

        before = set(sys.modules)
        import cuda.coop._core
        loaded = set(sys.modules) - before

        assert not any(
            name == "numba" or name.startswith("numba.") for name in loaded
        ), loaded
        assert not any(
            name == "cutlass" or name.startswith("cutlass.") for name in loaded
        ), loaded
        assert not any(name.startswith("numba_cuda_mlir") for name in loaded), loaded
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_algorithm_specialization_records_semantics_and_symbol_inputs():
    algorithm = Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=("cub/block/example.cuh",),
        template_parameters=(TemplateParameter("T"), TemplateParameter("N")),
        parameters=(
            (
                TempStorageParameter(),
                Array(Dependency("T"), Dependency("N"), name="values"),
                CxxFunction("4", INT32, name="count"),
            ),
        ),
        type_definitions=(TypeDefinition("BlockExample", "struct BlockExample {};"),),
    )
    specialization = algorithm.specialize(
        {"T": "int", "N": 4},
        metadata={"scope": "block", "primitive": "example"},
    )

    assert specialization.template_arguments == {"T": "int", "N": 4}
    assert specialization.ordered_template_arguments == (("T", "int"), ("N", 4))
    assert specialization.symbol_mangling_inputs == (
        "block_example",
        "Run",
        specialization.semantic_key,
    )
    equivalent = algorithm.specialize(
        {"T": "int", "N": 4},
        metadata={"scope": "block", "primitive": "example"},
    )
    assert specialization == equivalent
    assert hash(specialization) == hash(equivalent)

    with pytest.raises(TypeError):
        specialization.template_arguments["N"] = 8


def test_algorithm_symbol_inputs_cover_complete_specialization_identity():
    algorithm = Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=(),
        template_parameters=(TemplateParameter("T"),),
        parameters=((Value(Dependency("T"), name="value"),),),
    )
    base = algorithm.specialize({"T": "int"}, metadata={"mode": "base"})
    changed_metadata = algorithm.specialize(
        {"T": "int"}, metadata={"mode": "alternate"}
    )
    changed_parameters = Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=(),
        template_parameters=(TemplateParameter("T"),),
        parameters=((Value(Dependency("T"), name="alternate"),),),
    ).specialize({"T": "int"}, metadata={"mode": "base"})

    assert base.symbol_mangling_inputs != changed_metadata.symbol_mangling_inputs
    assert base.symbol_mangling_inputs != changed_parameters.symbol_mangling_inputs


def test_algorithm_specialization_freezes_nested_semantic_containers():
    algorithm = Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=(),
        template_parameters=(TemplateParameter("T"),),
        parameters=((),),
    )
    nested = {"values": [1], "modes": {"direct"}}
    specialization = algorithm.specialize({"T": "int", "settings": nested})
    semantic_key = specialization.semantic_key
    symbol_inputs = specialization.symbol_mangling_inputs
    specialization_hash = hash(specialization)

    nested["values"].append(2)
    nested["modes"].add("striped")

    assert specialization.template_arguments["settings"] == {
        "values": (1,),
        "modes": frozenset({"direct"}),
    }
    assert specialization.semantic_key == semantic_key
    assert specialization.symbol_mangling_inputs == symbol_inputs
    assert hash(specialization) == specialization_hash


def test_algorithm_specialization_rejects_container_cycles():
    algorithm = Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=(),
        template_parameters=(TemplateParameter("T"),),
        parameters=((),),
    )
    cyclic = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="container cycles"):
        algorithm.specialize({"T": "int", "settings": cyclic})


def test_parameter_classification_distinguishes_static_runtime_and_storage():
    op = PythonOperator(
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"), Dependency("T")),
        op=Dependency("Op"),
        name="op",
    )
    stateful = StatefulOperator(
        op=lambda left, right: left + right,
        state_dtype="state_t",
        ret_dtype="int",
        arg_dtypes=("int", "int"),
        name="stateful_op",
    )
    algorithm = Algorithm(
        "BlockExample",
        "Run",
        "block_example",
        (),
        (TemplateParameter("T"),),
        (
            (
                TempStorageParameter(UINT8),
                Value(Dependency("T"), name="value"),
                Value(Dependency("T"), name="output", is_output=True),
                Array(
                    Dependency("T"),
                    4,
                    name="values",
                    is_inout=True,
                ),
                CxxFunction("7", INT32, name="constant"),
                op,
                stateful,
            ),
        ),
    )
    specialization = algorithm.specialize({"T": "int", "Op": object()})

    classifications = specialization.classify_method()
    assert [(item.kind, item.role) for item in classifications] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.RUNTIME, ParameterRole.INOUT),
        (ArgumentKind.STATIC, ParameterRole.CONSTANT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
        (ArgumentKind.RUNTIME, ParameterRole.STATE),
    ]


def test_semantic_token_distinguishes_callable_closures():
    def make_op(offset):
        return lambda left, right: left + right + offset

    first = semantic_token(make_op(1))
    second = semantic_token(make_op(2))

    assert first != second
    assert first == semantic_token(make_op(1))

    def with_default(*, offset=1):
        return offset

    original = semantic_token(with_default)
    with_default.__kwdefaults__ = {"offset": 2}
    assert original != semantic_token(with_default)

    assert semantic_token(partial(pow, 2)) != semantic_token(partial(pow, 3))

    class Offset:
        def __init__(self, offset):
            self.offset = offset

        def __call__(self, value):
            return value + self.offset

    assert semantic_token(Offset(1)) != semantic_token(Offset(2))

    def make_stateless_callable(offset):
        class StatelessOffset:
            __slots__ = ()

            def __call__(self, value):
                return value + offset

        return StatelessOffset()

    assert semantic_token(make_stateless_callable(1)) != semantic_token(
        make_stateless_callable(2)
    )

    class StatelessDefaults:
        __slots__ = ()

        def __call__(self, value, offset=1, *, scale=1):
            return value * scale + offset

    stateless = StatelessDefaults()
    default_token = semantic_token(stateless)
    StatelessDefaults.__call__.__defaults__ = (2,)
    assert default_token != semantic_token(stateless)
    keyword_token = semantic_token(stateless)
    StatelessDefaults.__call__.__kwdefaults__ = {"scale": 2}
    assert keyword_token != semantic_token(stateless)


def test_semantic_token_tracks_private_slotted_callable_state():
    class Offset:
        __slots__ = ("__offset",)

        def __init__(self, offset):
            self.__offset = offset

        def __call__(self, value):
            return value + self.__offset

    assert semantic_token(Offset(1)) != semantic_token(Offset(2))


def test_semantic_token_tracks_class_dependencies_in_callable_state(monkeypatch):
    class Helper:
        @staticmethod
        def apply(value):
            return value + 1

    class StatefulPolicy:
        def __init__(self):
            self.helper = Helper

        def __call__(self, value):
            return self.helper.apply(value)

    class SlottedStatefulPolicy:
        __slots__ = ("helper",)

        def __init__(self):
            self.helper = Helper

        def __call__(self, value):
            return self.helper.apply(value)

    operations = (StatefulPolicy(), SlottedStatefulPolicy())
    originals = tuple(semantic_token(operation) for operation in operations)
    monkeypatch.setattr(
        Helper,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    assert originals != tuple(semantic_token(operation) for operation in operations)


def test_semantic_token_tracks_only_referenced_globals(monkeypatch):
    original = semantic_token(_global_dependent_operator)

    monkeypatch.setitem(
        _global_dependent_operator.__globals__,
        "_UNRELATED_SEMANTIC_GLOBAL",
        2,
    )
    assert semantic_token(_global_dependent_operator) == original

    monkeypatch.setitem(
        _global_dependent_operator.__globals__,
        "_REFERENCED_SEMANTIC_GLOBAL",
        2,
    )
    assert semantic_token(_global_dependent_operator) != original


def test_semantic_token_handles_recursive_callables():
    def recurse(value):
        return value if value <= 0 else recurse(value - 1)

    def even(value):
        return value == 0 or odd(value - 1)

    def odd(value):
        return value != 0 and even(value - 1)

    assert semantic_token(recurse) == semantic_token(recurse)
    assert semantic_token(even) == semantic_token(even)


def test_semantic_token_keeps_memoized_values_alive_during_traversal():
    @dataclass
    class DynamicFields:
        first: object = None
        second: object = None

        def __getattribute__(self, name):
            if name == "first":
                return [1]
            if name == "second":
                return [2]
            return object.__getattribute__(self, name)

    token = semantic_token(DynamicFields())
    field_tokens = dict(token[2])

    assert field_tokens["first"] != field_tokens["second"]


def test_semantic_token_tracks_class_dependencies_in_nested_defaults(monkeypatch):
    class DefaultPolicy:
        @staticmethod
        def apply(value):
            return value + 1

    default_settings = {"policies": (DefaultPolicy,)}

    def default_op(value, settings=default_settings):
        return settings["policies"][0].apply(value)

    original = semantic_token(default_op)

    monkeypatch.setattr(
        DefaultPolicy,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    assert original != semantic_token(default_op)


def test_semantic_token_tracks_class_dependencies_in_closures(monkeypatch):
    class ClosurePolicy:
        @staticmethod
        def apply(value):
            return value + 1

    policy = ClosurePolicy

    def closure_op(value):
        return policy.apply(value)

    original = semantic_token(closure_op)

    monkeypatch.setattr(
        ClosurePolicy,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    assert original != semantic_token(closure_op)


def test_semantic_token_tracks_class_valued_class_members(monkeypatch):
    class Helper:
        @staticmethod
        def apply(value):
            return value + 1

    class Policy:
        helper = Helper

    def member_op(value, policy=Policy):
        return policy.helper.apply(value)

    original = semantic_token(member_op)

    monkeypatch.setattr(
        Helper,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    assert original != semantic_token(member_op)


def test_semantic_token_shallows_standard_library_type_dependencies():
    assert _type_reference_token(Enum, _TokenState()) == (
        "type",
        "enum",
        "Enum",
    )
    assert _type_reference_token(type(lambda: None), _TokenState()) == (
        "type",
        "builtins",
        "function",
    )


def test_semantic_token_expands_heap_types_with_stdlib_metadata():
    class SpoofedType:
        offset = 1

    SpoofedType.__module__ = "builtins"
    original = _type_reference_token(SpoofedType, _TokenState())
    SpoofedType.offset = 2

    assert original != _type_reference_token(SpoofedType, _TokenState())


def test_semantic_token_shallows_verified_standard_library_functions():
    assert semantic_token(enum.unique) == (
        "callable-reference",
        "enum",
        "unique",
    )


def test_semantic_token_expands_user_wrappers_with_stdlib_metadata():
    @wraps(enum.unique)
    def increment(value):
        return value + 1

    @wraps(enum.unique)
    def decrement(value):
        return value - 1

    increment_token = semantic_token(increment)
    decrement_token = semantic_token(decrement)

    assert increment_token[0] == "callable"
    assert decrement_token[0] == "callable"
    assert increment_token != decrement_token


def test_semantic_token_expands_user_modules_with_stdlib_names(
    monkeypatch,
    tmp_path,
):
    module_name = "types.cuda_coop_token_test"
    module = ModuleType(module_name)
    source_path = tmp_path / "types" / "cuda_coop_token_test.py"
    module.__file__ = str(source_path)
    exec(
        compile(
            textwrap.dedent(
                """
                class Policy:
                    offset = 1

                def apply(value):
                    return value + Policy.offset
                """
            ),
            str(source_path),
            "exec",
        ),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, module_name, module)

    original = semantic_token(module.apply)
    module.Policy.offset = 2

    assert original != semantic_token(module.apply)


def test_semantic_token_tracks_function_metadata():
    def add(value):
        return value + 1

    def subtract(value):
        return value - 1

    def wrapper(value):
        return value

    wrapper.__wrapped__ = add
    original = semantic_token(wrapper)
    wrapper.__wrapped__ = subtract
    changed_wrapped = semantic_token(wrapper)
    assert original != changed_wrapped

    wrapper.policy = {"offset": 1}
    with_policy = semantic_token(wrapper)
    wrapper.policy["offset"] = 2

    assert changed_wrapped != with_policy
    assert with_policy != semantic_token(wrapper)


def test_semantic_token_preserves_malformed_callable_metadata(monkeypatch):
    class CallableWithMalformedMetadata:
        __code__ = "code-a"
        __globals__ = "globals-a"
        __closure__ = "closure-a"

        def __call__(self, value):
            return value

    op = CallableWithMalformedMetadata()
    original = semantic_token(op)

    monkeypatch.setattr(CallableWithMalformedMetadata, "__code__", "code-b")
    changed_code = semantic_token(op)
    assert original != changed_code

    monkeypatch.setattr(
        CallableWithMalformedMetadata,
        "__globals__",
        "globals-b",
    )
    changed_globals = semantic_token(op)
    assert changed_code != changed_globals

    monkeypatch.setattr(
        CallableWithMalformedMetadata,
        "__closure__",
        "closure-b",
    )

    assert changed_globals != semantic_token(op)


def test_semantic_token_validates_metadata_with_exposed_code(monkeypatch):
    class CallableWithExposedCode:
        __code__ = (lambda: None).__code__
        __globals__ = "globals-a"
        __closure__ = "closure-a"

        def __call__(self, value):
            return value

    op = CallableWithExposedCode()
    original = semantic_token(op)

    monkeypatch.setattr(CallableWithExposedCode, "__globals__", "globals-b")
    changed_globals = semantic_token(op)
    assert original != changed_globals

    monkeypatch.setattr(CallableWithExposedCode, "__closure__", "closure-b")

    assert changed_globals != semantic_token(op)


def test_semantic_token_handles_non_string_callable_module():
    class CallableWithoutStringModule:
        __module__ = []

        def __call__(self, value):
            return value

    op = CallableWithoutStringModule()
    token = semantic_token(op)

    assert token == semantic_token(op)
    hash(token)


def test_semantic_token_expands_types_without_string_modules(monkeypatch):
    class Policy:
        @staticmethod
        def apply(value):
            return value + 1

    Policy.__module__ = None

    def apply(value, policy=Policy):
        return policy.apply(value)

    original = semantic_token(apply)
    monkeypatch.setattr(Policy, "apply", staticmethod(lambda value: value - 1))

    assert original != semantic_token(apply)


def test_semantic_token_tracks_referenced_closure_attributes(monkeypatch):
    class ClosureSettings:
        offset = 1

    settings = ClosureSettings()

    def closure_op(value):
        return value + settings.offset

    original = semantic_token(closure_op)

    monkeypatch.setattr(ClosureSettings, "offset", 2)

    assert original != semantic_token(closure_op)


def test_semantic_token_tolerates_mismatched_closure_metadata(monkeypatch):
    class ClosurePolicy:
        @staticmethod
        def apply(value):
            return value + 1

    def capture(value):
        return lambda: value

    class CallableWithMismatchedClosure:
        __code__ = (lambda: None).__code__
        __closure__ = capture(ClosurePolicy).__closure__
        __globals__ = {}

        def __call__(self, value):
            return ClosurePolicy.apply(value)

    op = CallableWithMismatchedClosure()
    original = semantic_token(op)

    monkeypatch.setattr(
        ClosurePolicy,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    changed_dependency = semantic_token(op)
    assert original != changed_dependency

    monkeypatch.setattr(
        CallableWithMismatchedClosure,
        "__call__",
        lambda self, value: value - 2,
    )

    assert changed_dependency != semantic_token(op)


def test_semantic_token_tolerates_non_cell_closure_metadata():
    class CallableWithInvalidClosure:
        __code__ = (lambda: None).__code__
        __closure__ = ("not-a-cell",)
        __globals__ = {}

        def __call__(self, value):
            return value

    op = CallableWithInvalidClosure()
    assert semantic_token(op) == semantic_token(op)


def test_semantic_token_tracks_closure_attributes_in_nested_code(monkeypatch):
    class GeneratorSettings:
        offset = 1

    settings = GeneratorSettings()

    def generator_op(values):
        return sum(value + settings.offset for value in values)

    original = semantic_token(generator_op)

    monkeypatch.setattr(GeneratorSettings, "offset", 2)

    assert original != semantic_token(generator_op)


def test_semantic_token_tracks_custom_descriptor_implementations(monkeypatch):
    class OffsetDescriptor:
        def __get__(self, instance, owner):
            del instance, owner
            return 1

    class DescriptorPolicy:
        offset = OffsetDescriptor()

    def descriptor_op(value, policy=DescriptorPolicy):
        return value + policy.offset

    original = semantic_token(descriptor_op)

    monkeypatch.setattr(
        OffsetDescriptor,
        "__get__",
        lambda self, instance, owner: 2,
    )

    assert original != semantic_token(descriptor_op)


def test_semantic_token_tracks_custom_metaclass_implementations(monkeypatch):
    class PolicyMeta(type):
        def __call__(cls, value):
            del cls
            return value + 1

    class MetaclassPolicy(metaclass=PolicyMeta):
        pass

    def metaclass_op(value, policy=MetaclassPolicy):
        return policy(value)

    original = semantic_token(metaclass_op)

    monkeypatch.setattr(
        PolicyMeta,
        "__call__",
        lambda cls, value: value - 1,
    )

    assert original != semantic_token(metaclass_op)


def test_semantic_token_tracks_referenced_global_values(monkeypatch):
    original = semantic_token(_token_global_value_op)

    monkeypatch.setitem(globals(), "_TOKEN_GLOBAL_OFFSET", 2)

    assert original != semantic_token(_token_global_value_op)


def test_semantic_token_ignores_attribute_names_that_match_unread_globals(
    monkeypatch,
):
    original = semantic_token(_token_attribute_op)

    monkeypatch.setitem(globals(), "offset", 200)

    assert original == semantic_token(_token_attribute_op)


def test_semantic_token_tracks_referenced_global_attributes(monkeypatch):
    holder_original = semantic_token(_token_attribute_op)
    module_original = semantic_token(_token_module_attribute_op)

    monkeypatch.setattr(_TokenGlobalHolder, "offset", 8)
    monkeypatch.setattr(_TOKEN_GLOBAL_MODULE, "offset", 12)

    assert holder_original != semantic_token(_token_attribute_op)
    assert module_original != semantic_token(_token_module_attribute_op)


def test_semantic_token_tracks_referenced_global_class_bodies(monkeypatch):
    policy_original = semantic_token(_token_global_policy_op)
    static_original = semantic_token(_token_global_static_policy_op)

    monkeypatch.setattr(_TokenGlobalPolicy, "result", lambda self: self.value - 1)
    monkeypatch.setattr(
        _TokenGlobalStaticPolicy,
        "apply",
        staticmethod(lambda value: value - 1),
    )

    assert policy_original != semantic_token(_token_global_policy_op)
    assert static_original != semantic_token(_token_global_static_policy_op)


def test_semantic_token_tracks_rebound_global_helpers(monkeypatch):
    original = semantic_token(_token_global_helper_op)

    monkeypatch.setitem(globals(), "_TOKEN_GLOBAL_HELPER", _token_helper_subtract)

    assert original != semantic_token(_token_global_helper_op)


def test_semantic_token_handles_recursive_global_callables():
    assert semantic_token(_token_recursive_op) == semantic_token(_token_recursive_op)


def test_semantic_token_handles_recursive_values():
    @dataclass
    class Node:
        child: object | None = None

    class State:
        pass

    first_list = []
    first_list.append(first_list)
    second_list = []
    second_list.append(second_list)

    first_mapping = {}
    first_mapping["self"] = first_mapping
    second_mapping = {}
    second_mapping["self"] = second_mapping

    first_node = Node()
    first_node.child = first_node
    second_node = Node()
    second_node.child = second_node

    first_state = State()
    first_state.self = first_state
    second_state = State()
    second_state.self = second_state

    for first, second in (
        (first_list, second_list),
        (first_mapping, second_mapping),
        (first_node, second_node),
        (first_state, second_state),
    ):
        assert semantic_token(first) == semantic_token(second)


def test_semantic_token_tracks_container_subclass_state():
    class TaggedDict(dict):
        pass

    class TaggedList(list):
        pass

    class TaggedTuple(tuple):
        pass

    class TaggedSet(set):
        pass

    for container_type, contents in (
        (TaggedDict, {"value": 1}),
        (TaggedList, [1]),
        (TaggedTuple, (1,)),
        (TaggedSet, {1}),
    ):
        first = container_type(contents)
        first.tag = "first"
        second = container_type(contents)
        second.tag = "second"
        assert semantic_token(first) != semantic_token(second)


def test_semantic_token_tracks_defaultdict_factory():
    first = defaultdict(list, {"value": 1})
    equivalent = defaultdict(list, {"value": 1})
    different = defaultdict(set, {"value": 1})

    assert semantic_token(first) == semantic_token(equivalent)
    assert semantic_token(first) != semantic_token(different)


def test_semantic_token_handles_recursive_container_subclass_state():
    class RecursiveDict(dict):
        pass

    first = RecursiveDict(value=1)
    first.owner = first
    second = RecursiveDict(value=1)
    second.owner = second

    first_token = semantic_token(first)
    assert first_token == semantic_token(second)
    hash(first_token)


def test_semantic_token_tracks_container_state_in_callable_defaults():
    def with_default(value, table=defaultdict(list, {"value": 1})):
        return table[value]

    original = semantic_token(with_default)
    with_default.__defaults__ = (defaultdict(set, {"value": 1}),)

    assert original != semantic_token(with_default)


def test_semantic_token_distinguishes_cycle_topology():
    outer_cycle = []
    inner_cycle = []
    outer_cycle.append(inner_cycle)
    inner_cycle.append(outer_cycle)

    outer_self = []
    inner_self = []
    outer_self.append(inner_self)
    inner_self.append(inner_self)

    assert semantic_token(outer_cycle) != semantic_token(outer_self)


def test_semantic_token_memoizes_acyclic_diamonds():
    class CountingMapping(dict):
        visits = 0

        def items(self):
            self.visits += 1
            return super().items()

    nodes = [CountingMapping(value=1)]
    for _ in range(12):
        nodes.append(CountingMapping(left=nodes[-1], right=nodes[-1]))

    semantic_token(nodes[-1])

    assert [node.visits for node in nodes] == [1] * len(nodes)


def test_semantic_token_memoization_is_mapping_order_independent():
    def make_shared_cycle(order):
        first = []
        second = []
        shared = [first]
        first.append(shared)
        second.append(shared)
        nodes = {"first": first, "second": second}
        return {name: nodes[name] for name in order}

    forward = make_shared_cycle(("first", "second"))
    reverse = make_shared_cycle(("second", "first"))

    assert semantic_token(forward) == semantic_token(reverse)


def test_semantic_token_namespaces_string_enums():
    assert semantic_token(ArgumentKind.STATIC) != semantic_token("static")
    assert semantic_token(ParameterRole.STATE) != semantic_token("state")


def test_semantic_token_preserves_container_kinds():
    assert semantic_token((1, 2)) != semantic_token([1, 2])
    assert semantic_token({1, 2}) != semantic_token(frozenset({1, 2}))
    assert semantic_token({"key": 1}) != semantic_token([("key", 1)])

    def with_default(value=(1, 2)):
        return value

    tuple_default = semantic_token(with_default)
    with_default.__defaults__ = ([1, 2],)
    assert tuple_default != semantic_token(with_default)


def test_semantic_token_normalizes_nan_values():
    assert semantic_token(float("nan")) == semantic_token(float("nan"))


def test_semantic_token_preserves_signed_zero():
    assert semantic_token(0.0) != semantic_token(-0.0)


def test_semantic_token_preserves_float_type_and_value():
    assert semantic_token(1.0) == semantic_token(float(1.0))
    assert semantic_token(1.0) != semantic_token(1)


def test_semantic_token_for_nested_code_is_process_stable():
    script = textwrap.dedent(
        """
        from cuda.coop._core import semantic_token

        def outer(scale=2):
            def inner(value):
                return value * scale
            return [inner(value) for value in range(3)]

        class Opaque:
            __slots__ = ()

        print(repr((semantic_token(outer), semantic_token(Opaque()))))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    outputs = [
        subprocess.run(
            [sys.executable, "-S", "-B", "-c", script],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        ).stdout
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]


def test_core_backend_adapter_protocol_is_runtime_checkable():
    class FakeAdapter:
        def normalize_dtype(self, dtype):
            return dtype

        def cpp_type(self, dtype):
            return str(dtype)

        def lower_parameter(self, parameter, *, specialization):
            return parameter, specialization

        def lower_cxx_operator(self, operator, *, specialization):
            return operator, specialization

        def lower_python_operator(self, operator, *, specialization):
            return operator, specialization

        def lower_stateful_operator(self, operator, *, specialization):
            return operator, specialization

        def lower_temp_storage(self, parameter, *, specialization):
            return parameter, specialization

        def materialize(self, specialization, **kwargs):
            return specialization, kwargs

    assert isinstance(FakeAdapter(), CoreBackendAdapter)
