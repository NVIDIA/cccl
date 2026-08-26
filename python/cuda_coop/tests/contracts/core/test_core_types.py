# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

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


def test_semantic_token_namespaces_string_enums():
    assert semantic_token(ArgumentKind.STATIC) != semantic_token("static")
    assert semantic_token(ParameterRole.STATE) != semantic_token("state")


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
