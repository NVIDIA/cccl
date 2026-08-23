# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import os
import subprocess
import sys
import textwrap
from functools import partial

import pytest

from ...support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

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
        (("T", "int"), ("N", 4)),
    )
    equivalent = algorithm.specialize(
        {"T": "int", "N": 4},
        metadata={"scope": "block", "primitive": "example"},
    )
    assert specialization == equivalent
    assert hash(specialization) == hash(equivalent)

    with pytest.raises(TypeError):
        specialization.template_arguments["N"] = 8


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


def test_semantic_token_namespaces_string_enums():
    assert semantic_token(ArgumentKind.STATIC) != semantic_token("static")
    assert semantic_token(ParameterRole.STATE) != semantic_token("state")


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
