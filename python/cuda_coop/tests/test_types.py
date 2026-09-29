# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from cuda.coop._core import (
    Algorithm,
    ArgumentKind,
    Dependency,
    TemplateParameter,
    Value,
    semantic_token,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]

_REFERENCED_SEMANTIC_GLOBAL = 1
_UNRELATED_SEMANTIC_GLOBAL = 1


def _global_dependent_operator(left, right):
    return left + right + _REFERENCED_SEMANTIC_GLOBAL


def _algorithm():
    return Algorithm(
        struct_name="BlockExample",
        method_name="Run",
        c_name="block_example",
        includes=(),
        template_parameters=(TemplateParameter("T"),),
        parameters=((Value(Dependency("T"), name="value"),),),
    )


def test_algorithm_identity_distinguishes_specializations():
    algorithm = _algorithm()
    base = algorithm.specialize({"T": "int"}, metadata={"mode": "base"})
    variants = (
        algorithm.specialize({"T": "float"}, metadata={"mode": "base"}),
        algorithm.specialize({"T": "int"}, metadata={"mode": "alternate"}),
        replace(algorithm, method_name="Other").specialize(
            {"T": "int"}, metadata={"mode": "base"}
        ),
    )

    cache = {base: "compiled"}
    equivalent = algorithm.specialize({"T": "int"}, metadata={"mode": "base"})
    assert cache[equivalent] == "compiled"
    for variant in variants:
        assert variant not in cache
        assert variant.symbol_mangling_inputs != base.symbol_mangling_inputs


def test_algorithm_specialization_freezes_nested_semantic_containers():
    nested = {"values": [1], "modes": {"direct"}}
    specialization = _algorithm().specialize({"T": "int", "settings": nested})
    equivalent = _algorithm().specialize({"T": "int", "settings": nested})
    cache = {specialization: "compiled"}

    nested["values"].append(2)
    nested["modes"].add("striped")

    assert cache[equivalent] == "compiled"
    assert specialization.template_arguments["settings"] == {
        "values": (1,),
        "modes": frozenset({"direct"}),
    }


def test_algorithm_specialization_rejects_container_cycles():
    cyclic = []
    cyclic.append(cyclic)

    with pytest.raises(ValueError, match="container cycles"):
        _algorithm().specialize({"T": "int", "settings": cyclic})


def test_semantic_token_distinguishes_callable_closures():
    def make_op(offset):
        return lambda left, right: left + right + offset

    assert semantic_token(make_op(1)) != semantic_token(make_op(2))
    assert semantic_token(make_op(1)) == semantic_token(make_op(1))


def test_semantic_token_tracks_callable_defaults():
    def with_default(*, offset=1):
        return offset

    original = semantic_token(with_default)
    with_default.__kwdefaults__ = {"offset": 2}
    assert original != semantic_token(with_default)


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


def test_semantic_token_distinguishes_callbacks_observing_nan_sign():
    def make_op(sign):
        captured = math.copysign(float("nan"), sign)

        def op(left, right):
            return left + right + math.copysign(1.0, captured)

        return op

    positive, negative = make_op(1.0), make_op(-1.0)
    assert positive(2.0, 3.0) == 6.0
    assert negative(2.0, 3.0) == 4.0
    assert semantic_token(positive) != semantic_token(negative)
    assert semantic_token(positive) == semantic_token(make_op(1.0))


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


@pytest.mark.parametrize(
    ("left", "right"),
    [(ArgumentKind.STATIC, "static"), (0.0, -0.0), (1.0, 1)],
)
def test_semantic_token_preserves_scalar_identity(left, right):
    assert semantic_token(left) != semantic_token(right)
