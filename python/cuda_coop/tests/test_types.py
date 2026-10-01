# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import struct
from dataclasses import replace

import pytest

from cuda.coop._core import (
    Algorithm,
    ArgumentKind,
    Dependency,
    TemplateParameter,
    Value,
    semantic_token,
)


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


@pytest.mark.parametrize("value", (False, True))
def test_algorithm_identity_distinguishes_boolean_and_integer_settings(value):
    algorithm = _algorithm()
    boolean = algorithm.specialize({"T": "int", "settings": {"flag": (value,)}})
    integer = algorithm.specialize(
        {"T": "int", "settings": {"flag": (int(value),)}}
    )

    assert len({boolean: "boolean", integer: "integer"}) == 2


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


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (ArgumentKind.STATIC, "static"),
        (False, 0),
        (True, 1),
        (0.0, -0.0),
        (1.0, 1),
        (float("nan"), -float("nan")),
        (
            struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0],
            struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0],
        ),
    ],
)
def test_semantic_token_preserves_scalar_identity(left, right):
    assert semantic_token(left) != semantic_token(right)
