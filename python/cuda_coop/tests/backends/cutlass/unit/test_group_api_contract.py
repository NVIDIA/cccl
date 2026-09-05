# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from inspect import Parameter, signature

import pytest

import cuda.coop.cutlass as coop

from ....support.cases.api_contracts import (
    FULL_GROUP_FIRST_EXPORTS,
    GROUP_CONSTRUCTOR_SIGNATURES,
    GROUP_METHOD_SIGNATURES,
    PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
    QUALIFIED_GROUP_PRIMITIVE_SUFFIXES,
    REQUIRED_PARAMETER,
    portable_group_primitive_parameter_contract,
)


def _parameter_contract(member):
    result = []
    for parameter in signature(member).parameters.values():
        default = parameter.default
        if default is Parameter.empty:
            default = REQUIRED_PARAMETER
        elif isinstance(default, Enum):
            default = default.value
        result.append((parameter.name, parameter.kind.name, default))
    return tuple(result)


@pytest.mark.parametrize("name", GROUP_CONSTRUCTOR_SIGNATURES)
def test_group_constructor_signatures_match_contract(name):
    assert str(signature(getattr(coop, name))) == GROUP_CONSTRUCTOR_SIGNATURES[name]


@pytest.mark.parametrize("name", GROUP_METHOD_SIGNATURES)
def test_group_method_signatures_match_contract(name):
    assert (
        str(signature(getattr(coop.ThreadGroup, name))) == GROUP_METHOD_SIGNATURES[name]
    )


@pytest.mark.parametrize("name", PORTABLE_GROUP_PRIMITIVE_SIGNATURES)
def test_group_primitive_signatures_match_contract(name):
    suffix = QUALIFIED_GROUP_PRIMITIVE_SUFFIXES["cutlass"].get(name, ())

    assert _parameter_contract(getattr(coop, name)) == (
        portable_group_primitive_parameter_contract(name, suffix=suffix)
    )


def test_group_first_exports_match_contract():
    assert set(FULL_GROUP_FIRST_EXPORTS).issubset(coop.__all__)
