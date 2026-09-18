# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from contextvars import ContextVar
from types import ModuleType

import pytest

from cuda import coop
from cuda.coop._core.api import _dispatch
from cuda.coop._core.thread_group import CoopCompilerContextRequiredError


@pytest.fixture
def compiler_environment(monkeypatch):
    environment = ContextVar("test_compiler_environment", default=None)
    monkeypatch.setattr(_dispatch, "_COMPILER_CONTEXT_PROBES", {})
    for name in ("first", "second"):
        module = ModuleType(f"test_backend_{name}")
        module.ThreadData = lambda *args, _name=name, **kwargs: _name
        monkeypatch.setitem(sys.modules, module.__name__, module)
        _dispatch._register_compiler_context_probe(
            module.__name__, lambda _name=name: environment.get() == _name
        )
    return environment


def test_probe_selects_only_the_current_compiler(compiler_environment):
    assert coop.this_block().kind == "block"
    with pytest.raises(CoopCompilerContextRequiredError):
        coop.ThreadData(1)

    first = compiler_environment.set("first")
    try:
        assert coop.ThreadData(1) == "first"
        second = compiler_environment.set("second")
        try:
            assert coop.ThreadData(1) == "second"
        finally:
            compiler_environment.reset(second)
        assert coop.ThreadData(1) == "first"
    finally:
        compiler_environment.reset(first)

    with pytest.raises(CoopCompilerContextRequiredError):
        coop.ThreadData(1)


def test_explicit_compiler_scope_takes_precedence(compiler_environment):
    token = compiler_environment.set("first")
    try:
        with _dispatch._compiler_scope("test_backend_second"):
            assert coop.ThreadData(1) == "second"
        assert coop.ThreadData(1) == "first"
    finally:
        compiler_environment.reset(token)


def test_competing_compiler_environments_fail_closed(compiler_environment):
    _dispatch._register_compiler_context_probe("test_backend_second", lambda: True)
    token = compiler_environment.set("first")
    try:
        with pytest.raises(CoopCompilerContextRequiredError, match="Multiple backends"):
            coop.ThreadData(1)
    finally:
        compiler_environment.reset(token)


def test_probe_failure_preserves_cause_and_explicit_scope(compiler_environment):
    failure = RuntimeError("compiler environment is unavailable")

    def failed_probe():
        raise failure

    _dispatch._register_compiler_context_probe("test_backend_first", failed_probe)
    with pytest.raises(CoopCompilerContextRequiredError) as caught:
        coop.ThreadData(1)
    assert caught.value.__cause__ is failure
    with _dispatch._compiler_scope("test_backend_second"):
        assert coop.ThreadData(1) == "second"
