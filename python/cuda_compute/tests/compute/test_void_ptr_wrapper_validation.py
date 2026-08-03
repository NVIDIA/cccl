# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for _create_void_ptr_wrapper name handling and sanitize_identifier.

sanitize_identifier replaces non-alphanumeric/underscore characters with
underscores before names reach exec(). _create_void_ptr_wrapper then validates
that the *sanitized* name is a valid identifier (e.g. not empty, not
leading-digit-only).
"""

import pytest
from numba import types

from cuda.compute._odr_helpers import _ArgMode, _ArgSpec, _create_void_ptr_wrapper
from cuda.compute._utils import sanitize_identifier


def _make_arg_specs():
    """One float32 input, one float32 output."""
    return [
        _ArgSpec(types.float32, _ArgMode.LOAD),
        _ArgSpec(types.float32, _ArgMode.STORE),
    ]


def _make_inner_sig():
    return types.float32(types.float32)


def _passthrough(x):
    return x


# ---------------------------------------------------------------------------
# sanitize_identifier — the exec() injection boundary
# ---------------------------------------------------------------------------


def test_sanitize_lambda_name():
    assert sanitize_identifier("<lambda>") == "_lambda_"


def test_sanitize_newline():
    assert sanitize_identifier("foo\nbar") == "foo_bar"


def test_sanitize_space():
    assert sanitize_identifier("my func") == "my_func"


def test_sanitize_hyphen():
    assert sanitize_identifier("my-func") == "my_func"


def test_sanitize_semicolon_injection():
    assert sanitize_identifier("f; import os") == "f__import_os"


def test_sanitize_plain_name_unchanged():
    assert sanitize_identifier("my_op") == "my_op"


# ---------------------------------------------------------------------------
# _create_void_ptr_wrapper — names that sanitize to a valid identifier are OK
# ---------------------------------------------------------------------------


def test_lambda_name_is_accepted():
    """Lambdas have __name__ == '<lambda>'; sanitizes to '_lambda_'."""
    op = lambda x: x  # noqa: E731
    _create_void_ptr_wrapper(op, op.__name__, _make_arg_specs(), _make_inner_sig())


def test_newline_in_name_is_accepted():
    """Newlines sanitize to underscores — must not raise."""
    _create_void_ptr_wrapper(
        _passthrough, "foo\nbar", _make_arg_specs(), _make_inner_sig()
    )


def test_plain_name_is_accepted():
    _create_void_ptr_wrapper(
        _passthrough, "my_op", _make_arg_specs(), _make_inner_sig()
    )


# ---------------------------------------------------------------------------
# _create_void_ptr_wrapper — names that sanitize to an invalid identifier
# ---------------------------------------------------------------------------


def test_empty_name_is_rejected():
    """Empty string sanitizes to empty string, which is not a valid identifier."""
    with pytest.raises(ValueError, match="cannot be sanitized into a valid identifier"):
        _create_void_ptr_wrapper(_passthrough, "", _make_arg_specs(), _make_inner_sig())


def test_digits_only_name_is_rejected():
    """'123' sanitizes to '123', which is not a valid identifier (leading digit)."""
    with pytest.raises(ValueError, match="cannot be sanitized into a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "123", _make_arg_specs(), _make_inner_sig()
        )
