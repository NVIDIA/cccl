# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for wrapper name handling and sanitize_identifier.

sanitize_identifier replaces non-alphanumeric/underscore characters with
underscores before names reach exec(). _make_wrapper_name then validates that
the *sanitized* name is a valid identifier (e.g. not empty, not
leading-digit-only) before building the generated wrapper's symbol name.
"""

import pytest

from cuda.compute._odr_helpers import _make_wrapper_name
from cuda.compute._utils import sanitize_identifier

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
# _make_wrapper_name — names that sanitize to a valid identifier are OK
# ---------------------------------------------------------------------------


def test_lambda_name_is_accepted():
    """Lambdas have __name__ == '<lambda>'; sanitizes to '_lambda_'."""
    name = _make_wrapper_name("<lambda>")
    assert name.isidentifier()
    assert "_lambda_" in name


def test_newline_in_name_is_accepted():
    """Newlines sanitize to underscores — must not raise."""
    name = _make_wrapper_name("foo\nbar")
    assert name.isidentifier()
    assert "foo_bar" in name


def test_plain_name_is_accepted():
    name = _make_wrapper_name("my_op")
    assert name.isidentifier()
    assert "my_op" in name


def test_generated_names_are_unique():
    """The global counter disambiguates repeated uses of the same name."""
    assert _make_wrapper_name("my_op") != _make_wrapper_name("my_op")


# ---------------------------------------------------------------------------
# _make_wrapper_name — names that sanitize to an invalid identifier
# ---------------------------------------------------------------------------


def test_empty_name_is_rejected():
    """Empty string sanitizes to empty string, which is not a valid identifier."""
    with pytest.raises(ValueError, match="cannot be sanitized into a valid identifier"):
        _make_wrapper_name("")


def test_digits_only_name_is_rejected():
    """'123' sanitizes to '123', which is not a valid identifier (leading digit)."""
    with pytest.raises(ValueError, match="cannot be sanitized into a valid identifier"):
        _make_wrapper_name("123")
