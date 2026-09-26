# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for gpu_struct field-name validation.

These tests verify that invalid field names are rejected before
they can reach the exec() calls in _jit.py.
"""

import numpy as np
import pytest

from cuda.compute import gpu_struct
from cuda.compute import types as compute_types


def test_newline_in_field_name_is_rejected():
    """Field names with newlines must be rejected — they are the exec() injection vector."""
    payload = "x\n    else: pass\nprint('injected')\ndef _absorb():\n    if True:\n        pass\n"
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({payload: np.float32, "y": np.float32})


def test_space_in_field_name_is_rejected():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({"field name": np.float32})


def test_hyphen_in_field_name_is_rejected():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({"field-name": np.float32})


def test_semicolon_in_field_name_is_rejected():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({"a; import os": np.float32})


def test_non_string_key_is_rejected():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({0: np.float32})


def test_empty_string_field_name_is_rejected():
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        gpu_struct({"": np.float32})


def test_valid_field_names_are_accepted():
    """Sanity check: ordinary identifiers must still work."""
    S = gpu_struct(
        {"x": np.float32, "y": np.float32, "_z": np.int32, "field1": np.int64}
    )
    obj = S(1.0, 2.0, 3, 4)
    assert obj.x == np.float32(1.0)
    assert obj.y == np.float32(2.0)
    assert obj._z == np.int32(3)
    assert obj.field1 == np.int64(4)


def test_unicode_identifier_is_accepted():
    """Python allows unicode identifiers; they should be valid field names."""
    S = gpu_struct({"α": np.float32, "β": np.float32})
    obj = S(1.0, 2.0)
    assert obj.α == np.float32(1.0)
    assert obj.β == np.float32(2.0)


@pytest.mark.parametrize("factory", [compute_types.from_numpy_dtype, gpu_struct])
def test_native_struct_layout_is_accepted(factory):
    dtype = np.dtype([("a", np.int32), ("b", np.float64)], align=True)
    out = factory(dtype)
    assert out.dtype == dtype
    assert out.dtype.itemsize == 16
    assert [out.dtype.fields[n][1] for n in out.dtype.names] == [0, 8]


def test_struct_layout_with_field_title_is_accepted():
    dtype = np.dtype(
        [(("alias_a", "a"), np.int32), ("b", np.float64)],
        align=True,
    )
    out = compute_types.from_numpy_dtype(dtype)
    assert out.dtype.names == dtype.names
    assert out.dtype.itemsize == dtype.itemsize
    assert [out.dtype.fields[n][1] for n in out.dtype.names] == [
        dtype.fields[n][1] for n in dtype.names
    ]


@pytest.mark.parametrize("factory", [compute_types.from_numpy_dtype, gpu_struct])
@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype(
            {
                "names": ["a", "b"],
                "formats": ["<i4", "<f4"],
                "offsets": [0, 8],
                "itemsize": 16,
            }
        ),
        np.dtype(
            {
                "names": ["a", "b"],
                "formats": ["<i4", "<f4"],
                "offsets": [0, 4],
                "itemsize": 16,
            }
        ),
        np.dtype(
            {
                "names": ["a", "b"],
                "formats": ["<i4", "<f4"],
                "offsets": [4, 0],
                "itemsize": 8,
            }
        ),
    ],
)
def test_incompatible_struct_layout_is_rejected(factory, dtype):
    with pytest.raises(ValueError, match="does not match cuda.compute native layout"):
        factory(dtype)
