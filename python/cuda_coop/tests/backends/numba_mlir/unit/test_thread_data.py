# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

import numpy as np
import pytest

import cuda.coop.numba_mlir as coop

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _StringSharing(str, Enum):
    SHARED = "shared"


@pytest.mark.parametrize(
    "call",
    [
        lambda: coop.ThreadData(shape=4),
        lambda: coop.ThreadData(4, address_space="local"),
        lambda: coop.ThreadData(4, items_per_thread=4),
    ],
)
def test_thread_data_rejects_unknown_or_duplicate_arguments(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("alignment", [None, 1, 2, 4, 8, 16, np.int64(32)])
def test_temp_storage_accepts_minimum_alignment(alignment):
    storage = coop.TempStorage(alignment=alignment)
    assert storage.alignment == alignment
    assert storage.alignment is None or type(storage.alignment) is int


def test_temp_storage_uses_canonical_defaults_and_normalization():
    shared = coop.TempStorage(sharing=" SHARED ")
    exclusive = coop.TempStorage(sharing=" Exclusive ")

    assert shared.sharing == "shared"
    assert shared.auto_sync is False
    assert exclusive.sharing == "exclusive"
    assert exclusive.auto_sync is False
    assert coop.TempStorage(auto_sync=None).auto_sync is False
    assert coop.TempStorage(sharing="exclusive", auto_sync=True).auto_sync is True
    assert coop.TempStorage(sharing="exclusive", auto_sync=False).auto_sync is False


def test_temp_storage_rejects_string_enum_sharing():
    with pytest.raises(TypeError, match="sharing must be a string"):
        coop.TempStorage(sharing=_StringSharing.SHARED)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"size_in_bytes": True},
            TypeError,
            "TempStorage size_in_bytes must be an integer or None",
        ),
        (
            {"size_in_bytes": 0},
            ValueError,
            "TempStorage size_in_bytes must be a positive integer",
        ),
        (
            {"alignment": False},
            TypeError,
            "alignment must be an integer or None",
        ),
        (
            {"alignment": 3},
            ValueError,
            "alignment must be a power of 2",
        ),
        (
            {"auto_sync": 1},
            TypeError,
            "TempStorage auto_sync must be None/True/False",
        ),
        (
            {"sharing": 1},
            TypeError,
            "TempStorage sharing must be a string",
        ),
    ],
)
def test_temp_storage_validation(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        coop.TempStorage(**kwargs)
