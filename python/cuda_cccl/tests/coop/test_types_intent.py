# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._types import TempStorage, ThreadData


def test_temp_storage_request_normalizes_sharing():
    temp_storage = TempStorage(sharing=" Shared ")
    request = temp_storage.to_request()

    assert request.sharing == "shared"


def test_temp_storage_explicit_flags():
    temp_storage = TempStorage(size_in_bytes=64, alignment=16, auto_sync=False)

    assert temp_storage.has_explicit_size is True
    assert temp_storage.has_explicit_alignment is True
    assert temp_storage.has_explicit_auto_sync is True


def test_thread_data_request_preserves_items_and_dtype():
    thread_data = ThreadData(4, dtype="float32")
    request = thread_data.to_request()

    assert request.items_per_thread == 4
    assert request.dtype == "float32"
    assert thread_data.has_explicit_dtype is True


def test_thread_data_requires_positive_items_per_thread():
    with pytest.raises(ValueError, match="items_per_thread must be a positive integer"):
        ThreadData(0)
