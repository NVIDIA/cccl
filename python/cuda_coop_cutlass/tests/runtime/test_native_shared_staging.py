# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check native masked staging and its shared-memory consumers."""

import importlib.util

import pytest

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("width", (8, 64))
@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("striped", (False, True))
@pytest.mark.parametrize("tail", (0, 3))
def test_native_shared_staging(width, items_per_thread, striped, tail):
    path = PACKAGE_ROOT / "examples/shared_staging.py"
    spec = importlib.util.spec_from_file_location("shared_staging", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(
        width=width,
        items_per_thread=items_per_thread,
        striped=striped,
        tail=tail,
    )


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_block_storage_example(sharing, manual_sync, items_per_thread):
    path = PACKAGE_ROOT / "examples/block_storage.py"
    spec = importlib.util.spec_from_file_location("block_storage", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(
        sharing=sharing,
        manual_sync=manual_sync,
        items_per_thread=items_per_thread,
    )
